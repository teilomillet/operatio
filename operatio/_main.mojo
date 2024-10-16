# operatio/_main.mojo

from python import Python, PythonObject
import os
from testing import assert_equal
from max.tensor import Tensor
from time import perf_counter_ns

from algorithm.functional import vectorize
from sys.info import simdbitwidth

def prepare_file_path(model_name: String, output_dir: String) -> String:
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    return os.path.join(output_dir, safe_model_name + ".bin")

def load_from_binary(path: String) -> Tensor[DType.float32]:
    start_time = perf_counter_ns()
    weights = Tensor[DType.float32].fromfile(path)
    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000
    print("Loaded existing weights in:", elapsed_time, "ms")
    return weights

def load_from_transformers_and_save(model_name: String, output_path: String) -> Tensor[DType.float32]:
    start_time = perf_counter_ns()
    
    transformers = Python.import_module("transformers")
    torch = Python.import_module("torch")
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    
    state_dict = model.state_dict()
    torch.save(state_dict, output_path)
    
    weights = Tensor[DType.float32].fromfile(output_path)
    
    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000
    print("Loaded from transformers, saved and loaded in:", elapsed_time, "ms")
    
    return weights

def load_and_save_model(model_name: String, output_dir: String) -> Tensor[DType.float32]:
    output_path = prepare_file_path(model_name, output_dir)

    if os.path.exists(output_path):
        print("Binary file already exists at:", output_path)
        return load_from_binary(output_path)

    print("Loading model:", model_name)
    return load_from_transformers_and_save(model_name, output_path)


fn diff_weights(base: Tensor[DType.float32], ft: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    start_time = perf_counter_ns()
    assert_equal(base.shape(), ft.shape(), "Tensors must have the same shape for subtraction.")
    output = Tensor[DType.float32](base.shape())
    num_elements = base.num_elements()
    alias simd_width = 4  # Adjust as needed

    @parameter
    fn vec_subtract[simd_width: Int](i: Int):
        base_vector = base.load[simd_width](i)
        ft_vector = ft.load[simd_width](i)
        result_vector = ft_vector - base_vector
        output.store[simd_width](i, result_vector)

    vectorizable_elements = (num_elements // simd_width) * simd_width
    vectorize[vec_subtract, simd_width](vectorizable_elements)

    # Handle remaining elements
    for i in range(vectorizable_elements, num_elements):
        output[i] = ft[i] - base[i]

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000
    print("Operation on the weights takes : ", elapsed_time)
    return output

fn apply_task_vector(
    pretrained: Tensor[DType.float32], 
    task_vector: Tensor[DType.float32], 
    scaling_coef: Float32
) raises -> Tensor[DType.float32]:
    
    if not pretrained.shape() == task_vector.shape():
        raise "Shape mismatch between pretrained and task vector tensors."

    if scaling_coef < 0:
        raise "Scaling coefficient must be non-negative."

    start_time = perf_counter_ns()

    updated_weights = Tensor[DType.float32](pretrained.shape())
    num_elements = pretrained.num_elements()
    
    alias simd_width = simdbitwidth() // 32  # 32 bits per float32

    @parameter
    fn vec_apply_task[simd_width: Int](i: Int):
        if i + simd_width <= num_elements:
            pretrained_vector = pretrained.load[simd_width](i)
            task_vector_scaled = task_vector.load[simd_width](i) * SIMD[DType.float32, simd_width](scaling_coef)
            updated_vector = pretrained_vector + task_vector_scaled
            updated_weights.store[simd_width](i, updated_vector)
        else:
            for j in range(i, num_elements):
                updated_weights[j] = pretrained[j] + scaling_coef * task_vector[j]

    vectorize[vec_apply_task, simd_width](num_elements)

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000

    print("Applying task vector took: ", elapsed_time, " ms")
    return updated_weights


def main():
    start_time = perf_counter_ns()
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    ft_model = "KingNish/Reasoning-Llama-1b-v0.1"
    output_dir = "models"

    # Load model weights (now optimized to use binary if available)
    base_weights = load_and_save_model(base_model, output_dir)
    ft_weights = load_and_save_model(ft_model, output_dir)

    # Compute the difference between fine-tuned and base weights
    diff_weights_tensor = diff_weights(base_weights, ft_weights)

    # Optionally, save the difference weights
    diff_weights_tensor.tofile(os.path.join(output_dir, "diff_weights.bin"))
    print("Difference weights saved.")

    # Apply the task vector to the base model weights
    scaling_coef = Float32(0.5)
    updated_weights = apply_task_vector(base_weights, diff_weights_tensor, scaling_coef)

    # Save the updated weights
    updated_weights.tofile(os.path.join(output_dir, "updated_base_weights.bin"))
    print("Updated base model weights saved.")

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000

    print("Total execution time: ", elapsed_time, " ms")