from python import Python, PythonObject
import os
from testing import assert_equal
from max.tensor import Tensor
from time import perf_counter_ns

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


fn diff_weights[
    S: Int
](base: Tensor[DType.float32], ft: Tensor[DType.float32]) raises -> Tensor[
    DType.float32
]:
    start_time = perf_counter_ns()

    # Ensure tensors have the same shape using testing.assert_equal
    assert_equal(
        base.shape(),
        ft.shape(),
        "Tensors must have the same shape for subtraction.",
    )

    # Create an output tensor with the same shape
    output = Tensor[DType.float32](base.shape())

    var i = 0
    var num_elements = base.num_elements()

    while i + S <= num_elements:
        # Load SIMD vectors with fixed indices using fixed width
        base_vector = base.load[S](i)
        ft_vector = ft.load[S](i)

        # Perform SIMD subtraction
        result_vector = ft_vector - base_vector

        # Store the result back into the output tensor using fixed width
        output.store[S](i, result_vector)

        # Move to the next chunk
        i += S

    # Handle any remaining elements that don't fit into a SIMD chunk
    while i < num_elements:
        output[i] = ft[i] - base[i]
        i += 1

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000

    print("Operation on the weights takes : ", elapsed_time)
    return output

fn apply_task_vector[
    S: Int
](pretrained: Tensor[DType.float32], task_vector: Tensor[DType.float32], scaling_coef: Float32) raises -> Tensor[DType.float32]:
    
    # Validate tensor shapes
    if not pretrained.shape() == task_vector.shape():
        raise "Shape mismatch between pretrained and task vector tensors."

    # Validate scaling coefficient
    if scaling_coef < 0:
        raise "Scaling coefficient must be non-negative."

    start_time = perf_counter_ns()

    # Create an output tensor with the same shape
    updated_weights = Tensor[DType.float32](pretrained.shape())

    var i = 0
    var num_elements = pretrained.num_elements()

    while i + S <= num_elements:
        # Load SIMD vectors
        pretrained_vector = pretrained.load[S](i)
        
        # Create a SIMD vector with the scaling coefficient
        scaling_vector = SIMD[DType.float32, S](scaling_coef)
        
        # Scale the task vector
        task_vector_scaled = task_vector.load[S](i) * scaling_vector

        # Perform SIMD addition
        updated_vector = pretrained_vector + task_vector_scaled

        # Store the result back into the output tensor
        updated_weights.store[S](i, updated_vector)

        # Move to the next chunk
        i += S

    # Handle any remaining elements
    while i < num_elements:
        updated_weights[i] = pretrained[i] + scaling_coef * task_vector[i]
        i += 1

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

    # Define SIMD width as a compile-time constant
    #SIMD_WIDTH = 4

    # Compute the difference between fine-tuned and base weights
    diff_weights_tensor = diff_weights[4](base_weights, ft_weights)

    # Optionally, save the difference weights
    diff_weights_tensor.tofile(os.path.join(output_dir, "diff_weights.bin"))
    print("Difference weights saved.")

    # Apply the task vector to the base model weights
    scaling_coef = Float32(0.5)
    updated_weights = apply_task_vector[4](base_weights, diff_weights_tensor, scaling_coef)

    # Save the updated weights
    updated_weights.tofile(os.path.join(output_dir, "updated_base_weights.bin"))
    print("Updated base model weights saved.")

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000

    print("Total execution time: ", elapsed_time, " ms")