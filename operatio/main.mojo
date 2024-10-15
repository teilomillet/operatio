from python import Python, PythonObject
import os
from testing import assert_equal
from max.tensor import Tensor

def load_and_save_model(model_name: String, output_dir: String) -> Tensor[DType.float32]:
    transformers = Python.import_module("transformers")
    torch = Python.import_module("torch")
    
    os.makedirs(output_dir, exist_ok=True)
    
    safe_model_name = model_name.replace('/', '_')
    output_path = os.path.join(output_dir, safe_model_name + ".bin")
    
    if os.path.exists(output_path):
        print("Binary file already exists at: ", output_path)
        print("Loading existing weights.")
        weights = Tensor[DType.float32].fromfile(output_path)
        return weights
    
    print("Loading model:", model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    
    state_dict = model.state_dict()
    
    print("Saving model to:", output_path)
    torch.save(state_dict, output_path)
    print("Model successfully converted and saved.")
    
    weights = Tensor[DType.float32].fromfile(output_path)
    return weights

fn diff_weights[S: Int](base: Tensor[DType.float32], ft: Tensor[DType.float32]) raises -> Tensor[DType.float32]:

    # Ensure tensors have the same shape using testing.assert_equal
    assert_equal(base.shape(), ft.shape(), "Tensors must have the same shape for subtraction.")
    
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
    
    return output


def main():
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    ft_model = "KingNish/Reasoning-Llama-1b-v0.1"
    output_dir = "models"

    # Load model weights
    base_weights = load_and_save_model(base_model, output_dir)
    ft_weights = load_and_save_model(ft_model, output_dir)

    # Define SIMD width as a compile-time constant
    # You can change this to 8, 16, etc., based on your benchmarking
    # SIMD_WITH=[8]
    # Compute the difference between fine-tuned and base weights 
    diff_weights_tensor = diff_weights[8](base_weights, ft_weights)

    # Optionally, save the difference weights
    diff_weights_tensor.tofile(os.path.join(output_dir, "diff_weights.bin"))
    print("Difference weights saved.")


