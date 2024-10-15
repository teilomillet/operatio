# operatio/main.py

from pyoperatio.data.safetensors_loader import load_model_weights
from pyoperatio.utils.logger import info, error, debug, set_log_level
from max.driver import Tensor
from pathlib import Path

def main():
    # Set log level to DEBUG for more detailed output
    set_log_level("DEBUG")

    info("Starting main function")

    # Define model names
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    fine_tuned_model_name = "KingNish/Reasoning-Llama-1b-v0.1"

    # Load the models and extract weights
    try:
        base_tensors = load_model_weights(base_model_name, device="cpu")
        info(f"Successfully loaded base model weights: {len(base_tensors)} tensors loaded")
    except Exception as e:
        error(f"Failed to load base modeggl weights: {str(e)}")
        return

    try:
        fine_tuned_tensors = load_model_weights(fine_tuned_model_name, device="cpu")
        info(f"Successfully loaded fine-tuned model weights: {len(fine_tuned_tensors)} tensors loaded")
    except Exception as e:
        error(f"Failed to load fine-tuned model weights: {str(e)}")
        return

    # Perform task vector arithmetic
    task_vectors = {}
    for key in base_tensors.keys():
        if key not in fine_tuned_tensors:
            error(f"Parameter {key} not found in fine-tuned model.")
            continue

        base_tensor = base_tensors[key]
        fine_tuned_tensor = fine_tuned_tensors[key]

        # Ensure tensors have the same shape
        if base_tensor.shape != fine_tuned_tensor.shape:
            error(f"Shape mismatch for parameter '{key}': {base_tensor.shape} vs {fine_tuned_tensor.shape}")
            continue

        # Perform subtraction to get the task vector
        task_vector = fine_tuned_tensor - base_tensor # This is not working
        task_vectors[key] = task_vector

    # Now 'task_vectors' contains the differences in weights between the fine-tuned and base models

    # (Optional) Save the task vectors back to a safetensors file
    save_task_vectors(task_vectors, "models/task_vectors.safetensors")

def save_task_vectors(task_vectors: dict[str, Tensor], output_path: str):
    """
    Save the task vectors to a safetensors file.

    Args:
        task_vectors (Dict[str, Tensor]): The task vectors to save.
        output_path (str): The output file path.
    """
    from safetensors.numpy import save_file

    # Convert max.driver.Tensor objects back to NumPy arrays
    numpy_tensors = {}
    for name, tensor in task_vectors.items():
        # Convert max.driver.Tensor to NumPy array
        numpy_array = tensor.numpy()
        numpy_tensors[name] = numpy_array

    # Save the NumPy arrays to a safetensors file
    save_file(numpy_tensors, output_path)
    info(f"Task vectors saved to {output_path}")

if __name__ == "__main__":
    main()
