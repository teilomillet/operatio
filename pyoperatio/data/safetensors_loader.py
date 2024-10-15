# operatio/data/safetensors_loader.py

from pathlib import Path
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
from max.driver import Tensor, CPU, CUDA
from max.dtype import DType
from typing import Dict
from tqdm import tqdm
import torch
import numpy as np
import requests
import os

from pyoperatio.utils.logger import debug, info, error
from pyoperatio.utils.decorators import timed

def bfloat16_to_float32(bf16_array):
    """
    Convert a numpy array of bfloat16 (stored as uint16) to float32.
    """
    # View as uint16
    uint16_array = bf16_array.view(np.uint16)
    # Shift bits to align with float32 exponent and mantissa
    uint32_array = uint16_array.astype(np.uint32) << 16
    # View as float32
    float32_array = uint32_array.view(np.float32)
    return float32_array

def ensure_model_exists(model_name: str, model_dir: Path) -> Path:
    """
    Ensures the model safetensors file exists, downloading it if necessary.
    """
    model_path = model_dir / model_name
    safetensors_filename = f"{model_name.split('/')[-1]}.safetensors"
    safetensors_path = model_path / safetensors_filename
    
    if not safetensors_path.exists():
        print(f"Safetensors file not found for {model_name}. Downloading...")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Download the file using huggingface_hub
        downloaded_path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors",
            force_download=True,
            cache_dir=model_path
        )
        
        # Rename the downloaded file to match expected filename
        os.rename(downloaded_path, safetensors_path)
    
    return safetensors_path

@timed
def load_safetensors_as_max_tensors(model_path: str, device: str = "cpu") -> Dict[str, Tensor]:
    """
    Load a model from a safetensors file and convert tensors to max.driver.Tensor.
    """
    info(f"Attempting to load model from {model_path}")
    
    try:
        # Determine the device
        if device.lower() == "cpu":
            device_obj = CPU()
        elif device.lower() == "cuda":
            device_obj = CUDA()
        else:
            raise ValueError(f"Unsupported device: {device}")

        max_tensors = {}
        with safe_open(model_path, framework="np") as f:
            info(f"Successfully opened {model_path}")
            metadata = f.metadata()
            for k in f.keys():
                debug(f"Processing tensor: {k}")
                tensor_info = metadata[k]
                dtype = tensor_info.dtype
                shape = tensor_info.shape

                debug(f"Tensor {k} - dtype: {dtype}, shape: {shape}")

                if dtype == 'BF16' or dtype == 'bfloat16':
                    debug(f"Converting bfloat16 to float32 for tensor {k}")
                    raw_data = f.get_tensor(k)
                    np_array = bfloat16_to_float32(raw_data)
                else:
                    np_array = f.get_tensor(k)

                # Reshape the array
                np_array = np_array.reshape(shape)

                # Convert NumPy dtype to max.dtype.DType
                max_dtype = DType.from_numpy(np_array.dtype)

                # Create max.driver.Tensor
                tensor = Tensor.from_numpy(np_array, device=device_obj)
                max_tensors[k] = tensor

        info(f"Finished loading {len(max_tensors)} tensors")
        return max_tensors

    except Exception as e:
        error(f"Error loading safetensors: {str(e)}")
        raise



def load_model_weights(model_name_or_path: str, device: str = 'cpu') -> Dict[str, Tensor]:
    """
    Load a model using Hugging Face Transformers and extract weights as max.driver.Tensors.

    Args:
        model_name_or_path (str): The model name or local path.
        device (str): The device to allocate the tensors on ("cpu" or "cuda").

    Returns:
        Dict[str, Tensor]: A dictionary mapping parameter names to max.driver.Tensors.
    """
    debug(f"Attempting to load HF model from: {model_name_or_path}")

    # Load the model using Transformers
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            device_map=None  # Ensure the model is loaded on CPU
        )
        debug(f"Successfully loaded HF model from: {model_name_or_path}")
    except Exception as e:
        error(f"Failed to load HF model from {model_name_or_path}. Error: {str(e)}")
        raise

    model.eval()
    debug("HF Model set to evaluation mode")

    # Initialize the device
    if device.lower() == 'cpu':
        device_obj = CPU()
    elif device.lower() == 'cuda':
        device_obj = CUDA()
    else:
        raise ValueError(f"Unsupported device: {device}")

    # Extract the model's state_dict and convert parameters to max.driver.Tensors
    weights = {}
    for name, param in model.state_dict().items():
        debug(f"Processing parameter: {name}")

        # Convert PyTorch tensor to NumPy array
        np_array = param.cpu().numpy()

        # Create max.driver.Tensor
        tensor = Tensor.from_numpy(np_array, device=device_obj)
        weights[name] = tensor

    return weights
