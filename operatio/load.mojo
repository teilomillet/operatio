# operatio/load.mojo

from max.tensor import Tensor
from python import Python
from time import perf_counter_ns
import os

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