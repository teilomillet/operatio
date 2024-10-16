# operatio/run.mojo

from .cli import CLIArgs
from .load import load_and_save_model
from .extract import extract_weight_diff
from .transform import transform_weights
from max.tensor import Tensor
import os

fn load_models(args: CLIArgs) raises:
    print("Loading base model...")
    _ = load_and_save_model(args.base_model, args.output_dir)
    if args.ft_model != "":
        print("Loading fine-tuned model...")
        _ = load_and_save_model(args.ft_model, args.output_dir)
    print("Models loaded and saved.")

fn extract_difference(args: CLIArgs) raises:
    if args.ft_model == "":
        raise Error("Fine-tuned model is required for extraction.")
    print("Loading models...")
    var base_weights = load_and_save_model(args.base_model, args.output_dir)
    var ft_weights = load_and_save_model(args.ft_model, args.output_dir)
    print("Extracting weight difference...")
    var diff_weights_tensor = extract_weight_diff(base_weights, ft_weights)
    diff_weights_tensor.tofile(os.path.join(args.output_dir, "diff_weights.bin"))
    print("Weight difference extracted and saved.")

fn transform_model(args: CLIArgs) raises:
    print("Loading base model...")
    var base_weights = load_and_save_model(args.base_model, args.output_dir)
    print("Loading weight difference...")
    var diff_weights = Tensor[DType.float32].fromfile(os.path.join(args.output_dir, "diff_weights.bin"))
    print("Transforming weights...")
    var updated_weights = transform_weights(base_weights, diff_weights, args.scaling_coef)
    updated_weights.tofile(os.path.join(args.output_dir, "updated_base_weights.bin"))
    print("Updated base model weights saved.")

fn full_pipeline(args: CLIArgs) raises:
    load_models(args)
    extract_difference(args)
    transform_model(args)