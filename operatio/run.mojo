from .cli import parse_args
from .load import load_and_save_model
from .extract import extract_weight_diff
from .transform import transform_weights
import os

fn operatio_run() raises:
    try:
        var args = parse_args()

        print("Loading models...")
        var base_weights = load_and_save_model(args.base_model, args.output_dir)
        var ft_weights = load_and_save_model(args.ft_model, args.output_dir)

        print("Extracting weight difference...")
        var diff_weights_tensor = extract_weight_diff(base_weights, ft_weights)
        diff_weights_tensor.tofile(os.path.join(args.output_dir, "diff_weights.bin"))
        print("Weight difference extracted and saved.")

        print("Transforming weights...")
        var updated_weights = transform_weights(base_weights, diff_weights_tensor, args.scaling_coef)
        updated_weights.tofile(os.path.join(args.output_dir, "updated_base_weights.bin"))
        print("Updated base model weights saved.")
    except e:
        print("An error occurred: ", e)