# operatio/main.mojo

from operatio.cli import parse_args
from operatio.run import load_models, extract_difference, transform_model, full_pipeline

fn main() raises:
    try:
        var args = parse_args()
        
        if args.operation == "load":
            load_models(args)
        elif args.operation == "extract":
            extract_difference(args)
        elif args.operation == "transform":
            transform_model(args)
        elif args.operation == "full":
            full_pipeline(args)
        else:
            raise Error("Unknown operation: " + args.operation)
    except e:
        print("An error occurred: ", e)