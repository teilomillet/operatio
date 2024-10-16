# main.mojo

import sys
from operatio import download_model, extract_task_vector, apply_task_vector, full_pipeline

fn main() raises:
    var args = sys.argv()
    if len(args) < 2:
        print("Please specify the operation to perform. Choices include:")
        print("- download: Download and save model weights")
        print("- extract: Extract task vector from two models")
        print("- apply: Apply task vector to a model")
        print("- full: Perform full pipeline (download, extract, apply)")
        return

    var operation = String(args[1])
    try:
        if operation == "download":
            download_model()
        elif operation == "extract":
            extract_task_vector()
        elif operation == "apply":
            apply_task_vector()
        elif operation == "full":
            full_pipeline()
        else:
            raise Error("Unrecognized operation: " + operation)
    except e:
        print("Operation failed: ", e)