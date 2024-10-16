import sys
from operatio.run import operatio_run

fn main() raises:
    var args = sys.argv()
    if len(args) < 2:
        print("Please specify the operation to perform. Choices include:")
        print("- operatio")
        # Add more operations here in the future
        return

    var operation = String(args[1])
    if operation == "operatio":
        try:
            operatio_run()
        except e:
            print("Operation failed: ", e)
    else:
        raise Error("Unrecognized operation: " + operation)