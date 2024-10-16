# operatio/cli.mojo

from sys.arg import argv

struct CLIArgs:
    var operation: String
    var base_model: String
    var ft_model: String
    var output_dir: String
    var scaling_coef: Float32

    fn __init__(inout self, operation: String, base_model: String, ft_model: String, output_dir: String, scaling_coef: Float32):
        self.operation = operation
        self.base_model = base_model
        self.ft_model = ft_model
        self.output_dir = output_dir
        self.scaling_coef = scaling_coef

fn parse_args() raises -> CLIArgs:
    var args = argv()
    
    if len(args) < 3:
        print("Usage: mojo run main.mojo <operation> <base_model> [<ft_model>] [--output_dir <dir>] [--scaling_coef <coef>]")
        print("Operations: load, extract, transform, full")
        raise Error("Insufficient arguments")

    var operation = String(args[1])
    var base_model = String(args[2])
    var ft_model = String("")
    var output_dir = String("models")
    var scaling_coef = Float32(0.5)

    var i = 3
    while i < len(args):
        if args[i] == "--output_dir" and i + 1 < len(args):
            output_dir = String(args[i + 1])
            i += 2
        elif args[i] == "--scaling_coef" and i + 1 < len(args):
            scaling_coef = Float32(atol(args[i + 1]))
            i += 2
        elif ft_model == "" and args[i][0] != '-':
            ft_model = String(args[i])
            i += 1
        else:
            i += 1

    return CLIArgs(operation, base_model, ft_model, output_dir, scaling_coef)