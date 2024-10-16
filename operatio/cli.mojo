from sys.arg import argv

struct CLIArgs:
    var base_model: String
    var ft_model: String
    var output_dir: String
    var scaling_coef: Float32

    fn __init__(inout self, base_model: String, ft_model: String, output_dir: String, scaling_coef: Float32):
        self.base_model = base_model
        self.ft_model = ft_model
        self.output_dir = output_dir
        self.scaling_coef = scaling_coef

fn parse_args() raises -> CLIArgs:
    var args = argv()
    
    if len(args) < 4:
        print("Usage: mojo run main.mojo operatio <base_model> <ft_model> [--output_dir <dir>] [--scaling_coef <coef>]")
        raise Error("Insufficient arguments")

    var base_model = String(args[2])
    var ft_model = String(args[3])
    var output_dir = String("models")
    var scaling_coef = Float32(0.5)

    var i = 4
    while i < len(args):
        if args[i] == "--output_dir" and i + 1 < len(args):
            output_dir = String(args[i + 1])
            i += 2
        elif args[i] == "--scaling_coef" and i + 1 < len(args):
            scaling_coef = Float32(atol(args[i + 1]))
            i += 2
        else:
            i += 1

    return CLIArgs(base_model, ft_model, output_dir, scaling_coef)