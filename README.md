# Operatio

![Operatio](img/operatio-logo.svg)

Operatio is a tool for manipulating language models using task vectors. It's designed to help me learned about the underlying of the task arithmetic and about the mojo lang, it's also a tool to explore large language models through weight manipulation. I might try to implement the papers in the pdf/.

## What are Task Vectors?

Task vectors are the difference between the weights of a base model and a fine-tuned model. They capture the "knowledge" or "skills" that a model has acquired for a specific task during fine-tuning. By manipulating these task vectors, we can potentially transfer (or suppress) skills between models or create models with combined capabilities.

## What Operatio Does

Operatio provides a suite of operations for working with language models and task vectors:

1. **Load Models**: Efficiently load and save model weights.
2. **Extract Difference**: Compute the task vector by finding the difference between a base model and a fine-tuned model.
3. **Transform Model**: Apply a task vector to a base model, potentially transferring skills or knowledge.
4. **Full Pipeline**: Perform all of the above operations in sequence.

## Installation

I used `magic` from modular so I recommand using it too. After having clone this github, simply use `magic install` then `magic shell` and you can use the following commands.

## Usage

Operatio has a command-line tool with several operations:

```bash
mojo run main.mojo <operation> <base_model> [<ft_model>] [--output_dir <dir>] [--scaling_coef <coef>]
```

## Operations:

- load: Load and save model weights
- extract: Compute the task vector
- transform: Apply a task vector to a model
- full: Run the complete pipeline

## Arguments:

- `<operation>`: The operation to perform (load, extract, transform, or full)
- `<base_model>`: Path or identifier for the base model
- `<ft_model>`: Path or identifier for the fine-tuned model (required for extract and full operations)
- `--output_dir`: Directory to save output files (default: "models")
- `--scaling_coef`: Scaling coefficient for the task vector (default: 0.5)

## Examples:

Load models:
```bash
mojo run main.mojo load path/to/base/model path/to/finetuned/model
```

Extract task vector:
```bash
mojo run main.mojo extract path/to/base/model path/to/finetuned/model
``` 

Transform a model:
```bash
mojo run main.mojo transform path/to/base/model --scaling_coef 0.7
```

Run full pipeline:
```bash
mojo run main.mojo full path/to/base/model path/to/finetuned/model --scaling_coef 0.7
```

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.