# rasptorch CLI

**Agent-native command-line interface** for the rasptorch deep learning library.

Transform tensor operations, model creation, and training workflows into structured, agent-controllable commands.

## Features

- **Tensor Operations**: Create, manipulate, and query tensors with full device support (CPU/GPU)
- **Backend-First Runtime**: Select backends with `--backend` or `backend use`; CPU is labeled as `numpy`
- **Model Management**: Build, configure, and manage neural networks
- **Optimizer Support**: Train models with Adam, SGD, and other optimizers
- **Dual Interface**:
  - **Command-line mode** for scripting and pipelines
  - **JSON output** for seamless agent integration
- **Agent-Native**: Works with Claude Code, Codex, Cursor, and other AI agents

## Quick Start

### Basic Commands

```bash
# Show help
python -m rasptorch.cli --help

# Get version and environment info
python -m rasptorch.cli info

# Select backend globally (numpy|vulkan|opencl|cuda|auto)
python -m rasptorch.cli --backend numpy info

# Create a random tensor
python -m rasptorch.cli tensor random --shape 2,3,4

# Create zeros tensor
python -m rasptorch.cli tensor zeros --shape 3,4

# Create ones tensor
python -m rasptorch.cli tensor ones --shape 5,10
```

### JSON Output for Agents

```bash
# All commands support --json flag for structured output
python -m rasptorch.cli --json tensor zeros --shape 3,4
python -m rasptorch.cli --json model create-linear --input-size 10 --hidden-sizes 32,16 --output-size 2
```

### Tensor Commands

```bash
# Create random tensor
rasptorch tensor random --shape 2,3,4 [--device cpu|gpu] [--dtype float32|float64|int32|int64]

# Create zeros tensor
rasptorch tensor zeros --shape 3,4 [--device cpu|gpu] [--dtype float32|float64|int32|int64]

# Create ones tensor
rasptorch tensor ones --shape 5,10 [--device cpu|gpu] [--dtype float32|float64|int32|int64]
```

### Model Commands

```bash
# Create a linear neural network
rasptorch model create-linear --input-size 10 --hidden-sizes "32,16" --output-size 2

# Create other model types (mlp, cnn, gru, transformer, etc.)
rasptorch model mlp --layers "64,32,16,2"
rasptorch model cnn --in-channels 3 --out-channels "32,64,128"

# List all created models
rasptorch model list

# Remove a model from session storage
rasptorch model remove --model-id <model-id>

# Save model to file
rasptorch model save --model-id <model-id> --path model.pkl

# Load model from file
rasptorch model load --path model.pkl

# Train a model
rasptorch model train --model-id <model-id> --epochs 10 --lr 0.001
```

### System Commands

```bash
# Show version and environment info
rasptorch info
```

## Usage Examples

### Command-line Mode

```bash
# Create a random tensor
$ python -m rasptorch.cli tensor random --shape 2,3,4
✓ Created random tensor: [2, 3, 4] on cpu

# Create a linear model
$ python -m rasptorch.cli model create-linear --input-size 10 --hidden-sizes "32,16" --output-size 2
✓ Created linear model 140734406322768
  Architecture: 10 -> 32 -> 16 -> 2
```

### JSON Output for Agents

```bash
$ python -m rasptorch.cli --json tensor zeros --shape 3,4
{
  "status": "success",
  "tensor_id": "140732147753584",
  "shape": [3, 4],
  "dtype": "float32",
  "device": "cpu"
}
```

### Installation

The CLI is built directly into rasptorch. After installing rasptorch with the CLI dependency:

```bash
# Install rasptorch with CLI support
pip install rasptorch

# Access CLI via module invocation
python -m rasptorch.cli --help

# Or direct command (if installed via pip)
rasptorch --help
```

## Architecture

The CLI implementation consists of:

- **[rasptorch/cli.py](cli.py)** — Main Click CLI entry point with all commands
- **[rasptorch/_cli_commands.py](_cli_commands.py)** — Command implementations
- **[rasptorch/_cli_utils.py](_cli_utils.py)** — Utility functions and formatting

## Design Philosophy

This CLI follows the **CLI-Anything methodology** by making rasptorch fully agent-accessible:

1. **Authentic Integration** — Commands execute real rasptorch functions, not mocks
2. **Agent-Native Design** — Structured JSON output + self-documenting `--help`
3. **Dual Interfaces** — Both command-line and JSON output modes
4. **Zero Dependencies** — Works with standard rasptorch installation
5. **Production-Grade** — Proper error handling and validation

## API Reference

### Tensor Commands

#### `tensor random`

Create a random tensor with values from standard normal distribution.

**Options:**
- `--shape` (required): Shape as comma-separated values, e.g., `2,3,4`
- `--device` (default: `cpu`): `cpu` or `gpu`
- `--dtype` (default: `float32`): `float32`, `float64`, `int32`, `int64`

**Output:**
```json
{
  "status": "success",
  "tensor_id": "...",
  "shape": [2, 3, 4],
  "dtype": "float32",
  "device": "cpu"
}
```

#### `tensor zeros`

Create a tensor filled with zeros.

**Options:** Same as `tensor random`

#### `tensor ones`

Create a tensor filled with ones.

**Options:** Same as `tensor random`

### Model Commands

#### `model create-linear`

Create a linear neural network with optional hidden layers.

**Options:**
- `--input-size` (required): Input feature dimension
- `--hidden-sizes` (required): Hidden layer sizes as comma-separated values
- `--output-size` (required): Output dimension

**Output:**
```json
{
  "status": "success",
  "model_id": "...",
  "architecture": {
    "input_size": 10,
    "hidden_sizes": [32, 16],
    "output_size": 2,
    "num_layers": 5
  }
}
```

#### `model create-optimizer`

Create an optimizer for a model.

**Options:**
- `--model-id` (required): Model ID from `model create-linear`
- `--optimizer` (default: `Adam`): `Adam` or `SGD`
- `--lr` (default: `0.001`): Learning rate

**Output:**
```json
{
  "status": "success",
  "optimizer_id": "...",
  "type": "Adam",
  "learning_rate": 0.001
}
```

#### `model list`

List all created models in the current session.

**Output:**
```json
{
  "status": "success",
  "models": [
    {
      "model_id": "140734406322768",
      "type": "Sequential"
    }
  ],
  "total": 1
}
```

#### `model remove`

Remove a model from session storage and delete its persisted file.

**Options:**
- `--model-id` (required): Model ID to remove

**Output (success):**
```json
{
  "status": "success",
  "message": "Model <id> deleted successfully",
  "model_id": "<id>"
}
```

**Output (error):**
```json
{
  "status": "error",
  "message": "Model <id> not found"
}
```

**Example:**
```bash
# Remove a model by ID
rasptorch model remove --model-id 2ed8df05

# Remove with JSON output
rasptorch --json model remove --model-id 2ed8df05
```

### System Commands

#### `info`

Display version and environment information.

**Output:**
```json
{
  "rasptorch_version": "1.4.0",
  "numpy_version": "2.2.4",
  "device": "cpu"
}
```

## Error Handling

All commands return structured error messages:

```json
{
  "status": "error",
  "message": "Invalid shape: invalid"
}
```

## Integration with AI Agents

This CLI is designed to be fully discoverable and controllable by AI agents:

```bash
# Agents can discover available commands
rasptorch --help
rasptorch tensor --help
rasptorch model --help

# Get structured output for parsing
rasptorch --json tensor random --shape 3,4

# Chain commands for complex workflows
MODEL_ID=$(rasptorch --json model create-linear --input-size 10 --hidden-sizes 32 --output-size 2 | jq -r '.model_id')
rasptorch --json model create-optimizer --model-id $MODEL_ID --optimizer Adam --lr 0.001
```

## Development

### Testing the CLI

```bash
# Run from repository root
python -m rasptorch.cli --help
python -m rasptorch.cli info
python -m rasptorch.cli tensor random --shape 2,3,4
python -m rasptorch.cli --json tensor zeros --shape 3,4
python -m rasptorch.cli model create-linear --input-size 10 --hidden-sizes "32,16" --output-size 2
```

### Adding New Commands

1. Add command implementation to [rasptorch/_cli_commands.py](_cli_commands.py)
2. Add Click decorator and handler in [rasptorch/cli.py](cli.py)
3. Test with `python -m rasptorch.cli <command> --help`

## License

MIT License — See [LICENSE](../LICENSE) for details

## Contributing

Contributions welcome! Please ensure:

- Commands follow the existing pattern (Click groups + functions)
- JSON output is structured and predictable
- Error messages are informative
- All commands support `--json` flag
