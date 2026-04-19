# rasptorch CLI - Training & Model Management

Complete guide to training models and managing saved models through the rasptorch CLI.

> Backend naming note: in CLI/UI, CPU backend is labeled as **`numpy`**.
> Use `rasptorch --backend numpy ...` (or `backend use numpy` in chat mode).

## Model Management Commands

### Create Models

Create different types of neural network models:

```bash
# MLP (Multi-Layer Perceptron)
rasptorch model mlp --layers "64,32,16,2"

# Linear model with hidden layers
rasptorch model linear --input-size 10 --hidden-sizes "32,16" --output-size 2

# CNN (Convolutional Neural Network)
rasptorch model cnn --in-channels 3 --out-channels "32,64,128"

# GRU (Gated Recurrent Unit)
rasptorch model gru --input-size 128 --hidden-size 256 --num-layers 2

# Transformer
rasptorch model transformer --vocab-size 10000 --d-model 512 --num-heads 8 --num-layers 6
```

Each command returns a unique model ID (e.g., `69cf642f`) that persists across CLI sessions.

### List Models

View all models created in the current session:

```bash
rasptorch model list

# JSON output
rasptorch --json model list
```

### Train Models

Train a model with GPU (Vulkan) or CPU support:

```bash
# Train on CPU
rasptorch --backend numpy model train --model-id 69cf642f --epochs 10 --lr 0.001 --batch-size 32 --device cpu

# Train on GPU (Vulkan)
rasptorch --backend vulkan model train --model-id 69cf642f --epochs 10 --lr 0.001 --batch-size 32 --device gpu

# Specify optimizer (Adam or SGD)
rasptorch --backend numpy model train --model-id 69cf642f --epochs 5 --lr 0.01 --optimizer SGD --device cpu
```

**Training Options:**
- `--model-id`: ID of the model to train
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size for training (default: 32)
- `--device`: Training device - `cpu` or `gpu` (default: cpu)
- `--optimizer`: Optimizer type - `Adam` or `SGD` (default: Adam)

**Output:**
```
✓ Training complete
  Epochs: 10
  Device: gpu
  Final loss: 1.234567
```

### Save Models

Save a trained model to disk:

```bash
rasptorch model save --model-id 69cf642f --path /path/to/model.pkl
```

The saved model includes:
- Model architecture configuration
- Trained weights (state_dict)
- Model type information

### Load Models

Load a previously saved model:

```bash
rasptorch model load --path /path/to/model.pkl
```

After loading, you can train it further or use it for inference.

## GPU Training with Vulkan

The CLI supports GPU acceleration via Vulkan backend, ideal for Raspberry Pi 5:

```bash
# Create a model
rasptorch model mlp --layers "128,64,32,10"

# Train on Vulkan GPU (optimized for Raspberry Pi 4/5)
rasptorch model train --model-id <model_id> --epochs 20 --device gpu --batch-size 64

# Compared to CPU training (slower)
rasptorch model train --model-id <model_id> --epochs 20 --device cpu --batch-size 64
```

### Vulkan Optimizations & Performance

The Vulkan backend has been heavily optimized to minimize overhead and maximize GPU utilization:

**Key Optimizations Implemented:**
- ✅ **Command Buffer Batching** - Groups multiple kernel submissions to reduce API call overhead
- ✅ **Auto-Tuning** - Automatically selects best kernel (`matmul`, `matmul_vec4`, `matmul_a_bt`, `matmul_a_bt_tiled`) and submission strategy
- ✅ **Memory-Mapped Buffers** - Prioritizes host-mapped memory to minimize data copy overhead
- ✅ **Resource Pooling** - Reuses descriptor sets and command buffers to reduce allocation overhead

**Performance Benchmarks (Raspberry Pi 5 target):**
- **Vulkan (matmul_vec4, auto-tuned)**: 564 GFLOPS
- **NumPy**: 724 GFLOPS
- **Gap Closed**: ~22% performance gap (excellent for GPU-accelerated embedded computing)

**For Best Results:**
```bash
# Use auto-tuning for optimal kernel + submission strategy
rasptorch model train --model-id <id> --device gpu --vulkan-autotune-submit

# Or explicitly tune for your hardware
rasptorch model train --model-id <id> --device gpu --vulkan-kernel matmul_vec4 --vulkan-submit-every 16
```

### Vulkan Benefits
- **Hardware acceleration** on Raspberry Pi 5 (VideoCore VI GPU)
- **Optimized performance** - 564 GFLOPS matmul, competitive with CPU
- **Lower power consumption** than pure CPU training
- **Automatic fallback** - gracefully falls back to NumPy if Vulkan unavailable
- **Full feature support** - all neural network layers, training operations, and optimizers

## JSON Output for Integration

All commands support JSON output for scripting and integration:

```bash
# Training with JSON output
rasptorch --json model train --model-id 69cf642f --epochs 5 --device cpu

# Save with JSON output
rasptorch --json model save --model-id 69cf642f --path model.pkl

# Load with JSON output
rasptorch --json model load --path model.pkl
```

**JSON Response Format:**
```json
{
  "status": "success",
  "model_id": "69cf642f",
  "epochs": 5,
  "device": "cpu",
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "final_loss": 1.234567,
  "training_history": [3.21, 2.45, 1.89, 1.45, 1.23]
}
```

## Complete Workflow Example

```bash
# 1. Create a model
MODEL=$(rasptorch model mlp --layers "64,32,16,2" | grep -oP '[a-f0-9]{8}')
echo "Created model: $MODEL"

# 2. Train on GPU with progress
rasptorch model train --model-id $MODEL --epochs 20 --lr 0.01 --device gpu --batch-size 64

# 3. Save the trained model
rasptorch model save --model-id $MODEL --path ./my_trained_model.pkl

# 4. Load and continue training
MODEL2=$(rasptorch model load --path ./my_trained_model.pkl | grep -oP '[a-f0-9]{8}')
rasptorch model train --model-id $MODEL2 --epochs 10 --lr 0.005 --device gpu

# 5. Save final model
rasptorch model save --model-id $MODEL2 --path ./final_model.pkl
```

## Session Persistence

Models are automatically persisted in the session:
- Location: `/tmp/rasptorch_cli_session/`
- Models stay available across CLI invocations
- Session can be cleared by removing the directory

## Performance Tips

1. **Use GPU for large models** - GPU is faster for models with many parameters
2. **Increase batch size on GPU** - GPU can handle larger batches efficiently
3. **Use lower learning rates for longer training** - Scales better with more epochs
4. **Monitor training with JSON output** - Parse loss history for monitoring

## Troubleshooting

**Model not found error:**
- Verify the model ID is correct
- Check that the model was created in the same session

**GPU training slow:**
- May indicate Vulkan not available, check with `rasptorch info`
- Fall back to CPU if Vulkan is unavailable

**Shape mismatch during training:**
- Most common when layer configuration is incorrect
- Check model architecture matches expected input size

## API Integration

Use CLI output in scripts:

```bash
#!/bin/bash

# Create and train model
MODEL_ID=$(rasptorch model mlp --layers "128,64,32" | grep -oP '[a-f0-9]{8}')

echo "Training model $MODEL_ID on GPU..."
RESULT=$(rasptorch --json model train --model-id $MODEL_ID --epochs 50 --device gpu)

# Extract final loss using jq
FINAL_LOSS=$(echo "$RESULT" | jq '.final_loss')
echo "Final loss: $FINAL_LOSS"

# Save if training successful
if [ ! -z "$FINAL_LOSS" ]; then
    rasptorch model save --model-id $MODEL_ID --path ./model_$(date +%s).pkl
fi
```
