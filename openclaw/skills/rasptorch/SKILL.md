---
name: rasptorch
description: Use the rasptorch CLI to create tensors, inspect Vulkan (GPU) availability, build models, and run training.
metadata:
  clawdbot:
    os: [windows, linux]
    requires:
      anyBins: [rasptorch, python, py]
---

# rasptorch skill

Use this skill when you need to operate **rasptorch** from an OpenClaw agent by calling its **agent-native CLI** and consuming JSON output.

## Command rule

- Prefer JSON output: always pass `--json` when returning results to the user/agent.
- Prefer explicit device selection:
  - `--device auto` (default): use GPU when Vulkan is working, else CPU
  - `--device gpu`: force GPU (will error / fall back depending on Vulkan availability)
  - `--device cpu`: force CPU

## How to run the CLI

Prefer the installed console script:

```bash
rasptorch --json info
```

If the console script is unavailable, use module invocation from this repo:

```bash
python -m rasptorch.CLI.cli --json info
```

## Quick checks (read-only)

### Environment + Vulkan status

```bash
rasptorch --json info
```

Key fields:
- `vulkan_available`
- `vulkan_using_real_gpu`
- `vulkan_status`

## Tensor workflows

### Create tensors

```bash
rasptorch --json tensor random --shape 2,3,4 --device cpu --dtype float32
rasptorch --json tensor zeros  --shape 3,4   --device auto
rasptorch --json tensor ones   --shape 5,10  --device gpu
```

Notes:
- Keep shapes small unless the user explicitly requests large workloads.
- Prefer `--device auto` unless the user asks to force CPU/GPU.

## Model workflows (stateful)

Model commands keep a session-like registry of created models.

### Create models

```bash
rasptorch --json model linear --input-size 10 --hidden-sizes 32,16 --output-size 2
rasptorch --json model mlp --layers 64,32,16,2
rasptorch --json model cnn --in-channels 3 --out-channels 16,32
rasptorch --json model gru --input-size 8 --hidden-size 32 --num-layers 1
rasptorch --json model transformer --vocab-size 32000 --d-model 256 --num-heads 8 --num-layers 4
```

### List / remove

```bash
rasptorch --json model list
rasptorch --json model remove --model-id <model-id>
```

### Save / load (writes files)

Only do this when the user requests it (or approves writing files):

```bash
rasptorch --json model save --model-id <model-id> --path ./model.pkl
rasptorch --json model load --path ./model.pkl
```

### Train (writes compute + may take time)

Only run training after user approval.

```bash
rasptorch --json model train --model-id <model-id> --epochs 5 --lr 0.001 --batch-size 32 --device auto --optimizer Adam
```

## UI / chat modes

- `rasptorch ui` starts a Streamlit server (long-running). Only launch with user approval.
- `rasptorch chat` starts an interactive REPL. Avoid in automated tool flows.
