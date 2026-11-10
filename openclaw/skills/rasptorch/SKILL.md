---
name: rasptorch
description: Operate rasptorch (NumPy deep learning library) via its agent-native CLI that outputs JSON. Use when the user asks to create tensors, build or train models (mlp, cnn, cv, jev, gru, transformer, linear, lora), manage GPU backends, benchmark matmul throughput, or save/load/convert rasptorch models.
---

# Operate rasptorch

Operate the **rasptorch** deep learning library by calling its agent-native CLI and consuming structured JSON output.

For detailed CLI examples, see `rasptorch/CLI/rasptorch CLI.md` in the repo.

## Get started

Prefer the installed console script:

```bash
rasptorch --json <command> [args...]
```

If the console script is unavailable, use module invocation from the repo root:

```bash
python -m rasptorch.CLI.cli --json <command> [args...]
```

Rules:

- Pass `--json` on every call; parse stdout only after confirming the exit code is 0. Errors arrive as JSON with an `error` field and a non-zero exit code.
- Global options go after the program name and before the command: `--device cpu|gpu|auto`, `--backend auto|numpy|vulkan|opencl|cuda`. Prefer `--backend auto` unless the user asks to pin a backend.
- Model commands are stateful — a registry of created models persists in the session.

## Check environment first

```bash
rasptorch --json info          # version, device, vulkan_available, vulkan_status
rasptorch --json backend list  # active backend + availability of each
```

## Create tensors

```bash
rasptorch --json --backend numpy tensor random --shape 2,3,4 --device cpu --dtype float32
rasptorch --json tensor zeros --shape 3,4
rasptorch --json --backend vulkan tensor ones --shape 5,10 --device gpu
```

Keep shapes small unless the user explicitly requests large workloads.

## Manage backends

```bash
rasptorch --json backend connect vulkan          # connect; add --strict to fail instead of falling back to CPU
rasptorch --json backend benchmark --backends numpy,vulkan --size 256 --iterations 20
```

Benchmark results include `elapsed_seconds`, `iterations_per_second`, and `estimated_gflops` per backend; unavailable backends report `status: "unavailable"` with an `error`.

## Build models

```bash
rasptorch --json model linear --input-size 10 --hidden-sizes 32,16 --output-size 2
rasptorch --json model mlp --layers 64,32,16,2
rasptorch --json model cnn --in-channels 3 --out-channels 16,32 --kernels 3,3
rasptorch --json model cv --in-channels 3 --out-channels 32,64 --num-classes 10
rasptorch --json model jev --input-size 10 --hidden-sizes 32,16 --num-decisions 2
rasptorch --json model gru --input-size 8 --hidden-size 32 --num-layers 1
rasptorch --json model transformer --vocab-size 32000 --d-model 256 --num-heads 8 --num-layers 4
```

Options:

- `--activation relu|gelu|tanh|sigmoid|silu|leaky_relu|elu|none` (default `relu`) is supported by `linear`, `mlp`, `cnn`, `cv`, and `jev`; `--activations <a1,a2,...>` overrides it per layer-transition (count must match).
- `--pool max|avg|none` pools after each conv block in `cv`.

## Train, combine, save, and load

Only run training and any commands that write files after user approval.

```bash
rasptorch --json model train --model-id <model-id> --epochs 5 --lr 0.001 --batch-size 32 --device auto --optimizer Adam
rasptorch --json model combine <model-a-id> <model-b-id>     # chain two registered models
rasptorch --json model lora --base-model <model-id> --rank 8 --alpha 16.0
rasptorch --json model save --model-id <model-id> --path ./model.pkl
rasptorch --json model load --path ./model.pkl
rasptorch --json model convert-legacy --src ./old_ckpt.pth --dst ./model.pth
rasptorch --json model list
rasptorch --json model remove --model-id <model-id>
```

While training starts, a `{"status": "training_started"}` line is emitted on stdout before the final JSON result.

## Launch UI or chat

- `rasptorch ui` starts a Streamlit server (long-running; supports `--port`) — only launch with user approval.
- `rasptorch chat` starts an interactive REPL — avoid it in automated tool flows.

## Discover capabilities

Run `rasptorch --json info` and `rasptorch --json backend list` to discover available backends and GPU status at runtime. Every command exposes agent-friendly `--help` (e.g. `rasptorch model --help`).
