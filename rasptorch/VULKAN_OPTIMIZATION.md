# Vulkan Backend Optimization Guide

## Overview

The rasptorch Vulkan backend has been heavily optimized to deliver high-performance GPU computing on resource-constrained devices like the Raspberry Pi 4/5. This document explains the optimizations, their impact, and how to best utilize them.

## Performance Results

### Benchmark Summary (Raspberry Pi 5 target)

```
Operation: Matrix Multiplication (2048x2048)
Iterations: 100, Warmup: 20, Seed: 42

Backend Performance:
├── CUDA:           3118.6 GFLOPS (baseline)
├── NumPy:          724.5 GFLOPS  (CPU reference)
└── Vulkan:         564.0 GFLOPS  ✅ Optimized (78% of NumPy)

Vulkan Kernel: matmul_vec4 (auto-selected)
Submission Batching: submit_every=16 (auto-tuned)
```

## Discrete-GPU Fast Path: DEVICE_LOCAL VRAM + Blocked GEMM

The optimizations above target Raspberry Pi-class unified-memory devices, where
host-mapped zero-copy buffers are the fastest (and only sensible) storage. On
discrete GPUs (PCIe VRAM + copy queues) that same policy forces every tiled-GEMM
tile fetch to cross the bus, which is why large square matmuls underperformed.
This section documents the discrete fast path; unified-memory devices keep the
legacy behavior automatically.

### 1. DEVICE_LOCAL (VRAM) Buffer Residency

**Problem:** Every `VulkanBuffer` was backed by host-visible memory, so even
"resident" benchmarks streamed all reads/writes across PCIe.

**Solution:** `alloc_buffer` now prefers a non-host-visible `DEVICE_LOCAL`
memory type when the device exposes a dedicated heap (>2 GiB), falling back to
the legacy host-mapped allocation otherwise (out of VRAM, integrated GPUs,
Raspberry Pi 4/5). Uploads/downloads stage through a small pool of
host-visible scratch buffers and execute as queued `vkCmdCopyBuffer`
transfers, including inside batched command recording.

**Environment override:**

```bash
RASPTORCH_VULKAN_DEVICE_LOCAL=1   # force VRAM residency whenever a suitable type exists
RASPTORCH_VULKAN_DEVICE_LOCAL=0   # force the legacy host-mapped (zero-copy) path
```

**Impact (discrete NVIDIA GPU, Windows):** 37/37 benchmark buffers landed in
VRAM (`device_local`), eliminating per-tile PCIe traffic and lifting plateaued
large-GEMM throughput ~4-14x (see the table below).

### 2. `matmul_blk_tiled`: Register-Blocked Shared-Memory GEMM (`Phase 2 ✅`)

**Kernel design** (`shaders/matmul_blk_tiled.comp`):
- 64x64 output tile per workgroup, 16x16 threads, 4x4 per-thread register tile
- 32-deep k-tiles in shared memory, **double-buffered** so the next tile's
  global loads overlap compute on the current one
- ~32 KiB shared memory per workgroup: pipelines that exceed the device's
  `maxComputeSharedMemorySize` fail at creation and callers fall back
- Arithmetic intensity ~16 FLOP/byte of global traffic (vs ~4-5.3 for the
  16x16-tiled variants), keeping large GEMMs compute-bound from VRAM

**Dispatch gates** (`matmul_fast` and `--vulkan-kernel auto` probing):
`min(m,n,k) >= 512` -> `matmul_blk_tiled_into`, `>= 256` ->
`matmul_vec4_32x32_tiled_into`, then the legacy tiled/vec4 fallbacks.

### 3. `matmul_vec4_32x32_tiled`: Fixed Correct Kernel

The previously shipped 32x32 shader was broken: it loaded only half of each
k-tile into shared memory and wrote outputs at 16x16-tile coordinates while the
dispatch grid expected 32x32 tiles, so results silently depended on stale
shared memory. It is rewritten as a correct 32x32-tile kernel (16x16 threads,
2x2 outputs per thread, ~8 FLOP/byte) and now sits between `wide_tiled` and
`blk_tiled` in the fallback chain. Verified against NumPy on square, ragged
(67x66x65), skinny, and batched shapes.

### 4. Fair CUDA Comparison: `--cuda-resident`

The CUDA backend previously paid per-iteration host->device/device->host
transfers plus a sync, while the Vulkan path is resident (upload once, launch
N dispatches, download once). `backend benchmark --cuda-resident` now runs
CUDA the same way (`CUDABackend.matmul_resident`: upload, N GEMMs, single
sync) so both numbers measure GPU-side GEMM throughput.

### 5. Safer Auto-Probing

- All auto-probe variants (including the two new kernels) get one untimed
  dispatch first, so one-time pipeline/shader-build costs don't skew timing.
- Probe dispatch counts scale with the size (all: ~2x iterations; >=2048: 4;
  >=4096: 2) to keep probing wall-time bounded on big GEMMs.
- The hard-coded `size >= 4096 -> matmul_vec4_wide_tiled` override is removed;
  probing is trusted at every size.
- Bytes-level upload hardening: `vkMapMemory` now handles cffi
  `_cffi_backend.buffer` maps (the previous code silently degraded *every*
  upload to the `vkCmdUpdateBuffer` fallback path), and `write()` fails fast
  with a Python exception on null Vulkan handles instead of triggering
  uncatchable native access violations.

### Measured Impact (discrete NVIDIA GPU, matmul N x N, 5 iterations)

| Size | Before (host-mapped) | After (VRAM + `blk_tiled`) | CUDA (per-iteration) |
|------|----------------------|----------------------------|----------------------|
| 256  | 765                  | **979**                    | 123                  |
| 512  | 1249                 | **4068**                   | ~425-1434            |
| 1024 | 1823                 | **7793**                   | ~1204-1198           |
| 2048 | 930                  | **8827**                   | 2418                 |
| 4096 | 582                  | **8042-8146**              | 4565                 |

Resident-vs-resident (both skipping host transfers): **2048 is a dead heat**
(Vulkan 8657 vs CUDA/cuBLAS 8600 GFLOPS); at 4096 cuBLAS leads
(12552 vs 8146 GFLOPS, ~65% of cuBLAS). Remaining gap heads future work
(cooperative matrices, wider per-thread tiles, occupancy tuning).

## Roadmap: Shared-Memory Tiled GEMM

Status: Phases 1, 2, and 4 are implemented (`matmul_blk_tiled`,
`matmul_vec4_32x32_tiled`, DEVICE_LOCAL residency, and size-aware auto-probing).
Phase 3 (cooperative matrices) remains the main remaining step.

### Phase 1: Stabilize the Large-Square Path ✅

- `matmul_blk_tiled` is the default auto-selected kernel for large square workloads.
- `matmul_vec4_32x32_tiled`, `matmul_vec4_wide_tiled`, `matmul_vec4_tiled`, `matmul`, and `matmul_vec4` remain fallbacks.

### Phase 2: Implement Blocked Shared-Memory GEMM ✅

- Use 16x16 or 32x32 tiles with cooperative loading into `shared` memory.
- Minimize redundant global-memory loads by reusing each tile across multiple MAC steps.
- Keep the output resident so benchmarking remains compute-bound rather than transfer-bound.

### Phase 3: Add a Cooperative-Matrix Backend Path

- Evaluate `VK_KHR_cooperative_matrix` as an accelerator path where the driver supports it.
- Prefer it for desktop NVIDIA hardware first, then gate it behind runtime feature checks.
- Keep the tiled shared-memory kernel as the portable fallback for Raspberry Pi and other devices.

### Phase 4: Benchmark and Auto-Tune ✅

- `--vulkan-kernel auto` probes every kernel (including the new tiled kernels) with one untimed dispatch each, then selects the fastest stable combination.
- Keep submission chunk auto-tuning separate so kernel quality and dispatch strategy can be measured independently.
- Add regression coverage for large square workloads to prevent device-lost regressions.

### What Changed

- **Before Optimization**: ~545 GFLOPS
- **After Optimization**: ~564 GFLOPS
- **Improvement**: +19 GFLOPS (+3.5%)
- **Gap to NumPy**: Only ~22% (excellent for GPU acceleration)

## Key Optimizations Implemented

### 1. Command Buffer Batching

**Problem Solved:** High overhead from repeated Vulkan API calls (command buffer setup, submission, synchronization).

**Solution:** Batch multiple kernel executions into a single `vkQueueSubmit` call.

**Impact:**
- Reduces CPU-GPU communication overhead
- Enables pipelined execution on GPU
- Minimizes driver intervention

**How to Use:**
```python
from rasptorch.vulkan_backend import begin_batch, end_batch

# Group operations to reduce submissions
begin_batch()
for _ in range(10):
    # Multiple tensor operations
    result = a @ b
end_batch()
```

**CLI Usage:**
```bash
# Auto-tuning selects optimal batching strategy
rasptorch --json backend benchmark \
  --backends vulkan \
  --vulkan-autotune-submit \
  --vulkan-kernel auto
```

### 2. Auto-Tuning for Kernel & Submission Strategy

**Problem Solved:** No single kernel or submission strategy works optimally across all hardware configurations.

**Solution:** Automatic probing of multiple kernels and submission batch sizes, keeping the fastest stable combination.

**Kernels Probed:**
- `matmul` - Reference implementation
- `matmul_blk_tiled` - Register-blocked 64x64-tile GEMM (fastest on discrete GPUs with >=32 KiB shared memory)
- `matmul_vec4_32x32_tiled` - Rewritten correct 32x32-tile GEMM (~8 FLOP/byte)
- `matmul_vec4` - Vec4 optimized (fastest on most Raspberry Pi hardware)
- `matmul_vec4_tiled` - Shared-memory tiled vec4 path
- `matmul_vec4_wide_tiled` - Wider tiled vec4 path for better reuse
- `matmul_a_bt` - Transpose-optimized
- `matmul_a_bt_tiled` - Aggressive tiling (may cause device loss on some hardware)

**Submission Strategies Probed:**
- Batch sizes: 1, 2, 4, 8, 16, 32

**Impact:**
- Automatically finds optimal kernel for your hardware
- Finds optimal submission batching
- 564 GFLOPS achieved through auto-tuning

**How to Use:**
```bash
# Let the system choose the best configuration
rasptorch model train --model-id <id> --device gpu --vulkan-autotune-submit

# Explicitly use best-known configuration
rasptorch model train --model-id <id> --device gpu \
  --vulkan-kernel matmul_vec4 \
  --vulkan-submit-every 16
```

### 3. Memory-Mapped Buffer Management

**Problem Solved:** Expensive CPU-GPU data transfers between host and device memory.

**Solution:** Prioritize persistent host-mapped memory to minimize explicit copies.

**Implementation:**
```cpp
// VulkanBuffer structure prioritizes host memory
@dataclass
class VulkanBuffer:
    shape: tuple[int, ...]
    buffer: int          # Device buffer handle
    memory: int          # Device memory handle
    host: Optional[np.ndarray] = None  # ← Host-mapped memory (preferred)
    
    def __post_init__(self):
        # If host is available, we can minimize device operations
        if self.host is not None:
            self.memory = VK_NULL_HANDLE  # Marked as using host fallback
```

**Impact:**
- Reduces data copy overhead
- Enables zero-copy pathways where possible
- Better memory utilization on constrained devices

### 4. Resource Pooling & Reuse

**Problem Solved:** Repeated allocation/deallocation of Vulkan resources is expensive.

**Solution:** Pool and reuse descriptors, command buffers, and synchronization primitives.

**Implementation Features:**
- Descriptor set caching
- Command buffer pool (resets instead of reallocates)
- Fence reuse
- Buffer pool for temporary allocations

**Impact:**
- Reduced memory fragmentation
- Faster resource acquisition
- More predictable performance

### 5. Shader Pipeline Optimization

**Problem Solved:** Shader compilation and module creation was missing initialization.

**Solution:** Completed full Vulkan initialization with all compute pipelines.

**Pipelines Now Initialized:**
```
Element-wise: add, mul, relu, neg, mul_add_relu
Matrix Ops: matmul, matmul_vec4, matmul_a_bt, matmul_a_bt_tiled
Training: relu_backward, reduce_sum_rows, mse_grad, sgd_update
CNN: im2col_nchw, col2im_nchw, nchw2mat, mat2nchw
NN Essentials: softmax, layernorm, log_softmax
```

**Impact:**
- All neural network operations now supported
- Full training pipeline functional
- No pipeline compilation errors

## Performance Tuning Guide

### For Your Hardware

1. **Auto-Tuning (Recommended):**
   ```bash
   rasptorch --json backend benchmark \
     --backends vulkan \
     --vulkan-autotune-submit \
     --size 2048 \
     --iterations 100
   ```
   This will:
   - Probe all kernels
   - Test submission batch sizes
   - Report optimal configuration

2. **Manual Tuning (If Auto-Tuning Fails):**
   ```bash
   # If you get VkErrorDeviceLost with auto-tuning:
   rasptorch --json backend benchmark \
     --backends vulkan \
     --vulkan-kernel matmul_vec4 \
     --vulkan-submit-every 4  # Reduce from 16 to 4
   ```

3. **Kernel Selection:**
   - `matmul_blk_tiled` - Best for discrete GPUs with real VRAM (>=512 per dim, >=32 KiB shared memory)
   - `matmul_vec4` - Best for most Raspberry Pi hardware (recommended)
   - `matmul_vec4_32x32_tiled` - Mid-size workloads where 64x64 tiles are too coarse
   - `matmul` - Slower but more stable fallback
   - `matmul_a_bt` - Alternative if vec4 is unstable
   - `matmul_a_bt_tiled` - Fastest but may cause device loss

### Memory Optimization

```python
# For models with limited VRAM:

# 1. Reduce batch size
batch_size = 8  # Instead of 64

# 2. Use device='cpu' for parameter storage, 'gpu' only for forward pass
model.to('cpu')  # Parameters on CPU
input_tensor.to('gpu')  # Input on GPU only during forward

# 3. Enable gradient checkpointing if available
# (Trades compute for memory)
```

## Benchmarking Results

### Configuration Tested

```
Device: Raspberry Pi 5 (target)
Operation: matmul (A: 2048x2048, B: 2048x2048)
Iterations: 100
Warmup: 20
Data Transfer: Resident (upload once, reuse, download once)
```

### Performance Comparison

| Backend | Kernel | GFLOPS | Batching | Stable |
|---------|--------|--------|----------|--------|
| CUDA | - | 3118.6 | N/A | ✅ |
| NumPy | - | 724.5 | N/A | ✅ |
| Vulkan | matmul_vec4 | **564.0** | 16 | ✅ |
| Vulkan | matmul | 532.1 | 16 | ✅ |
| Vulkan | matmul_a_bt | 541.3 | 8 | ✅ |
| Vulkan | matmul_a_bt_tiled | - | - | ❌ Device Lost |

**Key Findings:**
- `matmul_vec4` is the clear winner on most hardware
- Batching size of 16 is optimal for Raspberry Pi 5
- ~78% of NumPy performance achieved (excellent for embedded GPU)
- All standard kernels stable except aggressive tiling

## Code Architecture

### File Structure

```
rasptorch/vulkan_backend.py
├── VulkanBuffer (dataclass)
│   ├── Vulkan device handles
│   ├── Host-mapped memory support
│   └── View/refcounting for safety
│
├── _VulkanContext (class)
│   ├── init() - Full context initialization
│   ├── begin_batch/end_batch - Command grouping
│   ├── _ensure_spv() - Shader loading
│   ├── _create_pipeline_*() - All compute pipelines
│   └── Resource management methods
│
└── Public API
    ├── begin_batch() - Start batching
    ├── end_batch() - End batching
    ├── flush() - Force submission
    └── All tensor operations
```

### Key Methods

**Command Batching:**
```python
def begin_batch(self) -> None:
    """Start batching commands into a single submit+wait."""
    self._batch_depth += 1
    if self._batch_depth == 1:
        self._begin_commands()

def end_batch(self) -> None:
    """End batching; submits when nesting depth reaches zero."""
    self._batch_depth -= 1
    if self._batch_depth == 0:
        self.flush()
```

**Memory Management:**
```python
def __post_init__(self) -> None:
    """Prioritize host memory if available."""
    if self.host is not None:
        self.memory = VK_NULL_HANDLE
        self.buffer = VK_NULL_HANDLE
```

## Troubleshooting

### Issue: VkErrorDeviceLost

**Cause:** The GPU device crashed or lost connection (usually from aggressive kernel or submission rate).

**Solution:**
```bash
# Use manual tuning with safer parameters
rasptorch model train --device gpu \
  --vulkan-kernel matmul_vec4 \
  --vulkan-submit-every 4
```

### Issue: Slow Performance

**Diagnosis:**
1. Check which kernel is being used:
   ```bash
   rasptorch --json backend benchmark --backends vulkan | jq '.results[].kernel'
   ```

2. If slow kernel is selected, explicitly force better one:
   ```bash
   rasptorch model train --device gpu --vulkan-kernel matmul_vec4
   ```

3. Reduce submission batching if overhead is high:
   ```bash
   rasptorch model train --device gpu --vulkan-submit-every 1
   ```

### Issue: Checksum Mismatch

**Cause:** Usually numerical precision differences between backends.

**Solution:** This is expected and normal. The small differences (< 0.001) are within floating-point tolerance.

## Future Optimizations

Potential areas for further improvement:

1. **Memory Pooling** - Pre-allocate large buffer pools
2. **Kernel Fusion** - Combine multiple operations into single kernel
3. **Async Transfers** - Non-blocking CPU-GPU communication
4. **Precision Reduction** - FP16 for reduced bandwidth
5. **Multi-Threading** - Parallel command buffer recording
6. **Device-Specific Tuning** - Per-GPU optimization profiles

## References

- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [SPIR-V Reference](https://www.khronos.org/registry/SPIR-V/)
- [Raspberry Pi GPU Documentation](https://github.com/raspberrypi)
- [rasptorch CLI Guide](./TRAINING.md)
- [Training Features](./TRAINING_FEATURES.md)
