# pyright: reportUndefinedVariable=false
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

"""Vulkan compute backend for rasptorch.

This backend is designed to run on Raspberry Pi 5 (VideoCore + Vulkan).

It provides a small set of GPU kernels used by rasptorch Tensor ops:
- add, mul, relu (elementwise)
- matmul (naive GEMM)

If Vulkan is unavailable (missing python bindings or driver issues), the
backend automatically falls back to NumPy so the rest of the project stays
usable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import ctypes
import subprocess

import numpy as np


_VULKAN_DISABLED_REASON: Optional[str] = None


try:
    # python package: "vulkan" (ctypes bindings)
    from vulkan import *  # type: ignore

    _HAS_VULKAN = True
except Exception as e:
    _HAS_VULKAN = False
    _VULKAN_DISABLED_REASON = f"Python package 'vulkan' not available: {e}"

# Help type checkers know the Vulkan symbols exist when installed.
if TYPE_CHECKING:  # pragma: no cover
    from vulkan import *  # type: ignore

if not _HAS_VULKAN:
    VK_NULL_HANDLE = 0  # type: ignore[assignment]


_SHADERS_DIR = Path(__file__).with_suffix("").parent / "shaders"


@dataclass
class VulkanBuffer:
    """A Vulkan buffer holding float32 data."""

    # Host-side metadata
    shape: tuple[int, ...]
    nbytes: int

    # Vulkan handles
    buffer: int
    memory: int

    # Optional CPU copy when Vulkan is unavailable (fallback mode)
    host: Optional[np.ndarray] = None

    # Simple view/refcounting so reshapes can share the same VkBuffer safely.
    # - For base buffers: base is None and refcount tracks outstanding views + self.
    # - For views: base points at the base buffer.
    base: Optional["VulkanBuffer"] = None
    refcount: int = 1


@dataclass
class _Pipeline:
    pipeline: "VkPipeline"
    pipeline_layout: "VkPipelineLayout"
    descriptor_set_layout: "VkDescriptorSetLayout"
    descriptor_pool: "VkDescriptorPool"


class _VulkanContext:
    def __init__(self) -> None:
        if not _HAS_VULKAN:
            raise RuntimeError("Python package 'vulkan' not installed")

        self.instance: Optional[VkInstance] = None
        self.physical_device: Optional[VkPhysicalDevice] = None
        self.device: Optional[VkDevice] = None
        self.queue: Optional[VkQueue] = None
        self.queue_family_index: Optional[int] = None

        self.command_pool: Optional[VkCommandPool] = None
        self.command_buffer: Optional[VkCommandBuffer] = None

        # Reuse sync + descriptor resources to reduce overhead.
        self._fence: Optional[VkFence] = None
        self._ds_cache: dict[object, VkDescriptorSet] = {}

        # Optional command buffer batching to reduce submit+wait overhead.
        self._batch_depth: int = 0
        self._cmd_recording: bool = False

        self.p_add: Optional[_Pipeline] = None
        self.p_mul: Optional[_Pipeline] = None
        self.p_relu: Optional[_Pipeline] = None
        self.p_mul_add_relu: Optional[_Pipeline] = None
        self.p_matmul: Optional[_Pipeline] = None

        self.p_neg: Optional[_Pipeline] = None

        self.p_add_scalar: Optional[_Pipeline] = None
        self.p_mul_scalar: Optional[_Pipeline] = None

        self.p_transpose2d: Optional[_Pipeline] = None

        self.p_reduce_sum_stage: Optional[_Pipeline] = None
        self.p_scale_fill: Optional[_Pipeline] = None

        # Training-oriented kernels
        self.p_add_rowvec: Optional[_Pipeline] = None
        self.p_mul_rowvec: Optional[_Pipeline] = None
        self.p_relu_backward: Optional[_Pipeline] = None
        self.p_reduce_sum_rows: Optional[_Pipeline] = None
        self.p_mse_grad: Optional[_Pipeline] = None
        self.p_sgd_update: Optional[_Pipeline] = None
        self.p_matmul_at_b: Optional[_Pipeline] = None
        self.p_matmul_a_bt: Optional[_Pipeline] = None

        # CNN helpers
        self.p_im2col_nchw: Optional[_Pipeline] = None
        self.p_col2im_nchw: Optional[_Pipeline] = None
        self.p_nchw2mat: Optional[_Pipeline] = None
        self.p_mat2nchw: Optional[_Pipeline] = None

        # Loss / optimizer helpers
        self.p_softmax_xent_loss_vec: Optional[_Pipeline] = None
        self.p_softmax_xent_backward: Optional[_Pipeline] = None
        self.p_sgd_momentum_update: Optional[_Pipeline] = None

        # NN essentials
        self.p_softmax2d: Optional[_Pipeline] = None
        self.p_softmax2d_backward: Optional[_Pipeline] = None
        self.p_log_softmax2d: Optional[_Pipeline] = None
        self.p_log_softmax2d_backward: Optional[_Pipeline] = None
        self.p_layernorm2d: Optional[_Pipeline] = None
        self.p_layernorm2d_backward: Optional[_Pipeline] = None

    # ------------------------------
    # Init
    # ------------------------------
    def init(self) -> None:
        if self.instance is not None:
            return

        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"rasptorch",
            applicationVersion=VK_MAKE_VERSION(0, 1, 0),
            pEngineName=b"rasptorch",
            engineVersion=VK_MAKE_VERSION(0, 1, 0),
            apiVersion=VK_API_VERSION_1_0,
        )
        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )

        self.instance = vkCreateInstance(create_info, None)

        devices = vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan physical devices found")

        # Pick the first device with a compute queue.
        for pd in devices:
            qprops = vkGetPhysicalDeviceQueueFamilyProperties(pd)
            for i, qp in enumerate(qprops):
                if qp.queueFlags & VK_QUEUE_COMPUTE_BIT:
                    self.physical_device = pd
                    self.queue_family_index = int(i)
                    break
            if self.physical_device is not None:
                break

        if self.physical_device is None or self.queue_family_index is None:
            raise RuntimeError("No Vulkan compute queue found")

        queue_priorities = [1.0]
        qci = VkDeviceQueueCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
            queueCount=1,
            pQueuePriorities=queue_priorities,
        )
        dci = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[qci],
        )
        self.device = vkCreateDevice(self.physical_device, dci, None)
        self.queue = vkGetDeviceQueue(self.device, self.queue_family_index, 0)

        # Command pool + single reusable command buffer
        cpci = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vkCreateCommandPool(self.device, cpci, None)
        cbai = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vkAllocateCommandBuffers(self.device, cbai)[0]

        fence_ci = VkFenceCreateInfo(sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        self._fence = vkCreateFence(self.device, fence_ci, None)

        # Pipelines
        self.p_add = self._create_pipeline_vec3("add")
        self.p_mul = self._create_pipeline_vec3("mul")
        self.p_relu = self._create_pipeline_vec2("relu")
        self.p_neg = self._create_pipeline_vec2("neg")
        self.p_mul_add_relu = self._create_pipeline_vec3("mul_add_relu")
        self.p_matmul = self._create_pipeline_matmul("matmul")

        # Scalar ops
        self.p_add_scalar = self._create_pipeline_vec2_u32f32("add_scalar")
        self.p_mul_scalar = self._create_pipeline_vec2_u32f32("mul_scalar")

        # Utility kernels
        self.p_transpose2d = self._create_pipeline_vec2_u32u32("transpose2d")

        # Reductions / broadcasts
        self.p_reduce_sum_stage = self._create_pipeline_vec2("reduce_sum_stage")
        self.p_scale_fill = self._create_pipeline_vec2_u32f32("scale_fill")

        # Extra kernels for training (bias, gradients, reductions, updates)
        self.p_add_rowvec = self._create_pipeline_vec3_u32u32("add_rowvec")
        self.p_mul_rowvec = self._create_pipeline_vec3_u32u32("mul_rowvec")
        self.p_relu_backward = self._create_pipeline_vec3("relu_backward")
        self.p_reduce_sum_rows = self._create_pipeline_vec2_u32u32("reduce_sum_rows")
        self.p_mse_grad = self._create_pipeline_vec3_u32f32("mse_grad")
        self.p_sgd_update = self._create_pipeline_vec2_u32f32("sgd_update")
        self.p_matmul_at_b = self._create_pipeline_matmul("matmul_at_b")
        self.p_matmul_a_bt = self._create_pipeline_matmul("matmul_a_bt")

        # CNN / layout helpers (larger push constants)
        self.p_im2col_nchw = self._create_pipeline_vec2_pc64("im2col_nchw")
        self.p_col2im_nchw = self._create_pipeline_vec2_pc64("col2im_nchw")
        self.p_nchw2mat = self._create_pipeline_vec2_pc64("nchw2mat")
        self.p_mat2nchw = self._create_pipeline_vec2_pc64("mat2nchw")

        # Loss / optimizer
        self.p_softmax_xent_loss_vec = self._create_pipeline_vec3_u32u32("softmax_xent_loss_vec")
        self.p_softmax_xent_backward = self._create_pipeline_vec3_u32u32("softmax_xent_backward")
        self.p_sgd_momentum_update = self._create_pipeline_vec3_pc16("sgd_momentum_update")

        # NN essentials
        self.p_softmax2d = self._create_pipeline_vec2_u32u32("softmax2d")
        self.p_softmax2d_backward = self._create_pipeline_vec3_u32u32("softmax2d_backward")
        self.p_log_softmax2d = self._create_pipeline_vec2_u32u32("log_softmax2d")
        self.p_log_softmax2d_backward = self._create_pipeline_vec3_u32u32("log_softmax2d_backward")
        self.p_layernorm2d = self._create_pipeline_vec2_u32u32("layernorm2d")
        self.p_layernorm2d_backward = self._create_pipeline_vec3_u32u32("layernorm2d_backward")

    def _create_pipeline_vec3_pc16(self, name: str) -> _Pipeline:
        """Compute pipeline with 3 storage buffers and a 16-byte push constant range."""
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=16,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=3 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    def _create_pipeline_vec2_pc64(self, name: str) -> _Pipeline:
        """Compute pipeline with 2 storage buffers and a 64-byte push constant range."""
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=64,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=2 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    def _create_pipeline_vec3_u32u32(self, name: str) -> _Pipeline:
        """Compute pipeline with 3 storage buffers and push constants (uint,uint)."""
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=8,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=3 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    def _create_pipeline_vec2_u32u32(self, name: str) -> _Pipeline:
        """Compute pipeline with 2 storage buffers and push constants (uint,uint)."""
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=8,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=2 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    def _create_pipeline_vec3_u32f32(self, name: str) -> _Pipeline:
        """Compute pipeline with 3 storage buffers and push constants (uint,float)."""
        # Same layout size as (uint,uint)
        return self._create_pipeline_vec3_u32u32(name)

    def _create_pipeline_vec2_u32f32(self, name: str) -> _Pipeline:
        """Compute pipeline with 2 storage buffers and push constants (uint,float)."""
        return self._create_pipeline_vec2_u32u32(name)

    # ------------------------------
    # Shaders / pipelines
    # ------------------------------
    def _ensure_spv(self, name: str) -> bytes:
        spv_path = _SHADERS_DIR / f"{name}.spv"
        comp_path = _SHADERS_DIR / f"{name}.comp"

        if spv_path.exists() and comp_path.exists():
            try:
                if spv_path.stat().st_mtime >= comp_path.stat().st_mtime:
                    return spv_path.read_bytes()
            except OSError:
                return spv_path.read_bytes()

        if spv_path.exists() and not comp_path.exists():
            return spv_path.read_bytes()

        if not comp_path.exists():
            raise FileNotFoundError(f"Missing shader source: {comp_path}")

        # Try to compile with glslc on the target machine.
        # On Raspberry Pi OS, install: glslang-tools or shaderc tools.
        try:
            subprocess.run(
                ["glslc", str(comp_path), "-o", str(spv_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError("glslc not found; install shader compiler tools") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed compiling {comp_path.name}:\n{e.stderr.decode('utf-8', errors='replace')}"
            ) from e

        return spv_path.read_bytes()

    def _create_shader_module(self, spv: bytes) -> VkShaderModule:
        # Vulkan expects uint32 words.
        if len(spv) % 4 != 0:
            raise ValueError("SPIR-V bytecode length must be multiple of 4")

        code_u32 = (ctypes.c_uint32 * (len(spv) // 4)).from_buffer_copy(spv)
        smci = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spv),
            pCode=code_u32,
        )
        assert self.device is not None
        return vkCreateShaderModule(self.device, smci, None)

    def _create_pipeline_vec3(self, name: str) -> _Pipeline:
        """Compute pipeline with 3 storage buffers and push constant uint n."""
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=4,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=3 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    def _create_pipeline_vec2(self, name: str) -> _Pipeline:
        """Compute pipeline with 2 storage buffers and push constant uint n."""
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=4,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=2 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    def _create_pipeline_matmul(self, name: str) -> _Pipeline:
        """Compute pipeline with 3 buffers and push constants (m,n,k)."""
        # Same as vec3 but with 12-byte push constant.
        assert self.device is not None

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        dsci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        dsl = vkCreateDescriptorSetLayout(self.device, dsci, None)

        pcr = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=12,
        )
        plci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[dsl],
            pushConstantRangeCount=1,
            pPushConstantRanges=[pcr],
        )
        pll = vkCreatePipelineLayout(self.device, plci, None)

        max_sets = 4096
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=3 * max_sets,
            )
        ]
        dpci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        dp = vkCreateDescriptorPool(self.device, dpci, None)

        spv = self._ensure_spv(name)
        sm = self._create_shader_module(spv)
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=sm,
            pName=b"main",
        )
        cpci = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pll,
        )
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [cpci], None)[0]
        vkDestroyShaderModule(self.device, sm, None)

        return _Pipeline(
            pipeline=pipeline,
            pipeline_layout=pll,
            descriptor_set_layout=dsl,
            descriptor_pool=dp,
        )

    # ------------------------------
    # Buffers
    # ------------------------------
    def _find_memory_type(self, type_bits: int, props: int) -> int:
        assert self.physical_device is not None
        mem_props = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & props) == props:
                return i
        raise RuntimeError("Failed to find suitable Vulkan memory type")

    def alloc_buffer(self, nbytes: int) -> tuple[VkBuffer, VkDeviceMemory]:
        assert self.device is not None

        bci = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=nbytes,
            usage=(
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            ),
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vkCreateBuffer(self.device, bci, None)
        req = vkGetBufferMemoryRequirements(self.device, buf)

        mem_type = self._find_memory_type(
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        mai = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=req.size,
            memoryTypeIndex=mem_type,
        )
        mem = vkAllocateMemory(self.device, mai, None)
        vkBindBufferMemory(self.device, buf, mem, 0)
        return buf, mem

    def free_buffer(self, buf: VkBuffer, mem: VkDeviceMemory) -> None:
        assert self.device is not None
        if buf != VK_NULL_HANDLE:
            vkDestroyBuffer(self.device, buf, None)
        if mem != VK_NULL_HANDLE:
            vkFreeMemory(self.device, mem, None)

    def write_buffer(self, buf: VkBuffer, mem: VkDeviceMemory, data: np.ndarray) -> None:
        assert self.device is not None
        arr = np.ascontiguousarray(data, dtype=np.float32)

        mapped = vkMapMemory(self.device, mem, 0, arr.nbytes, 0)
        if isinstance(mapped, (tuple, list)):
            mapped_ptr = mapped[1]
        else:
            mapped_ptr = mapped

        # Some bindings return a pointer wrapper; normalize to its underlying value early.
        if hasattr(mapped_ptr, "value"):
            mapped_ptr = mapped_ptr.value

        # Some environments return a bytes-like snapshot here (not writable).
        # In that case, fall back to vkCmdUpdateBuffer (supports arbitrary size, 4-byte aligned).
        if isinstance(mapped_ptr, (bytes, bytearray, memoryview, np.bytes_)):
            try:
                vkUnmapMemory(self.device, mem)
            except Exception:
                pass

            if self.command_buffer is None or self.queue is None:
                raise RuntimeError("Vulkan command buffer/queue not initialized")

            if (arr.nbytes % 4) != 0:
                raise RuntimeError(
                    "vkCmdUpdateBuffer fallback only supports 4-byte aligned sizes"
                )

            if self._cmd_recording:
                # Don't clobber an in-flight batch command buffer.
                self.flush()

            vkResetCommandBuffer(self.command_buffer, 0)
            begin = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            )
            vkBeginCommandBuffer(self.command_buffer, begin)

            raw = arr.tobytes()
            max_chunk = 65536
            for offset in range(0, len(raw), max_chunk):
                chunk = raw[offset : offset + max_chunk]
                if (len(chunk) % 4) != 0:
                    raise RuntimeError("vkCmdUpdateBuffer chunk not 4-byte aligned")
                import vulkan as _vk  # type: ignore

                p_data = _vk.ffi.new("char[]", chunk)
                vkCmdUpdateBuffer(self.command_buffer, buf, offset, len(chunk), p_data)

            vkEndCommandBuffer(self.command_buffer)
            self._submit_and_wait()
            return

        # Pointer-like: normalize to an integer address. If we can't, treat as non-writable and
        # use vkCmdUpdateBuffer.
        try:
            mapped_addr = int(mapped_ptr)
        except Exception:
            try:
                vkUnmapMemory(self.device, mem)
            except Exception:
                pass

            if self.command_buffer is None or self.queue is None:
                raise RuntimeError("Vulkan command buffer/queue not initialized")
            if (arr.nbytes % 4) != 0:
                raise RuntimeError(
                    "vkCmdUpdateBuffer fallback only supports 4-byte aligned sizes"
                )

            if self._cmd_recording:
                # Don't clobber an in-flight batch command buffer.
                self.flush()

            vkResetCommandBuffer(self.command_buffer, 0)
            begin = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            )
            vkBeginCommandBuffer(self.command_buffer, begin)

            raw = arr.tobytes()
            max_chunk = 65536
            for offset in range(0, len(raw), max_chunk):
                chunk = raw[offset : offset + max_chunk]
                if (len(chunk) % 4) != 0:
                    raise RuntimeError("vkCmdUpdateBuffer chunk not 4-byte aligned")
                import vulkan as _vk  # type: ignore

                p_data = _vk.ffi.new("char[]", chunk)
                vkCmdUpdateBuffer(self.command_buffer, buf, offset, len(chunk), p_data)

            vkEndCommandBuffer(self.command_buffer)
            self._submit_and_wait()
            return

        ctypes.memmove(mapped_addr, arr.ctypes.data, arr.nbytes)
        vkUnmapMemory(self.device, mem)

    def read_buffer(self, mem: VkDeviceMemory, nbytes: int) -> bytes:
        assert self.device is not None
        mapped = vkMapMemory(self.device, mem, 0, nbytes, 0)
        if isinstance(mapped, (tuple, list)):
            mapped_ptr = mapped[1]
        else:
            mapped_ptr = mapped

        if hasattr(mapped_ptr, "value"):
            mapped_ptr = mapped_ptr.value

        # Some environments return a bytes-like object (not a pointer).
        # Prefer treating any buffer-protocol object as the returned bytes.
        if not isinstance(mapped_ptr, (int, np.integer)):
            try:
                mv = memoryview(mapped_ptr)  # type: ignore[arg-type]
            except TypeError:
                mv = None
            if mv is not None:
                out = bytes(mv[:nbytes])
                try:
                    vkUnmapMemory(self.device, mem)
                except Exception:
                    pass
                return out

        mapped_addr = int(mapped_ptr)
        out = ctypes.string_at(mapped_addr, nbytes)
        vkUnmapMemory(self.device, mem)
        return out

    # ------------------------------
    # Dispatch
    # ------------------------------

    def _begin_commands(self) -> None:
        assert self.command_buffer is not None
        vkResetCommandBuffer(self.command_buffer, 0)
        begin = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vkBeginCommandBuffer(self.command_buffer, begin)

    def _end_commands(self) -> None:
        """End+submit unless a batch is active."""
        assert self.command_buffer is not None
        if self._batch_depth > 0:
            return
        vkEndCommandBuffer(self.command_buffer)
        self._submit_and_wait()

    def begin_batch(self) -> None:
        """Start batching commands into a single submit+wait."""
        self._batch_depth += 1
        if self._batch_depth == 1:
            self._begin_commands()
            self._cmd_recording = True

    def flush(self) -> None:
        """Submit+wait any batched commands recorded so far."""
        assert self.command_buffer is not None
        if not self._cmd_recording:
            return
        vkEndCommandBuffer(self.command_buffer)
        self._submit_and_wait()
        self._cmd_recording = False

    def end_batch(self) -> None:
        """End batching; submits when the nesting depth reaches zero."""
        if self._batch_depth <= 0:
            raise RuntimeError("end_batch called without begin_batch")
        self._batch_depth -= 1
        if self._batch_depth == 0:
            self.flush()

    def _maybe_continue_batch(self) -> None:
        """If batching is active and nothing is recording, begin a new command buffer."""
        if self._batch_depth > 0 and not self._cmd_recording:
            self._begin_commands()
            self._cmd_recording = True

    def _alloc_descriptor_set(self, p: _Pipeline) -> VkDescriptorSet:
        assert self.device is not None

        cache_key = p.descriptor_pool
        cached = self._ds_cache.get(cache_key)
        if cached is not None:
            return cached

        dsai = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=p.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[p.descriptor_set_layout],
        )
        try:
            ds = vkAllocateDescriptorSets(self.device, dsai)[0]
            self._ds_cache[cache_key] = ds
            return ds
        except KeyError as e:
            # python-vulkan may raise KeyError(error_code) if it can't map the Vulkan
            # result to an exception class.
            code = e.args[0] if e.args else None
            if code in (
                -1000069000,  # VK_ERROR_OUT_OF_POOL_MEMORY
                -1000069001,  # VK_ERROR_FRAGMENTED_POOL
            ):
                vkResetDescriptorPool(self.device, p.descriptor_pool, 0)
                self._ds_cache.pop(cache_key, None)
                ds = vkAllocateDescriptorSets(self.device, dsai)[0]
                self._ds_cache[cache_key] = ds
                return ds
            raise

    def _submit_and_wait(self) -> None:
        assert self.device is not None
        assert self.queue is not None
        assert self.command_buffer is not None

        if self._fence is None:
            fence_ci = VkFenceCreateInfo(sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
            self._fence = vkCreateFence(self.device, fence_ci, None)

        submit = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
        )
        vkResetFences(self.device, 1, [self._fence])
        vkQueueSubmit(self.queue, 1, [submit], self._fence)
        vkWaitForFences(self.device, 1, [self._fence], VK_TRUE, 10_000_000_000)  # 10s


_CTX: Optional[_VulkanContext] = None


def _disable_vulkan(reason: str) -> None:
    global _HAS_VULKAN, _VULKAN_DISABLED_REASON, _CTX
    _HAS_VULKAN = False
    _VULKAN_DISABLED_REASON = reason
    _CTX = None


def _ctx() -> _VulkanContext:
    global _CTX
    if not _HAS_VULKAN:
        raise RuntimeError(_VULKAN_DISABLED_REASON or "Vulkan backend disabled")
    if _CTX is None:
        _CTX = _VulkanContext()
        _CTX.init()
    return _CTX


def begin_batch() -> None:
    """Begin Vulkan command batching (no-op if Vulkan unavailable)."""

    if not _HAS_VULKAN:
        return
    ctx = _ctx()
    ctx.begin_batch()


def end_batch() -> None:
    """End Vulkan command batching (flushes when depth reaches zero)."""

    if not _HAS_VULKAN:
        return
    ctx = _ctx()
    ctx.end_batch()


def flush() -> None:
    """Flush any batched Vulkan commands (submit+wait)."""

    if not _HAS_VULKAN:
        return
    ctx = _ctx()
    ctx.flush()


def init(*, strict: bool = False) -> None:
    """Initialize Vulkan backend.

    When strict=False (default), initialization failures disable Vulkan and
    allow the library to continue in NumPy fallback mode.

    When strict=True, failures raise instead of silently falling back.
    """

    if not _HAS_VULKAN:
        if strict:
            raise RuntimeError(_VULKAN_DISABLED_REASON or "Vulkan backend disabled")
        return

    try:
        _ctx()
    except Exception as e:
        _disable_vulkan(str(e))
        if strict:
            raise RuntimeError(_VULKAN_DISABLED_REASON or str(e)) from e


def disabled_reason() -> Optional[str]:
    """Return the reason Vulkan is unavailable/disabled (if known)."""

    if _HAS_VULKAN:
        return _VULKAN_DISABLED_REASON
    return _VULKAN_DISABLED_REASON or "Vulkan bindings unavailable"


def self_test(*, strict: bool = True) -> dict[str, object]:
    """Run a fast backend self-test.

    - Compiles shaders if needed (via `glslc`).
    - Runs a small subset of correctness checks.

    When strict=True (default), this raises if Vulkan is unavailable.
    """

    init(strict=strict)
    if not _HAS_VULKAN:
        if strict:
            raise RuntimeError(disabled_reason() or "Vulkan unavailable")
        return {"available": False, "reason": disabled_reason()}

    rng = np.random.default_rng(0)

    # Elementwise: (x*y + x).relu()
    x = rng.standard_normal((33, 17), dtype=np.float32)
    y = rng.standard_normal((33, 17), dtype=np.float32)
    a = to_gpu(x)
    b = to_gpu(y)
    try:
        tmp = mul(a, b)
        tmp2 = add(tmp, a)
        out = relu(tmp2)
        try:
            got = to_cpu(out)
        finally:
            free(tmp)
            free(tmp2)
            free(out)
    finally:
        free(a)
        free(b)
    if not np.allclose(got, np.maximum(x * y + x, 0.0), rtol=2e-3, atol=1e-3):
        raise AssertionError("self_test: elemwise mismatch")

    # Matmul
    a_np = rng.standard_normal((17, 19), dtype=np.float32)
    b_np = rng.standard_normal((19, 23), dtype=np.float32)
    aa = to_gpu(a_np)
    bb = to_gpu(b_np)
    try:
        out = matmul(aa, bb)
        try:
            got = to_cpu(out)
        finally:
            free(out)
    finally:
        free(aa)
        free(bb)
    if not np.allclose(got, a_np @ b_np, rtol=2e-3, atol=1e-3):
        raise AssertionError("self_test: matmul mismatch")

    # Basic device info
    info: dict[str, object] = {"available": True}
    try:
        ctx = _ctx()
        info["queue_family_index"] = ctx.queue_family_index
    except Exception:
        pass

    return info


def to_gpu(data: np.ndarray) -> VulkanBuffer:
    """Upload a float32 NumPy array into a Vulkan storage buffer."""

    # NumPy 2.x may raise for np.array(..., copy=False) when a copy is required.
    # np.asarray preserves the old behavior (copy when needed).
    arr = np.asarray(data, dtype=np.float32)
    if not _HAS_VULKAN:
        return VulkanBuffer(
            shape=arr.shape,
            nbytes=int(arr.nbytes),
            buffer=0,
            memory=0,
            host=arr.copy(),
            base=None,
            refcount=1,
        )

    try:
        ctx = _ctx()
    except Exception as e:
        _disable_vulkan(str(e))
        return VulkanBuffer(
            shape=arr.shape,
            nbytes=int(arr.nbytes),
            buffer=0,
            memory=0,
            host=arr.copy(),
            base=None,
            refcount=1,
        )
    buf, mem = ctx.alloc_buffer(arr.nbytes)
    ctx.write_buffer(buf, mem, arr)
    return VulkanBuffer(
        shape=arr.shape,
        nbytes=int(arr.nbytes),
        buffer=buf,
        memory=mem,
        host=None,
        base=None,
        refcount=1,
    )


def to_cpu(buffer: VulkanBuffer) -> np.ndarray:
    """Download a Vulkan buffer back into a NumPy float32 array."""

    if buffer.host is not None:
        return np.array(buffer.host, copy=True)

    ctx = _ctx()
    raw = ctx.read_buffer(buffer.memory, buffer.nbytes)
    out = np.frombuffer(raw, dtype=np.float32).copy()
    return out.reshape(buffer.shape)


def empty(shape: tuple[int, ...]) -> VulkanBuffer:
    """Allocate an uninitialized float32 VulkanBuffer on device."""

    nbytes = int(np.prod(shape)) * 4
    if not _HAS_VULKAN:
        return VulkanBuffer(
            shape=shape,
            nbytes=nbytes,
            buffer=VK_NULL_HANDLE,  # type: ignore[name-defined]
            memory=VK_NULL_HANDLE,  # type: ignore[name-defined]
            host=np.empty(shape, dtype=np.float32),
            base=None,
            refcount=1,
        )

    ctx = _ctx()
    buf, mem = ctx.alloc_buffer(nbytes)
    return VulkanBuffer(shape=shape, nbytes=nbytes, buffer=buf, memory=mem, host=None, base=None, refcount=1)


def write(dst: VulkanBuffer, data: np.ndarray) -> None:
    """Upload a float32 NumPy array into an *existing* VulkanBuffer.

    This is the in-place counterpart to `to_gpu(...)` and is used to refill
    preallocated batch buffers without reallocating device memory.
    """

    arr = np.asarray(data, dtype=np.float32)
    if int(arr.nbytes) != int(dst.nbytes):
        raise ValueError(
            f"write: nbytes mismatch: data.nbytes={int(arr.nbytes)} dst.nbytes={int(dst.nbytes)} "
            f"data.shape={tuple(arr.shape)} dst.shape={tuple(dst.shape)}"
        )

    # Host/fallback buffers (including views) can be updated directly.
    if dst.host is not None or not _HAS_VULKAN:
        if dst.host is None:
            # Safety: if Vulkan is disabled but a host array wasn't allocated, create one.
            dst.host = np.empty(dst.shape, dtype=np.float32)
        np.copyto(dst.host, arr.reshape(dst.shape))
        return

    ctx = _ctx()
    ctx.write_buffer(dst.buffer, dst.memory, arr)


def free(buf: VulkanBuffer) -> None:
    """Free a VulkanBuffer's device resources (no-op for fallback/host buffers)."""

    # Host/fallback buffers don't need freeing.
    if buf.host is not None:
        return
    if not _HAS_VULKAN:
        return

    is_view = buf.base is not None
    base = buf.base if buf.base is not None else buf

    if base.refcount > 0:
        base.refcount -= 1

    # Views can be invalidated immediately; base buffers must keep handles alive
    # while views still exist.
    if is_view:
        buf.buffer = VK_NULL_HANDLE  # type: ignore[assignment]
        buf.memory = VK_NULL_HANDLE  # type: ignore[assignment]

    if base.refcount > 0:
        return

    if base.buffer == VK_NULL_HANDLE or base.memory == VK_NULL_HANDLE:
        return
    ctx = _ctx()
    ctx.free_buffer(base.buffer, base.memory)
    base.buffer = VK_NULL_HANDLE  # type: ignore[assignment]
    base.memory = VK_NULL_HANDLE  # type: ignore[assignment]


def view(buf: VulkanBuffer, shape: tuple[int, ...]) -> VulkanBuffer:
    """Create a lightweight shape view of an existing buffer.

    This shares the underlying VkBuffer and uses refcounting to avoid double-free.
    """

    n = int(np.prod(shape))
    if n * 4 != buf.nbytes:
        raise ValueError(f"view shape mismatch: shape={shape} nbytes={n*4} buf.nbytes={buf.nbytes}")

    base = buf.base if buf.base is not None else buf
    base.refcount += 1

    host_view = None
    if base.host is not None:
        host_view = base.host.reshape(shape)

    return VulkanBuffer(
        shape=tuple(shape),
        nbytes=int(buf.nbytes),
        buffer=base.buffer,
        memory=base.memory,
        host=host_view,
        base=base,
        refcount=1,
    )


def _fallback_buf(data: np.ndarray) -> VulkanBuffer:
    # Create a fake handle-like object when Vulkan is missing.
    arr = np.asarray(data, dtype=np.float32)
    return VulkanBuffer(
        shape=tuple(arr.shape),
        nbytes=int(arr.nbytes),
        buffer=VK_NULL_HANDLE,  # type: ignore[name-defined]
        memory=VK_NULL_HANDLE,  # type: ignore[name-defined]
        host=arr.copy(),
        base=None,
        refcount=1,
    )


def _ensure_vulkan_or_numpy(a: VulkanBuffer, b: Optional[VulkanBuffer] = None) -> bool:
    # If we have real Vulkan handles, use Vulkan. Otherwise caller should fallback.
    if not _HAS_VULKAN:
        return False
    if a.host is not None:
        return False
    if a.buffer == VK_NULL_HANDLE or a.memory == VK_NULL_HANDLE:
        return False
    if b is not None:
        if b.host is not None:
            return False
        if b.buffer == VK_NULL_HANDLE or b.memory == VK_NULL_HANDLE:
            return False
    return True


def add(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Elementwise add using Vulkan compute (falls back to NumPy if needed)."""

    if not _ensure_vulkan_or_numpy(a, b):
        if a.host is None or b.host is None:
            aa = np.zeros(a.shape, dtype=np.float32)
            bb = np.zeros(b.shape, dtype=np.float32)
            return _fallback_buf(aa + bb)
        return _fallback_buf(a.host + b.host)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_add is not None

    # Output buffer (no need to upload zeros)
    out = empty(a.shape)

    ds = ctx._alloc_descriptor_set(ctx.p_add)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    # Record commands
    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_add.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_add.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[1]", [n])
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_add.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def mul(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Elementwise multiply using Vulkan compute (falls back to NumPy if needed)."""

    if not _ensure_vulkan_or_numpy(a, b):
        if a.host is None or b.host is None:
            aa = np.zeros(a.shape, dtype=np.float32)
            bb = np.zeros(b.shape, dtype=np.float32)
            return _fallback_buf(aa * bb)
        return _fallback_buf(a.host * b.host)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mul is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_mul)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mul.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mul.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[1]", [n])
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mul.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def neg(a: VulkanBuffer) -> VulkanBuffer:
    """Elementwise negation."""

    if not _ensure_vulkan_or_numpy(a, None):
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        return _fallback_buf(-aa)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_neg is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_neg)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t *", int(n))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_neg.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_neg.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_neg.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def add_scalar(a: VulkanBuffer, s: float) -> VulkanBuffer:
    """Elementwise add a scalar: out = a + s."""

    if not _ensure_vulkan_or_numpy(a, None):
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        return _fallback_buf(aa + float(s))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_add_scalar is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_add_scalar)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<If", int(n), float(s)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_add_scalar.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_add_scalar.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_add_scalar.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def mul_scalar(a: VulkanBuffer, s: float) -> VulkanBuffer:
    """Elementwise multiply by scalar: out = a * s."""

    if not _ensure_vulkan_or_numpy(a, None):
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        return _fallback_buf(aa * float(s))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mul_scalar is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_mul_scalar)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<If", int(n), float(s)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mul_scalar.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mul_scalar.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mul_scalar.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def relu(a: VulkanBuffer) -> VulkanBuffer:
    """ReLU using Vulkan compute (falls back to NumPy if needed)."""

    if not _ensure_vulkan_or_numpy(a, None):
        if a.host is None:
            aa = np.zeros(a.shape, dtype=np.float32)
            return _fallback_buf(np.maximum(aa, 0))
        return _fallback_buf(np.maximum(a.host, 0))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_relu is not None

    out = empty(a.shape)

    ds = ctx._alloc_descriptor_set(ctx.p_relu)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_relu.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_relu.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[1]", [n])
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_relu.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def relu_out(a: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """ReLU writing into an existing output buffer."""

    if out.shape != a.shape:
        raise ValueError(f"relu_out shape mismatch: a={a.shape} out={out.shape}")

    if not _ensure_vulkan_or_numpy(a, None) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return relu(a)
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        out.host[...] = np.maximum(aa, 0)
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_relu is not None

    ds = ctx._alloc_descriptor_set(ctx.p_relu)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_relu.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_relu.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[1]", [n])
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_relu.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def mul_add_relu(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Fused elementwise (a*b + a) then ReLU using Vulkan compute."""

    if not _ensure_vulkan_or_numpy(a, b):
        if a.host is None or b.host is None:
            aa = np.zeros(a.shape, dtype=np.float32)
            bb = np.zeros(b.shape, dtype=np.float32)
            return _fallback_buf(np.maximum(aa * bb + aa, 0))
        return _fallback_buf(np.maximum(a.host * b.host + a.host, 0))

    if a.shape != b.shape:
        raise ValueError(f"mul_add_relu shape mismatch: {a.shape} vs {b.shape}")

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mul_add_relu is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_mul_add_relu)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mul_add_relu.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mul_add_relu.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t *", n)
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mul_add_relu.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    threads = (n + 3) // 4
    group_count_x = (threads + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def mul_add_relu_out(a: VulkanBuffer, b: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """Fused elementwise (a*b + a) then ReLU writing into an existing output buffer."""

    if a.shape != b.shape or out.shape != a.shape:
        raise ValueError(
            f"mul_add_relu_out shape mismatch: a={a.shape} b={b.shape} out={out.shape}"
        )

    if not _ensure_vulkan_or_numpy(a, b) or not _ensure_vulkan_or_numpy(out, None):
        # Fallback: compute on CPU and write into out.host.
        if out.host is None:
            return mul_add_relu(a, b)
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, dtype=np.float32)
        out.host[...] = np.maximum(aa * bb + aa, 0)
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mul_add_relu is not None

    ds = ctx._alloc_descriptor_set(ctx.p_mul_add_relu)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mul_add_relu.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mul_add_relu.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    n = int(np.prod(a.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t *", n)
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mul_add_relu.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    threads = (n + 3) // 4
    group_count_x = (threads + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def matmul_out(a: VulkanBuffer, b: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """Matmul writing into an existing output buffer."""

    if len(a.shape) != 2 or len(b.shape) != 2 or len(out.shape) != 2:
        raise ValueError("matmul_out expects 2D matrices")
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or out.shape != (m, n):
        raise ValueError(f"matmul_out shape mismatch: {a.shape} @ {b.shape} -> {out.shape}")

    if not _ensure_vulkan_or_numpy(a, b) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return matmul(a, b)
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, dtype=np.float32)
        out.host[...] = aa @ bb
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_matmul is not None

    ds = ctx._alloc_descriptor_set(ctx.p_matmul)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_matmul.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_matmul.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[3]", [int(m), int(n), int(k)])
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_matmul.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        12,
        pc,
    )
    group_count_x = (int(n) + 15) // 16
    group_count_y = (int(m) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    ctx._end_commands()
    return out


def add_rowvec(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Add a row-vector b (shape [cols]) to every row of matrix a ([rows, cols])."""

    if len(a.shape) != 2 or len(b.shape) != 1:
        raise ValueError("add_rowvec expects a:[rows,cols] and b:[cols]")
    rows, cols = a.shape
    if b.shape[0] != cols:
        raise ValueError(f"add_rowvec shape mismatch: a={a.shape} b={b.shape}")

    if not _ensure_vulkan_or_numpy(a, b):
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, dtype=np.float32)
        return _fallback_buf(aa + bb)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_add_rowvec is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_add_rowvec)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(rows * cols)
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[2]", [int(n), int(cols)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_add_rowvec.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_add_rowvec.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_add_rowvec.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def add_rowvec_out(a: VulkanBuffer, b: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """Add row-vector b to every row of a, writing into out."""

    if len(a.shape) != 2 or len(b.shape) != 1 or len(out.shape) != 2:
        raise ValueError("add_rowvec_out expects a:[rows,cols], b:[cols], out:[rows,cols]")
    rows, cols = a.shape
    if b.shape[0] != cols or out.shape != a.shape:
        raise ValueError(f"add_rowvec_out shape mismatch: a={a.shape} b={b.shape} out={out.shape}")

    if not _ensure_vulkan_or_numpy(a, b) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return add_rowvec(a, b)
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, dtype=np.float32)
        out.host[...] = aa + bb
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_add_rowvec is not None

    ds = ctx._alloc_descriptor_set(ctx.p_add_rowvec)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(rows * cols)
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[2]", [int(n), int(cols)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_add_rowvec.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_add_rowvec.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_add_rowvec.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def mul_rowvec(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Multiply a matrix a ([rows, cols]) by a row-vector b ([cols]) with broadcasting."""

    if len(a.shape) != 2 or len(b.shape) != 1:
        raise ValueError("mul_rowvec expects a:[rows,cols] and b:[cols]")
    rows, cols = a.shape
    if b.shape[0] != cols:
        raise ValueError(f"mul_rowvec shape mismatch: a={a.shape} b={b.shape}")

    if not _ensure_vulkan_or_numpy(a, b):
        aa = a.host if a.host is not None else np.zeros(a.shape, dtype=np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, dtype=np.float32)
        return _fallback_buf(aa * bb)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mul_rowvec is not None

    out = empty(a.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_mul_rowvec)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(rows * cols)
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[2]", [int(n), int(cols)])

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mul_rowvec.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mul_rowvec.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mul_rowvec.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(n) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def im2col_nchw(
    x: VulkanBuffer,
    *,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
) -> VulkanBuffer:
    """Convert NCHW image tensor into a 2D im2col matrix.

    x: [N,C,H,W] -> out: [N*OH*OW, C*KH*KW]
    """

    if len(x.shape) != 4:
        raise ValueError("im2col_nchw expects x shape [N,C,H,W]")
    N, C, H, W = x.shape
    KH, KW = int(kh), int(kw)
    SH, SW = int(stride_h), int(stride_w)
    PH, PW = int(pad_h), int(pad_w)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    out_shape = (int(N * OH * OW), int(C * KH * KW))
    total = int(np.prod(out_shape))

    if not _ensure_vulkan_or_numpy(x, None):
        xx = x.host if x.host is not None else np.zeros(x.shape, np.float32)
        out = np.zeros(out_shape, np.float32)
        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    row = n * (OH * OW) + oh * OW + ow
                    for c in range(C):
                        for rkh in range(KH):
                            for rkw in range(KW):
                                ih = oh * SH + rkh - PH
                                iw = ow * SW + rkw - PW
                                col = c * (KH * KW) + rkh * KW + rkw
                                if 0 <= ih < H and 0 <= iw < W:
                                    out[row, col] = xx[n, c, ih, iw]
        return _fallback_buf(out)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_im2col_nchw is not None

    out = empty(out_shape)
    ds = ctx._alloc_descriptor_set(ctx.p_im2col_nchw)

    infos = [
        VkDescriptorBufferInfo(buffer=x.buffer, offset=0, range=x.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new(
        "uint32_t[16]",
        [
            int(total),
            int(N),
            int(C),
            int(H),
            int(W),
            int(KH),
            int(KW),
            int(SH),
            int(SW),
            int(PH),
            int(PW),
            0,
            0,
            0,
            0,
            0,
        ],
    )

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_im2col_nchw.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_im2col_nchw.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_im2col_nchw.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        64,
        pc,
    )
    group_count_x = (int(total) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def col2im_nchw(
    col: VulkanBuffer,
    *,
    out_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
) -> VulkanBuffer:
    """Convert an im2col matrix back into an NCHW tensor by summing overlaps."""

    N, C, H, W = out_shape
    KH, KW = int(kh), int(kw)
    SH, SW = int(stride_h), int(stride_w)
    PH, PW = int(pad_h), int(pad_w)

    out = empty(out_shape)
    total = int(np.prod(out_shape))

    if not _ensure_vulkan_or_numpy(col, None):
        cc = col.host if col.host is not None else np.zeros(col.shape, np.float32)
        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1
        K = C * KH * KW
        xx = np.zeros(out_shape, np.float32)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        acc = 0.0
                        for rkh in range(KH):
                            for rkw in range(KW):
                                oh_i = h + PH - rkh
                                ow_i = w + PW - rkw
                                if oh_i < 0 or ow_i < 0:
                                    continue
                                if (oh_i % SH) != 0 or (ow_i % SW) != 0:
                                    continue
                                oh = oh_i // SH
                                ow = ow_i // SW
                                if oh < 0 or ow < 0 or oh >= OH or ow >= OW:
                                    continue
                                row = n * (OH * OW) + oh * OW + ow
                                col_idx = c * (KH * KW) + rkh * KW + rkw
                                acc += cc[row, col_idx]
                        xx[n, c, h, w] = acc
        return _fallback_buf(xx)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_col2im_nchw is not None

    ds = ctx._alloc_descriptor_set(ctx.p_col2im_nchw)
    infos = [
        VkDescriptorBufferInfo(buffer=col.buffer, offset=0, range=col.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new(
        "uint32_t[16]",
        [
            int(total),
            int(N),
            int(C),
            int(H),
            int(W),
            int(KH),
            int(KW),
            int(SH),
            int(SW),
            int(PH),
            int(PW),
            0,
            0,
            0,
            0,
            0,
        ],
    )

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_col2im_nchw.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_col2im_nchw.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_col2im_nchw.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        64,
        pc,
    )
    group_count_x = (int(total) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out


def nchw2mat(x: VulkanBuffer) -> VulkanBuffer:
    """Convert [N,C,H,W] to [N*H*W, C]"""
    if len(x.shape) != 4:
        raise ValueError("nchw2mat expects x shape [N,C,H,W]")
    N, C, H, W = x.shape
    out_shape = (int(N * H * W), int(C))
    total = int(np.prod(out_shape))

    if not _ensure_vulkan_or_numpy(x, None):
        xx = x.host if x.host is not None else np.zeros(x.shape, np.float32)
        out = np.zeros(out_shape, np.float32)
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    row = n * (H * W) + h * W + w
                    out[row, :] = xx[n, :, h, w]
        return _fallback_buf(out)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_nchw2mat is not None

    out = empty(out_shape)
    ds = ctx._alloc_descriptor_set(ctx.p_nchw2mat)

    infos = [
        VkDescriptorBufferInfo(buffer=x.buffer, offset=0, range=x.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    pc = _vk.ffi.new(
        "uint32_t[16]",
        [int(total), int(N), int(C), int(H), int(W)] + [0] * 11,
    )

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_nchw2mat.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_nchw2mat.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_nchw2mat.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        64,
        pc,
    )
    group_count_x = (int(total) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def mat2nchw(mat: VulkanBuffer, *, out_shape: tuple[int, int, int, int]) -> VulkanBuffer:
    """Convert [N*H*W, C] to [N,C,H,W]"""
    if len(mat.shape) != 2:
        raise ValueError("mat2nchw expects mat shape [N*H*W, C]")
    N, C, H, W = out_shape
    if mat.shape != (int(N * H * W), int(C)):
        raise ValueError(f"mat2nchw shape mismatch: mat={mat.shape} out_shape={out_shape}")

    total = int(np.prod(out_shape))
    out = empty(out_shape)

    if not _ensure_vulkan_or_numpy(mat, None):
        mm = mat.host if mat.host is not None else np.zeros(mat.shape, np.float32)
        xx = np.zeros(out_shape, np.float32)
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    row = n * (H * W) + h * W + w
                    xx[n, :, h, w] = mm[row, :]
        return _fallback_buf(xx)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mat2nchw is not None

    ds = ctx._alloc_descriptor_set(ctx.p_mat2nchw)
    infos = [
        VkDescriptorBufferInfo(buffer=mat.buffer, offset=0, range=mat.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    pc = _vk.ffi.new(
        "uint32_t[16]",
        [int(total), int(N), int(C), int(H), int(W)] + [0] * 11,
    )

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mat2nchw.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mat2nchw.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mat2nchw.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        64,
        pc,
    )
    group_count_x = (int(total) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def softmax_xent_loss_vec(logits: VulkanBuffer, target: VulkanBuffer) -> VulkanBuffer:
    """Per-sample softmax cross entropy.

    logits: [N,C], target: [N,C] (one-hot or probabilities) -> out: [N]
    """

    if len(logits.shape) != 2 or logits.shape != target.shape:
        raise ValueError(f"softmax_xent_loss_vec expects logits and target both [N,C]; got {logits.shape} and {target.shape}")
    N, C = logits.shape

    if not _ensure_vulkan_or_numpy(logits, target):
        l = logits.host if logits.host is not None else np.zeros(logits.shape, np.float32)
        t = target.host if target.host is not None else np.zeros(target.shape, np.float32)
        m = l.max(axis=1, keepdims=True)
        z = l - m
        logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True)) + m
        loss = -(t * (l - logsumexp)).sum(axis=1)
        return _fallback_buf(loss.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_softmax_xent_loss_vec is not None

    out = empty((int(N),))
    ds = ctx._alloc_descriptor_set(ctx.p_softmax_xent_loss_vec)

    infos = [
        VkDescriptorBufferInfo(buffer=logits.buffer, offset=0, range=logits.nbytes),
        VkDescriptorBufferInfo(buffer=target.buffer, offset=0, range=target.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_softmax_xent_loss_vec.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_softmax_xent_loss_vec.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_softmax_xent_loss_vec.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def softmax_xent_backward(logits: VulkanBuffer, target: VulkanBuffer) -> VulkanBuffer:
    """Gradient of mean softmax cross entropy w.r.t logits.

    Returns grad logits: [N,C] scaled by 1/N.
    """

    if len(logits.shape) != 2 or logits.shape != target.shape:
        raise ValueError(f"softmax_xent_backward expects logits and target both [N,C]; got {logits.shape} and {target.shape}")
    N, C = logits.shape

    if not _ensure_vulkan_or_numpy(logits, target):
        l = logits.host if logits.host is not None else np.zeros(logits.shape, np.float32)
        t = target.host if target.host is not None else np.zeros(target.shape, np.float32)
        m = l.max(axis=1, keepdims=True)
        z = l - m
        e = np.exp(z)
        s = e / e.sum(axis=1, keepdims=True)
        g = (s - t) / max(1.0, float(N))
        return _fallback_buf(g.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_softmax_xent_backward is not None

    out = empty(logits.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_softmax_xent_backward)

    infos = [
        VkDescriptorBufferInfo(buffer=logits.buffer, offset=0, range=logits.nbytes),
        VkDescriptorBufferInfo(buffer=target.buffer, offset=0, range=target.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_softmax_xent_backward.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_softmax_xent_backward.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_softmax_xent_backward.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def softmax2d(logits: VulkanBuffer) -> VulkanBuffer:
    """Row-wise softmax for 2D logits.

    logits: [N,C] -> out: [N,C]
    """

    if len(logits.shape) != 2:
        raise ValueError("softmax2d expects logits with shape [N,C]")
    N, C = logits.shape

    if not _ensure_vulkan_or_numpy(logits, None):
        l = logits.host if logits.host is not None else np.zeros(logits.shape, np.float32)
        m = l.max(axis=1, keepdims=True)
        z = l - m
        e = np.exp(z)
        s = e / e.sum(axis=1, keepdims=True)
        return _fallback_buf(s.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_softmax2d is not None

    out = empty(logits.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_softmax2d)

    infos = [
        VkDescriptorBufferInfo(buffer=logits.buffer, offset=0, range=logits.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_softmax2d.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_softmax2d.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_softmax2d.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def softmax2d_backward(y: VulkanBuffer, grad_out: VulkanBuffer) -> VulkanBuffer:
    """Backward for row-wise softmax.

    y: [N,C] (softmax output), grad_out: [N,C] -> grad_in: [N,C]
    """

    if len(y.shape) != 2 or y.shape != grad_out.shape:
        raise ValueError(f"softmax2d_backward expects y and grad_out both [N,C]; got {y.shape} and {grad_out.shape}")
    N, C = y.shape

    if not _ensure_vulkan_or_numpy(y, grad_out):
        yy = y.host if y.host is not None else np.zeros(y.shape, np.float32)
        go = grad_out.host if grad_out.host is not None else np.zeros(grad_out.shape, np.float32)
        dot = (go * yy).sum(axis=1, keepdims=True)
        dx = yy * (go - dot)
        return _fallback_buf(dx.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_softmax2d_backward is not None

    out = empty(y.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_softmax2d_backward)

    infos = [
        VkDescriptorBufferInfo(buffer=y.buffer, offset=0, range=y.nbytes),
        VkDescriptorBufferInfo(buffer=grad_out.buffer, offset=0, range=grad_out.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_softmax2d_backward.pipeline,
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_softmax2d_backward.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_softmax2d_backward.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def log_softmax2d(logits: VulkanBuffer) -> VulkanBuffer:
    """Row-wise log_softmax for 2D logits.

    logits: [N,C] -> out: [N,C]
    """

    if len(logits.shape) != 2:
        raise ValueError("log_softmax2d expects logits with shape [N,C]")
    N, C = logits.shape

    if not _ensure_vulkan_or_numpy(logits, None):
        l = logits.host if logits.host is not None else np.zeros(logits.shape, np.float32)
        m = l.max(axis=1, keepdims=True)
        z = l - m
        lse = np.log(np.exp(z).sum(axis=1, keepdims=True)) + m
        out_np = l - lse
        return _fallback_buf(out_np.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_log_softmax2d is not None

    out = empty(logits.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_log_softmax2d)

    infos = [
        VkDescriptorBufferInfo(buffer=logits.buffer, offset=0, range=logits.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_log_softmax2d.pipeline,
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_log_softmax2d.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_log_softmax2d.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def log_softmax2d_backward(logp: VulkanBuffer, grad_out: VulkanBuffer) -> VulkanBuffer:
    """Backward for row-wise log_softmax.

    logp: [N,C] (log_softmax output), grad_out: [N,C] -> grad_in: [N,C]
    """

    if len(logp.shape) != 2 or logp.shape != grad_out.shape:
        raise ValueError(
            f"log_softmax2d_backward expects logp and grad_out both [N,C]; got {logp.shape} and {grad_out.shape}"
        )
    N, C = logp.shape

    if not _ensure_vulkan_or_numpy(logp, grad_out):
        lp = logp.host if logp.host is not None else np.zeros(logp.shape, np.float32)
        go = grad_out.host if grad_out.host is not None else np.zeros(grad_out.shape, np.float32)
        s = np.exp(lp)
        sumg = go.sum(axis=1, keepdims=True)
        dx = go - s * sumg
        return _fallback_buf(dx.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_log_softmax2d_backward is not None

    out = empty(logp.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_log_softmax2d_backward)

    infos = [
        VkDescriptorBufferInfo(buffer=logp.buffer, offset=0, range=logp.nbytes),
        VkDescriptorBufferInfo(buffer=grad_out.buffer, offset=0, range=grad_out.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_log_softmax2d_backward.pipeline,
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_log_softmax2d_backward.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_log_softmax2d_backward.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def layernorm2d(x: VulkanBuffer) -> VulkanBuffer:
    """LayerNorm for 2D input over the last dimension (eps=1e-5).

    x: [N,C] -> out: [N,C]
    """

    if len(x.shape) != 2:
        raise ValueError("layernorm2d expects x with shape [N,C]")
    N, C = x.shape

    if not _ensure_vulkan_or_numpy(x, None):
        xx = x.host if x.host is not None else np.zeros(x.shape, np.float32)
        mean = xx.mean(axis=1, keepdims=True)
        var = ((xx - mean) ** 2).mean(axis=1, keepdims=True)
        out_np = (xx - mean) / np.sqrt(var + 1e-5)
        return _fallback_buf(out_np.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_layernorm2d is not None

    out = empty(x.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_layernorm2d)

    infos = [
        VkDescriptorBufferInfo(buffer=x.buffer, offset=0, range=x.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_layernorm2d.pipeline,
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_layernorm2d.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_layernorm2d.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def layernorm2d_backward(x: VulkanBuffer, grad_out: VulkanBuffer) -> VulkanBuffer:
    """Backward for layernorm2d (eps=1e-5).

    Inputs:
    - x: [N,C]
    - grad_out: [N,C] gradient w.r.t normalized output (xhat)
    Output:
    - grad_x: [N,C]
    """

    if len(x.shape) != 2 or len(grad_out.shape) != 2:
        raise ValueError("layernorm2d_backward expects x and grad_out with shape [N,C]")
    if x.shape != grad_out.shape:
        raise ValueError(f"layernorm2d_backward shape mismatch: {x.shape} vs {grad_out.shape}")
    N, C = x.shape

    if not _ensure_vulkan_or_numpy(x, grad_out):
        xx = x.host if x.host is not None else np.zeros(x.shape, np.float32)
        gg = grad_out.host if grad_out.host is not None else np.zeros(x.shape, np.float32)

        mean = xx.mean(axis=1, keepdims=True)
        var = ((xx - mean) ** 2).mean(axis=1, keepdims=True)
        invstd = 1.0 / np.sqrt(var + 1e-5)
        xhat = (xx - mean) * invstd

        sum1 = gg.sum(axis=1, keepdims=True)
        sum2 = (gg * xhat).sum(axis=1, keepdims=True)
        cc = max(1.0, float(C))
        dx = (invstd / cc) * (gg * cc - sum1 - xhat * sum2)
        return _fallback_buf(dx.astype(np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_layernorm2d_backward is not None

    out = empty(x.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_layernorm2d_backward)

    infos = [
        VkDescriptorBufferInfo(buffer=x.buffer, offset=0, range=x.nbytes),
        VkDescriptorBufferInfo(buffer=grad_out.buffer, offset=0, range=grad_out.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<II", int(N), int(C)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_layernorm2d_backward.pipeline,
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_layernorm2d_backward.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_layernorm2d_backward.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(N) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def sgd_momentum_update_inplace(
    param: VulkanBuffer,
    grad: VulkanBuffer,
    velocity: VulkanBuffer,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> None:
    """In-place momentum SGD update.

    velocity = momentum*velocity + grad + weight_decay*param
    param -= lr*velocity
    """

    if param.shape != grad.shape or param.shape != velocity.shape:
        raise ValueError(f"sgd_momentum_update_inplace shape mismatch: {param.shape}, {grad.shape}, {velocity.shape}")

    if not _ensure_vulkan_or_numpy(param, grad):
        if param.host is None:
            return
        g = grad.host if grad.host is not None else np.zeros(grad.shape, np.float32)
        v = velocity.host if velocity.host is not None else np.zeros(velocity.shape, np.float32)
        v[:] = float(momentum) * v + g + float(weight_decay) * param.host
        param.host[:] = param.host - float(lr) * v
        velocity.host = v
        return

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_sgd_momentum_update is not None

    ds = ctx._alloc_descriptor_set(ctx.p_sgd_momentum_update)
    infos = [
        VkDescriptorBufferInfo(buffer=param.buffer, offset=0, range=param.nbytes),
        VkDescriptorBufferInfo(buffer=grad.buffer, offset=0, range=grad.nbytes),
        VkDescriptorBufferInfo(buffer=velocity.buffer, offset=0, range=velocity.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore
    import struct

    n = int(np.prod(param.shape))
    pc = _vk.ffi.new(
        "char[]",
        struct.pack(
            "<Ifff",
            int(n),
            float(lr),
            float(momentum),
            float(weight_decay),
        ),
    )

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_sgd_momentum_update.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_sgd_momentum_update.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_sgd_momentum_update.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        16,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()


def relu_backward(grad_out: VulkanBuffer, x: VulkanBuffer) -> VulkanBuffer:
    """Compute grad_in = grad_out * (x > 0)."""

    if grad_out.shape != x.shape:
        raise ValueError(f"relu_backward shape mismatch: {grad_out.shape} vs {x.shape}")

    if not _ensure_vulkan_or_numpy(grad_out, x):
        gg = grad_out.host if grad_out.host is not None else np.zeros(grad_out.shape, np.float32)
        xx = x.host if x.host is not None else np.zeros(x.shape, np.float32)
        return _fallback_buf(gg * (xx > 0))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_relu_backward is not None

    out = empty(x.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_relu_backward)

    infos = [
        VkDescriptorBufferInfo(buffer=grad_out.buffer, offset=0, range=grad_out.nbytes),
        VkDescriptorBufferInfo(buffer=x.buffer, offset=0, range=x.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(x.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t *", n)

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_relu_backward.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_relu_backward.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_relu_backward.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def relu_backward_out(grad_out: VulkanBuffer, x: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """ReLU backward writing into an existing output buffer."""

    if grad_out.shape != x.shape or out.shape != x.shape:
        raise ValueError(
            f"relu_backward_out shape mismatch: grad_out={grad_out.shape} x={x.shape} out={out.shape}"
        )

    if not _ensure_vulkan_or_numpy(grad_out, x) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return relu_backward(grad_out, x)
        gg = grad_out.host if grad_out.host is not None else np.zeros(grad_out.shape, np.float32)
        xx = x.host if x.host is not None else np.zeros(x.shape, np.float32)
        out.host[...] = gg * (xx > 0)
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_relu_backward is not None

    ds = ctx._alloc_descriptor_set(ctx.p_relu_backward)
    infos = [
        VkDescriptorBufferInfo(buffer=grad_out.buffer, offset=0, range=grad_out.nbytes),
        VkDescriptorBufferInfo(buffer=x.buffer, offset=0, range=x.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(x.shape))
    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t *", n)

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_relu_backward.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_relu_backward.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_relu_backward.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def reduce_sum_rows(a: VulkanBuffer) -> VulkanBuffer:
    """Sum a matrix over rows (axis=0). a:[rows,cols] -> out:[cols]."""

    if len(a.shape) != 2:
        raise ValueError("reduce_sum_rows expects a 2D matrix")
    rows, cols = a.shape

    if not _ensure_vulkan_or_numpy(a, None):
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        return _fallback_buf(aa.sum(axis=0))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_reduce_sum_rows is not None

    out = empty((cols,))
    ds = ctx._alloc_descriptor_set(ctx.p_reduce_sum_rows)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[2]", [int(rows), int(cols)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_reduce_sum_rows.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_reduce_sum_rows.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_reduce_sum_rows.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(cols) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def reduce_sum_rows_out(a: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """Sum a matrix over rows into an existing output buffer."""

    if len(a.shape) != 2 or len(out.shape) != 1:
        raise ValueError("reduce_sum_rows_out expects a:[rows,cols] and out:[cols]")
    rows, cols = a.shape
    if out.shape != (cols,):
        raise ValueError(f"reduce_sum_rows_out shape mismatch: a={a.shape} out={out.shape}")

    if not _ensure_vulkan_or_numpy(a, None) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return reduce_sum_rows(a)
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        out.host[...] = aa.sum(axis=0)
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_reduce_sum_rows is not None

    ds = ctx._alloc_descriptor_set(ctx.p_reduce_sum_rows)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[2]", [int(rows), int(cols)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_reduce_sum_rows.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_reduce_sum_rows.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_reduce_sum_rows.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(cols) + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def transpose2d(a: VulkanBuffer) -> VulkanBuffer:
    """Transpose a 2D matrix. a:[rows,cols] -> out:[cols,rows]."""

    if len(a.shape) != 2:
        raise ValueError("transpose2d expects a 2D matrix")
    rows, cols = a.shape

    if not _ensure_vulkan_or_numpy(a, None):
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        return _fallback_buf(aa.T)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_transpose2d is not None

    out = empty((cols, rows))
    ds = ctx._alloc_descriptor_set(ctx.p_transpose2d)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[2]", [int(rows), int(cols)])

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_transpose2d.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_transpose2d.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_transpose2d.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (int(cols) + 15) // 16
    group_count_y = (int(rows) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def scale_fill(scalar: VulkanBuffer, out: VulkanBuffer, scale: float) -> VulkanBuffer:
    """Fill out with scalar[0] * scale."""

    if not _ensure_vulkan_or_numpy(scalar, None) or not _ensure_vulkan_or_numpy(out, None):
        s = 0.0
        if scalar.host is not None:
            s = float(np.asarray(scalar.host).reshape(-1)[0])
        elif not _HAS_VULKAN:
            s = float(np.asarray(to_cpu(scalar)).reshape(-1)[0])
        arr = np.empty(out.shape, dtype=np.float32)
        arr[...] = s * float(scale)
        return _fallback_buf(arr)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_scale_fill is not None

    ds = ctx._alloc_descriptor_set(ctx.p_scale_fill)
    infos = [
        VkDescriptorBufferInfo(buffer=scalar.buffer, offset=0, range=scalar.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(out.shape))
    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<If", int(n), float(scale)))

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_scale_fill.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_scale_fill.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_scale_fill.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    vkEndCommandBuffer(ctx.command_buffer)
    ctx._submit_and_wait()
    return out


def reduce_sum(a: VulkanBuffer) -> VulkanBuffer:
    """Reduce all elements to a single scalar buffer of shape (1,)."""

    n = int(np.prod(a.shape))
    if n == 0:
        return _fallback_buf(np.array([0.0], dtype=np.float32))

    if not _ensure_vulkan_or_numpy(a, None):
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        return _fallback_buf(np.array([float(np.sum(aa))], dtype=np.float32))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_reduce_sum_stage is not None

    cur = a
    cur_n = n
    intermediates: list[VulkanBuffer] = []
    try:
        while cur_n > 1:
            out_len = (cur_n + 511) // 512
            out = empty((out_len,))
            intermediates.append(out)

            ds = ctx._alloc_descriptor_set(ctx.p_reduce_sum_stage)
            infos = [
                VkDescriptorBufferInfo(buffer=cur.buffer, offset=0, range=cur.nbytes),
                VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
            ]
            writes = [
                VkWriteDescriptorSet(
                    sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=ds,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[infos[0]],
                ),
                VkWriteDescriptorSet(
                    sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=ds,
                    dstBinding=1,
                    descriptorCount=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[infos[1]],
                ),
            ]
            vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

            import vulkan as _vk  # type: ignore

            pc = _vk.ffi.new("uint32_t *", int(cur_n))

            vkResetCommandBuffer(ctx.command_buffer, 0)
            begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
            vkBeginCommandBuffer(ctx.command_buffer, begin)
            vkCmdBindPipeline(
                ctx.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                ctx.p_reduce_sum_stage.pipeline,
            )
            vkCmdBindDescriptorSets(
                ctx.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                ctx.p_reduce_sum_stage.pipeline_layout,
                0,
                1,
                [ds],
                0,
                None,
            )
            vkCmdPushConstants(
                ctx.command_buffer,
                ctx.p_reduce_sum_stage.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                4,
                pc,
            )
            vkCmdDispatch(ctx.command_buffer, int(out_len), 1, 1)
            vkEndCommandBuffer(ctx.command_buffer)
            ctx._submit_and_wait()

            cur = out
            cur_n = out_len

        # Ensure output is shape (1,)
        if cur.shape != (1,):
            # This only happens if n==0; already handled above.
            pass
        return cur
    finally:
        # Caller owns the returned buffer; free the other intermediates.
        for buf in intermediates[:-1]:
            free(buf)


def mean(a: VulkanBuffer) -> VulkanBuffer:
    """Mean of all elements as a scalar buffer of shape (1,)."""

    n = int(np.prod(a.shape))
    if n == 0:
        return _fallback_buf(np.array([0.0], dtype=np.float32))
    s = reduce_sum(a)
    try:
        out = empty((1,))
        return scale_fill(s, out, 1.0 / float(n))
    finally:
        free(s)


def mse_grad(pred: VulkanBuffer, target: VulkanBuffer) -> VulkanBuffer:
    """Gradient of mean squared error w.r.t pred (elementwise)."""

    if pred.shape != target.shape:
        raise ValueError(f"mse_grad shape mismatch: {pred.shape} vs {target.shape}")

    if not _ensure_vulkan_or_numpy(pred, target):
        pp = pred.host if pred.host is not None else np.zeros(pred.shape, np.float32)
        tt = target.host if target.host is not None else np.zeros(target.shape, np.float32)
        n = pp.size
        return _fallback_buf(2.0 * (pp - tt) / max(1, n))

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mse_grad is not None

    out = empty(pred.shape)
    ds = ctx._alloc_descriptor_set(ctx.p_mse_grad)

    infos = [
        VkDescriptorBufferInfo(buffer=pred.buffer, offset=0, range=pred.nbytes),
        VkDescriptorBufferInfo(buffer=target.buffer, offset=0, range=target.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(pred.shape))
    inv_n = 1.0 / float(max(1, n))
    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<If", int(n), float(inv_n)))

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mse_grad.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mse_grad.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mse_grad.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def mse_grad_out(pred: VulkanBuffer, target: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """MSE gradient writing into an existing output buffer."""

    if pred.shape != target.shape or out.shape != pred.shape:
        raise ValueError(
            f"mse_grad_out shape mismatch: pred={pred.shape} target={target.shape} out={out.shape}"
        )

    if not _ensure_vulkan_or_numpy(pred, target) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return mse_grad(pred, target)
        pp = pred.host if pred.host is not None else np.zeros(pred.shape, np.float32)
        tt = target.host if target.host is not None else np.zeros(target.shape, np.float32)
        n = pp.size
        out.host[...] = 2.0 * (pp - tt) / max(1, n)
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_mse_grad is not None

    ds = ctx._alloc_descriptor_set(ctx.p_mse_grad)
    infos = [
        VkDescriptorBufferInfo(buffer=pred.buffer, offset=0, range=pred.nbytes),
        VkDescriptorBufferInfo(buffer=target.buffer, offset=0, range=target.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(pred.shape))
    inv_n = 1.0 / float(max(1, n))
    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<If", int(n), float(inv_n)))

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_mse_grad.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_mse_grad.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_mse_grad.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()
    return out


def sgd_update_inplace(param: VulkanBuffer, grad: VulkanBuffer, lr: float) -> None:
    """In-place SGD update: param -= lr * grad."""

    if param.shape != grad.shape:
        raise ValueError(f"sgd_update_inplace shape mismatch: {param.shape} vs {grad.shape}")

    if not _ensure_vulkan_or_numpy(param, grad):
        if param.host is None:
            return
        gg = grad.host if grad.host is not None else np.zeros(grad.shape, np.float32)
        param.host[...] = param.host - float(lr) * gg
        return

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_sgd_update is not None

    ds = ctx._alloc_descriptor_set(ctx.p_sgd_update)
    infos = [
        VkDescriptorBufferInfo(buffer=param.buffer, offset=0, range=param.nbytes),
        VkDescriptorBufferInfo(buffer=grad.buffer, offset=0, range=grad.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    n = int(np.prod(param.shape))
    import vulkan as _vk  # type: ignore
    import struct

    pc = _vk.ffi.new("char[]", struct.pack("<If", int(n), float(lr)))

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_sgd_update.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_sgd_update.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_sgd_update.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        8,
        pc,
    )
    group_count_x = (n + 255) // 256
    vkCmdDispatch(ctx.command_buffer, group_count_x, 1, 1)
    ctx._end_commands()


def matmul_at_b(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Compute A^T @ B. A:[m,k], B:[m,n] -> out:[k,n]."""

    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul_at_b expects 2D matrices")
    m, k = a.shape
    m2, n = b.shape
    if m != m2:
        raise ValueError(f"matmul_at_b shape mismatch: {a.shape} vs {b.shape}")

    if not _ensure_vulkan_or_numpy(a, b):
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, np.float32)
        return _fallback_buf(aa.T @ bb)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_matmul_at_b is not None

    out = empty((k, n))
    ds = ctx._alloc_descriptor_set(ctx.p_matmul_at_b)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[3]", [int(m), int(n), int(k)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_matmul_at_b.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_matmul_at_b.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_matmul_at_b.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        12,
        pc,
    )
    group_count_x = (int(n) + 15) // 16
    group_count_y = (int(k) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    ctx._end_commands()
    return out


def matmul_at_b_out(a: VulkanBuffer, b: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """Compute A^T @ B writing into an existing output buffer."""

    if len(a.shape) != 2 or len(b.shape) != 2 or len(out.shape) != 2:
        raise ValueError("matmul_at_b_out expects 2D matrices")
    m, k = a.shape
    m2, n = b.shape
    if m != m2 or out.shape != (k, n):
        raise ValueError(f"matmul_at_b_out shape mismatch: a={a.shape} b={b.shape} out={out.shape}")

    if not _ensure_vulkan_or_numpy(a, b) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return matmul_at_b(a, b)
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, np.float32)
        out.host[...] = aa.T @ bb
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_matmul_at_b is not None

    ds = ctx._alloc_descriptor_set(ctx.p_matmul_at_b)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[3]", [int(m), int(n), int(k)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_matmul_at_b.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_matmul_at_b.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_matmul_at_b.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        12,
        pc,
    )
    group_count_x = (int(n) + 15) // 16
    group_count_y = (int(k) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    ctx._end_commands()
    return out


def matmul_a_bt(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Compute A @ B^T. A:[m,k], B:[n,k] -> out:[m,n]."""

    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul_a_bt expects 2D matrices")
    m, k = a.shape
    n, k2 = b.shape
    if k != k2:
        raise ValueError(f"matmul_a_bt shape mismatch: {a.shape} vs {b.shape}")

    if not _ensure_vulkan_or_numpy(a, b):
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, np.float32)
        return _fallback_buf(aa @ bb.T)

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_matmul_a_bt is not None

    out = empty((m, n))
    ds = ctx._alloc_descriptor_set(ctx.p_matmul_a_bt)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[3]", [int(m), int(n), int(k)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_matmul_a_bt.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_matmul_a_bt.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_matmul_a_bt.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        12,
        pc,
    )
    group_count_x = (int(n) + 15) // 16
    group_count_y = (int(m) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    ctx._end_commands()
    return out


def matmul_a_bt_out(a: VulkanBuffer, b: VulkanBuffer, out: VulkanBuffer) -> VulkanBuffer:
    """Compute A @ B^T writing into an existing output buffer."""

    if len(a.shape) != 2 or len(b.shape) != 2 or len(out.shape) != 2:
        raise ValueError("matmul_a_bt_out expects 2D matrices")
    m, k = a.shape
    n, k2 = b.shape
    if k != k2 or out.shape != (m, n):
        raise ValueError(f"matmul_a_bt_out shape mismatch: a={a.shape} b={b.shape} out={out.shape}")

    if not _ensure_vulkan_or_numpy(a, b) or not _ensure_vulkan_or_numpy(out, None):
        if out.host is None:
            return matmul_a_bt(a, b)
        aa = a.host if a.host is not None else np.zeros(a.shape, np.float32)
        bb = b.host if b.host is not None else np.zeros(b.shape, np.float32)
        out.host[...] = aa @ bb.T
        return out

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_matmul_a_bt is not None

    ds = ctx._alloc_descriptor_set(ctx.p_matmul_a_bt)
    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[3]", [int(m), int(n), int(k)])

    ctx._maybe_continue_batch()
    if ctx._batch_depth == 0:
        ctx._begin_commands()
    vkCmdBindPipeline(
        ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_matmul_a_bt.pipeline
    )
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_matmul_a_bt.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_matmul_a_bt.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        12,
        pc,
    )
    group_count_x = (int(n) + 15) // 16
    group_count_y = (int(m) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    ctx._end_commands()
    return out


def matmul(a: VulkanBuffer, b: VulkanBuffer) -> VulkanBuffer:
    """Naive matmul using Vulkan compute (falls back to NumPy if needed)."""

    if not _ensure_vulkan_or_numpy(a, b):
        if a.host is None or b.host is None:
            aa = np.zeros(a.shape, dtype=np.float32)
            bb = np.zeros(b.shape, dtype=np.float32)
            return _fallback_buf(aa @ bb)
        return _fallback_buf(a.host @ b.host)

    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul expects 2D matrices")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")

    ctx = _ctx()
    assert ctx.device is not None
    assert ctx.command_buffer is not None
    assert ctx.p_matmul is not None

    out = empty((m, n))
    ds = ctx._alloc_descriptor_set(ctx.p_matmul)

    infos = [
        VkDescriptorBufferInfo(buffer=a.buffer, offset=0, range=a.nbytes),
        VkDescriptorBufferInfo(buffer=b.buffer, offset=0, range=b.nbytes),
        VkDescriptorBufferInfo(buffer=out.buffer, offset=0, range=out.nbytes),
    ]
    writes = [
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[0]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[1]],
        ),
        VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=ds,
            dstBinding=2,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[infos[2]],
        ),
    ]
    vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.p_matmul.pipeline)
    vkCmdBindDescriptorSets(
        ctx.command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx.p_matmul.pipeline_layout,
        0,
        1,
        [ds],
        0,
        None,
    )

    import vulkan as _vk  # type: ignore

    pc = _vk.ffi.new("uint32_t[3]", [int(m), int(n), int(k)])
    vkCmdPushConstants(
        ctx.command_buffer,
        ctx.p_matmul.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        12,
        pc,
    )
    group_count_x = (int(n) + 15) // 16
    group_count_y = (int(m) + 15) // 16
    vkCmdDispatch(ctx.command_buffer, group_count_x, group_count_y, 1)
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
    return out
