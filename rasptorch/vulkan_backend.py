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


try:
    # python package: "vulkan" (ctypes bindings)
    from vulkan import *  # type: ignore

    _HAS_VULKAN = True
except Exception:
    _HAS_VULKAN = False

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

        self.p_add: Optional[_Pipeline] = None
        self.p_mul: Optional[_Pipeline] = None
        self.p_relu: Optional[_Pipeline] = None
        self.p_mul_add_relu: Optional[_Pipeline] = None
        self.p_matmul: Optional[_Pipeline] = None

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
        self.p_mul_add_relu = self._create_pipeline_vec3("mul_add_relu")
        self.p_matmul = self._create_pipeline_matmul("matmul")

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
_VULKAN_DISABLED_REASON: Optional[str] = None


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


def init() -> None:
    """Initialize Vulkan backend (or do nothing if unavailable)."""

    if not _HAS_VULKAN:
        return
    try:
        _ctx()
    except Exception as e:
        _disable_vulkan(str(e))


def to_gpu(data: np.ndarray) -> VulkanBuffer:
    """Upload a float32 NumPy array into a Vulkan storage buffer."""

    arr = np.array(data, dtype=np.float32, copy=False)
    if not _HAS_VULKAN:
        return VulkanBuffer(shape=arr.shape, nbytes=arr.nbytes, buffer=0, memory=0, host=arr.copy())

    try:
        ctx = _ctx()
    except Exception as e:
        _disable_vulkan(str(e))
        return VulkanBuffer(shape=arr.shape, nbytes=arr.nbytes, buffer=0, memory=0, host=arr.copy())
    buf, mem = ctx.alloc_buffer(arr.nbytes)
    ctx.write_buffer(buf, mem, arr)
    return VulkanBuffer(shape=arr.shape, nbytes=arr.nbytes, buffer=buf, memory=mem, host=None)


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
        )

    ctx = _ctx()
    buf, mem = ctx.alloc_buffer(nbytes)
    return VulkanBuffer(shape=shape, nbytes=nbytes, buffer=buf, memory=mem, host=None)


def _fallback_buf(data: np.ndarray) -> VulkanBuffer:
    # Create a fake handle-like object when Vulkan is missing.
    arr = np.array(data, dtype=np.float32, copy=False)
    return VulkanBuffer(
        shape=tuple(arr.shape),
        nbytes=int(arr.nbytes),
        buffer=VK_NULL_HANDLE,  # type: ignore[name-defined]
        memory=VK_NULL_HANDLE,  # type: ignore[name-defined]
        host=arr.copy(),
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

    vkResetCommandBuffer(ctx.command_buffer, 0)
    begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(ctx.command_buffer, begin)
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
    vkEndCommandBuffer(ctx.command_buffer)

    ctx._submit_and_wait()
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
