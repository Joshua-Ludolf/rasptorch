from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from rasptorch.CLI._cli_chat import ChatREPL


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chat_info_reports_active_device(capsys) -> None:
    repl = ChatREPL(device="cpu")

    repl._cmd_info()

    captured = capsys.readouterr().out
    assert "rasptorch version:" in captured
    assert "numpy version:" in captured
    assert "device: gpu (Vulkan)" in captured or "device: cpu" in captured


def test_chat_info_reports_gpu_for_cuda_backend(capsys) -> None:
    repl = ChatREPL(device="cpu")

    class _Backend:
        name = "cuda"

    repl._backend_api = lambda: ({}, None, lambda: _Backend())  # type: ignore[assignment]
    repl._cmd_info()

    captured = capsys.readouterr().out
    assert "backend: cuda" in captured
    assert "device: gpu (CUDA)" in captured


def test_force_cpu_is_reversible() -> None:
    code = """
from rasptorch import vulkan_backend as vk

vk.init(strict=False)
before = vk.using_vulkan()
vk.force_cpu()
after_force = vk.using_vulkan()
vk.init(strict=False)
after_init = vk.using_vulkan()
print(f'before={before}')
print(f'after_force={after_force}')
print(f'after_init={after_init}')
print(f'reason={vk.disabled_reason()}')
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )

    lines = {line.split("=", 1)[0]: line.split("=", 1)[1].strip() for line in proc.stdout.splitlines() if "=" in line}
    assert lines["after_force"] == "False"
    assert lines["after_init"] == lines["before"]
    # On systems without Vulkan bindings/devices, a non-empty reason is expected.
    assert lines["reason"] in {"None", ""} or "Vulkan" in lines["reason"]


def test_chat_device_can_switch_back_and_forth() -> None:
    code = """
from rasptorch import vulkan_backend as vk
from rasptorch.CLI._cli_chat import ChatREPL

repl = ChatREPL(device='cpu')
vk.init(strict=False)
before = vk.using_vulkan()
repl._handle_device_command(['cpu'])
after_cpu = vk.using_vulkan()
repl._handle_device_command(['gpu'])
after_gpu = vk.using_vulkan()
print(f'before={before}')
print(f'after_cpu={after_cpu}')
print(f'after_gpu={after_gpu}')
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )

    lines = {line.split("=", 1)[0]: line.split("=", 1)[1].strip() for line in proc.stdout.splitlines() if "=" in line}
    assert lines["after_cpu"] == "False"
    assert lines["after_gpu"] == lines["before"]
