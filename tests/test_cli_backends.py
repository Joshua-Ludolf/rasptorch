from __future__ import annotations

from click.testing import CliRunner

from rasptorch.CLI.cli import cli


def test_backend_list_command_outputs_backends() -> None:
    runner = CliRunner()
    res = runner.invoke(cli, ["backend", "list"])
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert "active backend" in out
    assert "numpy" in out  # CPU is displayed as "numpy" in CLI
    assert "vulkan" in out


def test_backend_connect_numpy_succeeds() -> None:
    runner = CliRunner()
    res = runner.invoke(cli, ["backend", "connect", "numpy"])
    assert res.exit_code == 0
    assert "active backend: numpy" in (res.output or "").lower()


def test_global_backend_option_is_accepted() -> None:
    runner = CliRunner()
    res = runner.invoke(cli, ["--backend", "numpy", "backend", "list"])
    assert res.exit_code == 0
    assert "active backend" in (res.output or "").lower()


def test_backend_benchmark_json_outputs_results() -> None:
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "numpy",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "0",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"results"' in out
    assert '"backend": "numpy"' in out
    assert '"status": "ok"' in out


def test_backend_benchmark_handles_backend_runtime_errors(monkeypatch) -> None:
    import numpy as np
    import rasptorch.backend as backend_mod

    class _OkBackend:
        name = "cpu"

        def matmul(self, a, b):
            return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)

    class _BrokenBackend:
        name = "opencl"

        def matmul(self, a, b):
            raise ModuleNotFoundError("No module named 'mako'")

    def _fake_connect(name: str, *, strict: bool = False):
        if name == "opencl":
            return _BrokenBackend()
        return _OkBackend()

    monkeypatch.setattr(backend_mod, "connect_backend", _fake_connect)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "numpy,opencl",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "0",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"status": "success"' in out
    assert '"backend": "opencl"' in out
    assert '"status": "unavailable"' in out


def test_backend_benchmark_vulkan_uses_resident_mode(monkeypatch) -> None:
    import numpy as np
    import rasptorch.backend as backend_mod
    import rasptorch.vulkan_backend as vk_mod

    class _Backend:
        name = "vulkan"

    def _fake_connect(name: str, *, strict: bool = False):
        return _Backend()

    class _Buf:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)

    def _to_gpu(x):
        return _Buf(np.asarray(x, dtype=np.float32))

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _matmul_into(a, b, out):
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    monkeypatch.setattr(backend_mod, "connect_backend", _fake_connect)
    monkeypatch.setattr(vk_mod, "to_gpu", _to_gpu)
    monkeypatch.setattr(vk_mod, "empty", _empty)
    monkeypatch.setattr(vk_mod, "matmul_into", _matmul_into)
    monkeypatch.setattr(vk_mod, "to_cpu", lambda b: np.asarray(b.arr, dtype=np.float32))
    monkeypatch.setattr(vk_mod, "free", lambda _b: None)
    monkeypatch.setattr(vk_mod, "begin_batch", lambda: None)
    monkeypatch.setattr(vk_mod, "end_batch", lambda: None)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "vulkan",
            "--vulkan-kernel",
            "matmul",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"backend": "vulkan"' in out
    assert '"status": "ok"' in out
    assert '"mode": "resident"' in out
    assert '"submit_every": 8' in out
    assert '"kernel": "matmul"' in out


def test_backend_benchmark_vulkan_vec4_option(monkeypatch) -> None:
    import numpy as np
    import rasptorch.backend as backend_mod
    import rasptorch.vulkan_backend as vk_mod

    class _Backend:
        name = "vulkan"

    def _fake_connect(name: str, *, strict: bool = False):
        return _Backend()

    class _Buf:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)

    def _to_gpu(x):
        return _Buf(np.asarray(x, dtype=np.float32))

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _matmul_vec4_into(a, b, out):
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    monkeypatch.setattr(backend_mod, "connect_backend", _fake_connect)
    monkeypatch.setattr(vk_mod, "to_gpu", _to_gpu)
    monkeypatch.setattr(vk_mod, "empty", _empty)
    monkeypatch.setattr(vk_mod, "matmul_vec4_into", _matmul_vec4_into)
    monkeypatch.setattr(vk_mod, "to_cpu", lambda b: np.asarray(b.arr, dtype=np.float32))
    monkeypatch.setattr(vk_mod, "free", lambda _b: None)
    monkeypatch.setattr(vk_mod, "begin_batch", lambda: None)
    monkeypatch.setattr(vk_mod, "end_batch", lambda: None)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "vulkan",
            "--vulkan-kernel",
            "matmul_vec4",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"backend": "vulkan"' in out
    assert '"status": "ok"' in out
    assert '"kernel": "matmul_vec4"' in out


def test_backend_benchmark_vulkan_auto_skips_failing_variant(monkeypatch) -> None:
    import numpy as np
    import rasptorch.backend as backend_mod
    import rasptorch.vulkan_backend as vk_mod

    class _Backend:
        name = "vulkan"

    def _fake_connect(name: str, *, strict: bool = False):
        return _Backend()

    class _Buf:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)
            self.shape = self.arr.shape
            self.nbytes = self.arr.nbytes
            self.buffer = 0  # Mock Vulkan buffer handle
            self.memory = 0  # Mock Vulkan memory handle
            self.host = None  # Mock host-mapped memory
            self.base = None  # View tracking
            self.refcount = 1  # Reference counting

    def _to_gpu(x):
        return _Buf(np.asarray(x, dtype=np.float32))

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _matmul_into(a, b, out):
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _matmul_vec4_into(_a, _b, _out):
        raise RuntimeError("glslc not found; install shader compiler tools")

    def _matmul_a_bt_out(_a, _b, _out):
        raise RuntimeError("matmul_a_bt not available")
    
    def _matmul_a_bt_tiled_out(_a, _b, _out):
        raise RuntimeError("matmul_a_bt_tiled not available")

    monkeypatch.setattr(backend_mod, "connect_backend", _fake_connect)
    monkeypatch.setattr(vk_mod, "to_gpu", _to_gpu)
    monkeypatch.setattr(vk_mod, "empty", _empty)
    monkeypatch.setattr(vk_mod, "matmul_into", _matmul_into)
    monkeypatch.setattr(vk_mod, "matmul_vec4_into", _matmul_vec4_into)
    monkeypatch.setattr(vk_mod, "matmul_a_bt_out", _matmul_a_bt_out)
    monkeypatch.setattr(vk_mod, "matmul_a_bt_tiled_out", _matmul_a_bt_tiled_out)
    monkeypatch.setattr(vk_mod, "transpose2d", lambda b: _Buf(b.arr.T))
    monkeypatch.setattr(vk_mod, "to_cpu", lambda b: np.asarray(b.arr, dtype=np.float32))
    monkeypatch.setattr(vk_mod, "free", lambda _b: None)
    monkeypatch.setattr(vk_mod, "begin_batch", lambda: None)
    monkeypatch.setattr(vk_mod, "end_batch", lambda: None)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "vulkan",
            "--vulkan-kernel",
            "auto",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"status": "ok"' in out
    assert '"kernel": "matmul"' in out


def test_backend_benchmark_vulkan_a_bt_tiled_option(monkeypatch) -> None:
    import numpy as np
    import rasptorch.backend as backend_mod
    import rasptorch.vulkan_backend as vk_mod

    class _Backend:
        name = "vulkan"

    def _fake_connect(name: str, *, strict: bool = False):
        return _Backend()

    class _Buf:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)

    def _to_gpu(x):
        return _Buf(np.asarray(x, dtype=np.float32))

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _transpose2d(x):
        return _Buf(np.asarray(x.arr.T, dtype=np.float32))

    def _matmul_a_bt_tiled_out(a, b_t, out):
        out.arr = np.asarray(a.arr @ b_t.arr.T, dtype=np.float32)
        return out

    monkeypatch.setattr(backend_mod, "connect_backend", _fake_connect)
    monkeypatch.setattr(vk_mod, "to_gpu", _to_gpu)
    monkeypatch.setattr(vk_mod, "empty", _empty)
    monkeypatch.setattr(vk_mod, "transpose2d", _transpose2d)
    monkeypatch.setattr(vk_mod, "matmul_a_bt_tiled_out", _matmul_a_bt_tiled_out)
    monkeypatch.setattr(vk_mod, "to_cpu", lambda b: np.asarray(b.arr, dtype=np.float32))
    monkeypatch.setattr(vk_mod, "free", lambda _b: None)
    monkeypatch.setattr(vk_mod, "begin_batch", lambda: None)
    monkeypatch.setattr(vk_mod, "end_batch", lambda: None)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "vulkan",
            "--vulkan-kernel",
            "matmul_a_bt_tiled",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"backend": "vulkan"' in out
    assert '"status": "ok"' in out
    assert '"kernel": "matmul_a_bt_tiled"' in out


def test_backend_benchmark_vulkan_reports_non_empty_error(monkeypatch) -> None:
    import rasptorch.backend as backend_mod
    import rasptorch.vulkan_backend as vk_mod

    class _Backend:
        name = "vulkan"

    def _fake_connect(name: str, *, strict: bool = False):
        return _Backend()

    monkeypatch.setattr(backend_mod, "connect_backend", _fake_connect)
    monkeypatch.setattr(vk_mod, "to_gpu", lambda _x: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(vk_mod, "disabled_reason", lambda: "Vulkan initialization failed")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--json",
            "backend",
            "benchmark",
            "--backends",
            "vulkan",
            "--vulkan-kernel",
            "matmul",
            "--size",
            "8",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--seed",
            "1",
        ],
    )
    assert res.exit_code == 0
    out = (res.output or "").lower()
    assert '"status": "unavailable"' in out
    assert "vulkan initialization failed" in out

