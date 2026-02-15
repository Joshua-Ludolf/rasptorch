from __future__ import annotations

import pickle

import numpy as np

from rasptorch import vulkan_backend as vk
from rasptorch.gpu_training import GpuMLP


def test_gpu_training_model_save_payload(tmp_path) -> None:
    """Covers the README-documented save format.

    If torch is installed, the file should be torch.load()-able.
    Otherwise, it should be a pickle containing the same keys.
    """

    path = tmp_path / "model.pth"

    m = GpuMLP(in_features=2, hidden=4, out_features=3, seed=0)
    try:
        m.save(str(path))
    finally:
        m.close()

    assert path.exists()

    try:
        import torch  # type: ignore

        payload = torch.load(str(path))
    except ModuleNotFoundError:
        payload = pickle.load(open(path, "rb"))

    assert isinstance(payload, dict)
    assert "arch" in payload
    assert "state_dict" in payload


def test_torch_bridge_linear_readme_option() -> None:
    """README option: torch_bridge.

    This test is designed to be non-skipping:
    - If torch is installed, it validates correctness via the CPU/fallback bridge path.
    - If torch is not installed, it asserts we raise a clear error message.
    - If Vulkan is unavailable, the bridge still works in fallback mode, but strict `device='gpu'`
      conversion should raise.
    """

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        # Importing is fine; using it should fail with a clear message.
        from rasptorch import torch_bridge

        try:
            torch_bridge.convert_torch_model(object(), device="gpu")
        except RuntimeError as e:
            msg = str(e).lower()
            assert "pytorch" in msg or "torch" in msg
        else:
            raise AssertionError("expected RuntimeError when torch is not installed")
        return

    from rasptorch.torch_bridge import convert_torch_model

    torch.manual_seed(0)

    mod = torch.nn.Linear(6, 4, bias=True)
    x = torch.randn(3, 6, dtype=torch.float32)

    ref = mod(x)

    # Use device='cpu' to avoid requiring strict Vulkan init on machines without Vulkan.
    bridged = convert_torch_model(mod, device="cpu")
    out = bridged(x)
    np.testing.assert_allclose(out.detach().cpu().numpy(), ref.detach().cpu().numpy(), rtol=2e-2, atol=2e-2)

    # README also mentions a strict Vulkan-backed path; assert it either works or fails clearly.
    try:
        bridged_gpu = convert_torch_model(mod, device="gpu")
        out2 = bridged_gpu(x)
        np.testing.assert_allclose(out2.detach().cpu().numpy(), ref.detach().cpu().numpy(), rtol=2e-2, atol=2e-2)
    except Exception as e:
        # On non-Vulkan systems, strict init can fail; ensure it's an informative error.
        msg = (str(e) or "").lower()
        reason = (vk.disabled_reason() or "").lower()
        assert ("vulkan" in msg) or ("glslc" in msg) or (reason and reason in msg)
