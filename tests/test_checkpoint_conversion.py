from __future__ import annotations

import sys

import numpy as np

from rasptorch.checkpoint import convert_legacy_torch_checkpoint, load_checkpoint


class _FakeTorchModule:
    @staticmethod
    def load(path, map_location="cpu", weights_only=True):  # noqa: ANN001
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "arch": "Sequential",
            "state_dict": {
                "0.weight": np.ones((2, 3), dtype=np.float32),
                "0.bias": np.zeros((2,), dtype=np.float32),
            },
        }


class _FakeTorchStateDictOnly:
    @staticmethod
    def load(path, map_location="cpu", weights_only=True):  # noqa: ANN001
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "linear.weight": np.full((4, 5), 2.0, dtype=np.float32),
            "linear.bias": np.full((4,), -1.0, dtype=np.float32),
        }


def test_convert_legacy_torch_checkpoint_payload_dict(tmp_path, monkeypatch) -> None:
    src = tmp_path / "legacy.pth"
    dst = tmp_path / "converted.pth"
    src.write_bytes(b"legacy")

    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule)

    out = convert_legacy_torch_checkpoint(str(src), str(dst))
    assert out["status"] == "success"
    assert out["format"] == "rasptorch-npz"
    assert out["num_tensors"] == 2

    payload = load_checkpoint(str(dst))
    assert payload.get("arch") == "Sequential"
    sd = payload.get("state_dict")
    assert isinstance(sd, dict)
    np.testing.assert_allclose(sd["0.weight"], np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(sd["0.bias"], np.zeros((2,), dtype=np.float32))


def test_convert_legacy_torch_checkpoint_state_dict_only(tmp_path, monkeypatch) -> None:
    src = tmp_path / "legacy.pt"
    dst = tmp_path / "converted.pt"
    src.write_bytes(b"legacy")

    monkeypatch.setitem(sys.modules, "torch", _FakeTorchStateDictOnly)

    out = convert_legacy_torch_checkpoint(str(src), str(dst))
    assert out["status"] == "success"
    assert out["num_tensors"] == 2

    payload = load_checkpoint(str(dst))
    sd = payload.get("state_dict")
    assert isinstance(sd, dict)
    np.testing.assert_allclose(sd["linear.weight"], np.full((4, 5), 2.0, dtype=np.float32))
    np.testing.assert_allclose(sd["linear.bias"], np.full((4,), -1.0, dtype=np.float32))
