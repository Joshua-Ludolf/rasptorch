from __future__ import annotations

from rasptorch.CLI._cli_commands import ModelCommands


def _fresh_cmds() -> ModelCommands:
    cmds = ModelCommands()
    cmds.models = {}
    return cmds


def test_create_transfer_model_resnet50() -> None:
    cmds = _fresh_cmds()
    out = cmds.create_transfer_model("resnet50", 5, freeze_backbone=True, fine_tune=False)
    assert "error" not in out
    mid = out["model_id"]
    md = cmds.models[mid]
    assert md.get("type") == "Transfer"
    cfg = md.get("config") or {}
    assert cfg.get("backbone") == "resnet50"
    assert cfg.get("num_classes") == 5
    assert int(cfg.get("feature_size")) == 2048


def test_create_transfer_model_invalid_backbone() -> None:
    cmds = _fresh_cmds()
    out = cmds.create_transfer_model("unknown_backbone", 3)
    assert "error" in out


def test_train_transfer_model_runs() -> None:
    cmds = _fresh_cmds()
    made = cmds.create_transfer_model("mobilenet_v2", 4)
    assert "error" not in made
    mid = made["model_id"]
    res = cmds.train_model(mid, epochs=2, learning_rate=0.001, batch_size=4, device="cpu", optimizer_type="Adam")
    assert "error" not in res
    hist = res.get("training_history")
    assert isinstance(hist, list)
    assert len(hist) == 2
