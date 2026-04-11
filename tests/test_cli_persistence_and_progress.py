from __future__ import annotations

from pathlib import Path

from rasptorch.CLI._cli_commands import ModelCommands


def _fresh_cmds() -> ModelCommands:
    cmds = ModelCommands()
    cmds.models = {}
    return cmds


def test_save_load_pickle_roundtrip_preserves_preprocessing(tmp_path: Path) -> None:
    cmds = _fresh_cmds()
    made = cmds.create_mlp([10, 8, 2], activation="relu")
    assert "error" not in made
    mid = made["model_id"]

    cmds.models[mid]["preprocessing"] = {
        "normalize": "dataset",
        "resize_hw": [224, 224],
        "mean": 0.5,
        "std": 0.2,
    }

    rel_name = "roundtrip_model.pkl"
    saved = cmds.save_model(mid, rel_name)
    assert "error" not in saved

    loaded = cmds.load_model(rel_name)
    assert "error" not in loaded
    new_id = loaded["model_id"]
    assert cmds.models[new_id].get("preprocessing", {}).get("normalize") == "dataset"
    assert cmds.models[new_id].get("preprocessing", {}).get("resize_hw") == [224, 224]


def test_train_model_progress_callback_emits_epoch_events() -> None:
    cmds = _fresh_cmds()
    made = cmds.create_mlp([10, 8, 2], activation="relu")
    assert "error" not in made
    mid = made["model_id"]

    events = []

    def cb(ev):
        events.append(dict(ev))

    out = cmds.train_model(
        mid,
        epochs=3,
        learning_rate=0.001,
        batch_size=4,
        device="cpu",
        optimizer_type="Adam",
        progress_callback=cb,
    )
    assert "error" not in out

    epoch_events = [e for e in events if e.get("event") == "epoch"]
    assert len(epoch_events) == 3
    assert any(e.get("event") == "done" for e in events)
    assert out.get("elapsed_seconds", 0.0) >= 0.0
