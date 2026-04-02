from __future__ import annotations

import importlib


def test_combine_rejects_incompatible_feature_sizes(tmp_path) -> None:
    """CLI combine should block clearly incompatible pairs.

    Example: CNN outputs out_channels[-1] features, GRU expects input_size.
    """

    import rasptorch._cli_commands as cc

    # Isolate session state.
    cc._SESSION_DIR = str(tmp_path)
    importlib.reload(cc)
    cc._SESSION_DIR = str(tmp_path)

    cmds = cc.ModelCommands()

    cnn = cmds.create_cnn(in_channels=3, out_channels=[64, 64], kernel_sizes=None)
    assert "error" not in cnn
    gru = cmds.create_gru(input_size=128, hidden_size=256, num_layers=1)
    assert "error" not in gru

    res = cmds.combine_models(cnn["model_id"], gru["model_id"])
    assert "error" in res
    msg = str(res["error"]).lower()
    assert "cannot combine" in msg
    assert "incompatible" in msg
    assert "outputs" in msg
    assert "expects" in msg


def test_combine_allows_matching_feature_sizes(tmp_path) -> None:
    import rasptorch._cli_commands as cc

    cc._SESSION_DIR = str(tmp_path)
    importlib.reload(cc)
    cc._SESSION_DIR = str(tmp_path)

    cmds = cc.ModelCommands()

    a = cmds.create_linear_model(10, [32], 128, activation="relu")
    assert "error" not in a
    b = cmds.create_gru(input_size=128, hidden_size=64, num_layers=1)
    assert "error" not in b

    res = cmds.combine_models(a["model_id"], b["model_id"])
    assert res.get("status") == "success"
    cfg = cmds.models[res["model_id"]].get("config") or {}
    assert cfg.get("input_size") == 10
    assert cfg.get("output_size") == 64


def test_combine_snapshots_do_not_persist_redundant_activation(tmp_path) -> None:
    import rasptorch._cli_commands as cc

    cc._SESSION_DIR = str(tmp_path)
    importlib.reload(cc)
    cc._SESSION_DIR = str(tmp_path)

    cmds = cc.ModelCommands()

    # Per-layer activations should not be persisted alongside the legacy single `activation`.
    a = cmds.create_linear_model(10, [32], 128, activation="tanh", activations=["relu"])
    assert "error" not in a
    b = cmds.create_gru(input_size=128, hidden_size=64, num_layers=1)
    assert "error" not in b

    combined = cmds.combine_models(a["model_id"], b["model_id"])
    assert combined.get("status") == "success"

    combined_cfg = (cmds.models[combined["model_id"]].get("config") or {})
    snap = combined_cfg.get("model_a_snapshot") or {}
    snap_cfg = (snap.get("config") or {})
    assert "activations" in snap_cfg
    assert "activation" not in snap_cfg
