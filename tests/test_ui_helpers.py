from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import tempfile
import numpy as np


def _import_ui_app():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app_path = repo_root / "rasptorch" / "CLI" / "ui" / "app.py"
    spec = importlib.util.spec_from_file_location("rasptorch_ui_app", app_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mermaid_diagram_for_mlp() -> None:
    ui = _import_ui_app()

    models = {
        "abcd1234": {"type": "MLP", "config": {"layer_sizes": [10, 32, 2]}},
    }
    m = ui._model_mermaid_diagram(models, "abcd1234")
    assert m is not None
    assert "flowchart" in m
    assert "MLP" in m
    assert "10" in m and "32" in m and "2" in m


def test_mermaid_diagram_for_combined_sequential() -> None:
    ui = _import_ui_app()

    models = {
        "aaaa1111": {"type": "Linear", "config": {"input_size": 10, "hidden_sizes": [32], "output_size": 128}},
        "bbbb2222": {"type": "GRU", "config": {"input_size": 128, "hidden_size": 64, "num_layers": 1}},
        "cccc3333": {
            "type": "Combined",
            "config": {
                "combine": "sequential",
                "model_a_id": "aaaa1111",
                "model_b_id": "bbbb2222",
                "model_a_type": "Linear",
                "model_b_type": "GRU",
                "input_size": 10,
                "output_size": 64,
            },
        },
    }

    m = ui._model_mermaid_diagram(models, "cccc3333")
    assert m is not None
    assert "Combined" in m
    assert "Linear" in m and "GRU" in m
    assert "I/O" in m


def test_train_model_returns_history() -> None:
    # Train API should return per-epoch loss history for charting.
    from rasptorch.CLI._cli_commands import get_model_commands

    cmds = get_model_commands()
    res = cmds.create_mlp([10, 8, 2], activation="relu")
    assert "error" not in res
    mid = res["model_id"]

    out = cmds.train_model(mid, epochs=3, learning_rate=0.001, batch_size=4, device="cpu", optimizer_type="Adam")
    assert "error" not in out
    hist = out.get("training_history")
    assert isinstance(hist, list)
    assert len(hist) == 3
    assert all(isinstance(x, (int, float)) for x in hist)

def test_info_config_filtering_logic_removes_activation_when_activations_present() -> None:
    # Mirrors the Info-panel filtering behavior (without rendering Streamlit).
    cfg = {"activation": "relu", "activations": ["relu", "none"], "layer_sizes": [10, 32, 2]}
    if "activations" in cfg and "activation" in cfg:
        cfg.pop("activation", None)
    assert "activation" not in cfg
    assert "activations" in cfg


def test_info_config_filtering_is_recursive_for_combined_snapshots() -> None:
    ui = _import_ui_app()

    cfg = {
        "combine": "sequential",
        "model_a_snapshot": {
            "type": "MLP",
            "config": {"activation": "relu", "activations": ["relu", "none"], "layer_sizes": [10, 8, 2]},
        },
        "model_b_snapshot": {
            "type": "GRU",
            "config": {"activation": "tanh"},
        },
    }

    cleaned = ui._strip_redundant_activation(cfg)

    # Nested snapshot config should not show both keys.
    a_cfg = cleaned["model_a_snapshot"]["config"]
    assert "activations" in a_cfg
    assert "activation" not in a_cfg

    # If there is no per-layer list, keep the single activation.
    b_cfg = cleaned["model_b_snapshot"]["config"]
    assert b_cfg.get("activation") == "tanh"


def test_uploaded_name_stored_in_config_for_display() -> None:
    # UI stores a user friendly name under config['name'].
    models = {"id123456": {"type": "MLP", "config": {"name": "my_model"}}}
    cfg = dict(models["id123456"].get("config") or {})
    assert cfg.get("name") == "my_model"


def test_activation_mode_submit_logic_matches_mode() -> None:
    # This test mirrors the logic in _render_build_train_page without invoking Streamlit.
    # We reload the module to ensure globals exist.
    ui = _import_ui_app()

    activation_mode = "Per-layer (CSV)"
    activation = "tanh"
    per_layer = "relu,none"

    activations = None
    if activation_mode.startswith("Per-layer"):
        activations = [x.strip() for x in per_layer.split(",") if x.strip()] or None

    assert activations == ["relu", "none"]
    assert activation == "tanh"  # single activation stays independent


def test_csv_training_payload_parser_uses_named_columns() -> None:
    ui = _import_ui_app()

    csv_text = "f1,f2,target\n1.0,2.0,0.0\n3.0,4.0,1.0\n"
    parsed = ui._parse_csv_training_payload(csv_text, ["f1", "f2"], "target")

    assert parsed["source"] == "csv"
    assert parsed["rows"] == 2
    assert parsed["x"].shape == (2, 2)
    assert parsed["y"].shape == (2, 1)
    assert parsed["features"] == ["f1", "f2"]
    assert parsed["feature_encoding"][0]["kind"] == "numeric"
    assert parsed["target_encoding"]["kind"] == "numeric"


def test_csv_training_payload_parser_encodes_string_features_and_target() -> None:
    ui = _import_ui_app()

    csv_text = "name,color,label\nSheryl,red,cat\nMina,blue,dog\nSheryl,blue,cat\n"
    parsed = ui._parse_csv_training_payload(csv_text, ["name", "color"], "label")

    assert parsed["source"] == "csv"
    assert parsed["rows"] == 3
    assert parsed["x"].shape == (3, 4)
    assert parsed["y"].shape == (3, 2)
    assert parsed["feature_encoding"][0]["kind"] == "categorical"
    assert parsed["feature_encoding"][1]["kind"] == "categorical"
    assert parsed["target_encoding"]["kind"] == "categorical"


def test_npz_training_payload_parser_prefers_standard_keys() -> None:
    ui = _import_ui_app()

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        np.savez(path, x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), y=np.array([0.0, 1.0], dtype=np.float32))
        parsed = ui._parse_npz_training_payload(path.read_bytes())
    finally:
        path.unlink(missing_ok=True)

    assert parsed["source"] == "npz"
    assert parsed["rows"] == 2
    assert parsed["x"].shape == (2, 2)
    assert parsed["y"].shape == (2, 1)
    assert "x" in parsed["keys"]


def test_training_run_summary_includes_hyperparameters() -> None:
    ui = _import_ui_app()

    summary = ui._summarize_training_run(
        {
            "model_id": "abcd1234efgh",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
            "final_loss": 0.123456,
            "train_data_summary": "CSV: 100 rows, input_shape=(8,)",
        }
    )

    assert "abcd1234" in summary
    assert "opt=Adam" in summary
    assert "lr=0.001" in summary
    assert "batch=32" in summary
    assert "loss=0.123456" in summary


def test_ui_constants_exist_and_are_nonempty() -> None:
    ui = _import_ui_app()

    assert hasattr(ui, "HELP_TEXT")
    assert isinstance(ui.HELP_TEXT, list)
    assert len(ui.HELP_TEXT) >= 3

    assert hasattr(ui, "ACTIVATIONS")
    assert isinstance(ui.ACTIVATIONS, list)
    assert "relu" in ui.ACTIVATIONS

    assert hasattr(ui, "OPTIMIZERS")
    assert isinstance(ui.OPTIMIZERS, list)
    assert "Adam" in ui.OPTIMIZERS


def test_ui_help_matches_chat_repl_help() -> None:
    ui = _import_ui_app()
    from rasptorch.CLI._cli_chat import ChatREPL

    expected = [line.rstrip() for line in ChatREPL.get_help(ChatREPL.__new__(ChatREPL)).splitlines() if line.strip()]
    assert ui._chat_repl_help_lines() == expected


def test_infer_model_input_size_helper() -> None:
    ui = _import_ui_app()

    md = {"config": {"feature_size": 1280, "num_classes": 4}}
    out = ui._infer_model_input_size(md)
    assert out == 1280


def test_gradcam_like_overlay_fallback() -> None:
    ui = _import_ui_app()

    class _DummyModel:
        pass

    img = (np.random.RandomState(0).rand(32, 32, 3) * 255.0).astype("uint8")
    md = {"config": {"input_size": 64}}
    overlay, info = ui._gradcam_like_overlay(img, md, _DummyModel())

    assert overlay.shape == img.shape
    assert overlay.dtype == img.dtype
    assert info.get("xai_method") == "gradcam_style_approx"


def test_display_name_with_type_prefers_friendly_name() -> None:
    ui = _import_ui_app()

    models = {
        "id1234567890": {"type": "MLP", "config": {"name": "NiceName"}},
        "id_no_name": {"type": "CNN", "config": {}},
    }

    # Mirror the logic used in the Build & Train page format_func.
    def _display_name_with_type(mid: str) -> str:
        md = models.get(mid) or {}
        mtype = str(md.get("type", "Unknown"))
        cfg = md.get("config") or {}
        friendly = cfg.get("name")
        base = str(friendly) if friendly else mid[:8]
        return f"{base} ({mtype})"

    assert _display_name_with_type("id1234567890") == "NiceName (MLP)"
    assert _display_name_with_type("id_no_name") == "id_no_na (CNN)"
