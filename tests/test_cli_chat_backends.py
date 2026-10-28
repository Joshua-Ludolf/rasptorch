from __future__ import annotations

from rasptorch.CLI._cli_chat import ChatREPL


def test_chat_backend_use_cpu_updates_context() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_backend_command(["use", "numpy"])
    assert repl.context.get("backend") == "numpy"


def test_chat_backend_list_does_not_error() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_backend_command(["list"])


from rasptorch.CLI._cli_commands import get_model_commands


def test_chat_backend_shorthand_cpu_updates_context() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_backend_command(["numpy"])
    assert repl.context.get("backend") == "numpy"


def test_chat_model_cv_creation() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_model_command(["cv", "3", "32,64", "3,3", "10", "max", "32", "act=relu"])
    assert repl.context.get("current_model") is not None
    cmds = get_model_commands()
    mid = repl.context["current_model"]
    md = cmds.models[mid]
    assert md["type"] == "CV"
    assert md["config"]["in_channels"] == 3
    assert md["config"]["out_channels"] == [32, 64]
    assert md["config"]["num_classes"] == 10
    assert md["config"]["pool"] == "max"
    assert md["config"]["input_hw"] == 32


def test_chat_model_jev_creation() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_model_command(["jev", "10", "32,16", "2", "act=relu"])
    assert repl.context.get("current_model") is not None
    cmds = get_model_commands()
    mid = repl.context["current_model"]
    md = cmds.models[mid]
    assert md["type"] == "JEV"
    assert md["config"]["input_size"] == 10
    assert md["config"]["hidden_sizes"] == [32, 16]
    assert md["config"]["num_decisions"] == 2

