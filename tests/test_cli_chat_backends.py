from __future__ import annotations

from rasptorch.CLI._cli_chat import ChatREPL


def test_chat_backend_use_cpu_updates_context() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_backend_command(["use", "numpy"])
    assert repl.context.get("backend") == "numpy"


def test_chat_backend_list_does_not_error() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_backend_command(["list"])


def test_chat_backend_shorthand_cpu_updates_context() -> None:
    repl = ChatREPL(device="cpu")
    repl._handle_backend_command(["numpy"])
    assert repl.context.get("backend") == "numpy"

