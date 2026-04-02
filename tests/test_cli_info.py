from __future__ import annotations

from rasptorch.CLI._cli_chat import ChatREPL


def test_chat_info_uses_backend_import(capsys) -> None:
    repl = ChatREPL(device="cpu")

    repl._cmd_info()

    captured = capsys.readouterr().out
    assert "rasptorch version:" in captured
    assert "numpy version:" in captured
