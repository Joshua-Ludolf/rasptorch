"""Compatibility shim for CLI commands.

Some callers (including the Streamlit UI tests) import `rasptorch._cli_commands`.
The canonical implementation lives in `rasptorch.CLI._cli_commands`.
"""

from __future__ import annotations

from .CLI._cli_commands import *  # noqa: F403,F401
