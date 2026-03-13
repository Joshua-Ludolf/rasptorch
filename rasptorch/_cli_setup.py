"""Post-installation setup script for rasptorch CLI.

Generates shell alias for easy CLI access from any directory.
"""

import os
import sys


def setup_shell_alias():
    """Generate shell alias for rasptorch CLI."""
    
    rasptorch_install_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    shell_configs = [
        os.path.expanduser("~/.bashrc"),
        os.path.expanduser("~/.zshrc"),
        os.path.expanduser("~/.bash_profile"),
    ]
    
    alias_line = f'alias rasptorch="cd {rasptorch_install_path} && uv run rasptorch"\n'
    
    for config_file in shell_configs:
        if not os.path.exists(config_file):
            continue
            
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Skip if alias already exists
        if 'alias rasptorch=' in content:
            continue
        
        # Add alias
        with open(config_file, 'a') as f:
            f.write(alias_line)
    
    print("✓ rasptorch CLI alias installed!")
    print(f"  Run this to activate it:")
    print(f"  source ~/.bashrc  # or ~/.zshrc")
    print(f"")
    print(f"  Then use: rasptorch --help")


if __name__ == "__main__":
    try:
        setup_shell_alias()
    except Exception as e:
        print(f"Warning: Could not set up shell alias: {e}", file=sys.stderr)
        print(f"You can manually add to ~/.bashrc:", file=sys.stderr)
        print(f'  alias rasptorch="cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))} && uv run rasptorch"', file=sys.stderr)
