"""Build QuantConnect project directory for upload.

Creates a self-contained project folder with all required modules
copied from the main codebase. Upload via LEAN CLI:

    python3 quantconnect/deploy.py
    cd quantconnect/ManciniMES
    lean cloud push
    lean cloud backtest "ManciniMES"
    lean cloud live "ManciniMES" --brokerage "Paper Trading"
"""

import os
import shutil
from pathlib import Path

# Project root (parent of quantconnect/)
ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT / "quantconnect" / "ManciniMES"

# Modules to copy into the QC project (relative to ROOT)
MODULES = {
    # Config
    "config/__init__.py": "config/__init__.py",
    "config/settings.py": "config/settings.py",
    "config/levels.py": "config/levels.py",
    # Core
    "core/__init__.py": "core/__init__.py",
    "core/signals.py": "core/signals.py",
    "core/patterns.py": "core/patterns.py",
    "core/price_levels.py": "core/price_levels.py",
    "core/elevator_down.py": "core/elevator_down.py",
    "core/indicators.py": "core/indicators.py",
    # Strategy
    "strategy/__init__.py": "strategy/__init__.py",
    "strategy/mancini_long.py": "strategy/mancini_long.py",
    "strategy/entry_manager.py": "strategy/entry_manager.py",
    "strategy/exit_manager.py": "strategy/exit_manager.py",
    "strategy/position_manager.py": "strategy/position_manager.py",
    "strategy/risk_manager.py": "strategy/risk_manager.py",
    # QC adapter (the main algorithm file)
    "quantconnect/main.py": "main.py",
}


def build_project():
    """Create the QC project directory with all required files."""
    print(f"Building QuantConnect project at: {PROJECT_DIR}")

    # Clean and create
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.mkdir(parents=True)

    # Create subdirectories
    for subdir in ("config", "core", "strategy"):
        (PROJECT_DIR / subdir).mkdir(exist_ok=True)

    # Copy modules
    copied = 0
    for src_rel, dst_rel in MODULES.items():
        src = ROOT / src_rel
        dst = PROJECT_DIR / dst_rel

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
            print(f"  {src_rel} -> {dst_rel}")
        else:
            # Create empty __init__.py if missing
            if src_rel.endswith("__init__.py"):
                dst.write_text("")
                copied += 1
                print(f"  (created empty) {dst_rel}")
            else:
                print(f"  WARNING: {src_rel} not found!")

    # Create config.json for LEAN CLI (algorithm entry point)
    config = {
        "algorithm-language": "Python",
        "parameters": {},
        "description": "Mancini MES long-only day trading strategy",
    }
    import json
    (PROJECT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  config.json (LEAN CLI config)")

    print(f"\nDone! {copied} files copied to {PROJECT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Create a free account at https://www.quantconnect.com")
    print(f"  2. pip install lean")
    print(f"  3. lean login")
    print(f"  4. cd {PROJECT_DIR}")
    print(f"  5. lean cloud push")
    print(f"  6. lean cloud backtest 'ManciniMES'")
    print(f"  7. lean cloud live 'ManciniMES' --brokerage 'Paper Trading'")


if __name__ == "__main__":
    build_project()
