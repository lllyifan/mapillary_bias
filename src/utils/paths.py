from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def root(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
