from pathlib import Path

def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"[MISSING] Required file not found: {path}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
