# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Patterns (simple & robust)
# -----------------------------
# We deliberately keep it regex-based (not AST) because you often use f-strings and Path objects.

IO_PATTERNS = [
    # Reads
    ("read", r"\bpd\.read_csv\(\s*([^\)]+?)\s*\)"),
    ("read", r"\bgpd\.read_file\(\s*([^\)]+?)\s*\)"),
    ("read", r"\bnp\.load\(\s*([^\)]+?)\s*\)"),
    ("read", r"\bjoblib\.load\(\s*([^\)]+?)\s*\)"),
    ("read", r"\bopen\(\s*([^,]+?)\s*,"),
    # Writes
    ("write", r"\bto_csv\(\s*([^\)]+?)\s*\)"),
    ("write", r"\bnp\.savez?\(\s*([^\)]+?)\s*\)"),
    ("write", r"\bjoblib\.dump\(\s*[^,]+,\s*([^\)]+?)\s*\)"),
    ("write", r"\bplt\.savefig\(\s*([^\)]+?)\s*\)"),
    ("write", r"\bfig\.savefig\(\s*([^\)]+?)\s*\)"),
    ("write", r"\bjson\.dump\(\s*[^,]+,\s*([^\)]+?)\s*\)"),
]

# Find likely absolute paths inside strings (even if not used in I/O call)
ABSOLUTE_PATH_REGEXES = [
    re.compile(r"[A-Za-z]:\\[^\"'\s]+"),                 # Windows drive path
    re.compile(r"\\\\[^\"'\s]+\\[^\"'\s]+"),            # UNC path \\server\share
    re.compile(r"/user/home/[^\"'\s]+"),                # HPC style
    re.compile(r"/home/[^\"'\s]+"),                     # Linux home
    re.compile(r"/mnt/[^\"'\s]+"),                      # Linux mounts
    re.compile(r"OneDrive\s*-\s*University of Bristol", re.IGNORECASE),
]


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class IORecord:
    script: str
    op: str               # read / write
    raw_target: str       # raw extracted argument text
    lineno: int
    hint: str             # which pattern matched

    # resolved fields (best-effort)
    resolved_type: str = "unknown"   # relative / absolute / fstring / expr / unknown
    resolved_path: str = ""          # best-effort normalized path string
    exists: Optional[bool] = None    # None if cannot check


@dataclass
class RepoChecks:
    root: str
    must_have: Dict[str, bool]
    io_total: int
    io_missing_known_paths: int
    abs_path_hits: int
    scripts_scanned: int


# -----------------------------
# Helpers
# -----------------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_py_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts])


def strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith(("'", '"')) and s.endswith(("'", '"'))) and len(s) >= 2:
        return s[1:-1]
    return s


def classify_and_resolve_target(raw: str, script_dir: Path, repo_root: Path) -> Tuple[str, str, Optional[Path]]:
    """
    Try to resolve a raw argument like:
      r"data/file.csv" or "data/file.csv"
      DATA_DIR / "x.csv"
      os.path.join(...)

    We do NOT execute code. Heuristics only:
    - if it contains '{' or 'f"' -> fstring
    - if it contains Path ops or join -> expr
    - if it looks like a quoted string -> maybe resolve relative/absolute
    """
    r = raw.strip()

    # Cut trailing parameters if present: e.g. to_csv(PATH, index=False)
    # We only want the first argument expression.
    # This regex tries to split by comma not inside quotes/brackets (simple heuristic).
    # If it fails, keep the whole.
    parts = re.split(r",(?![^\(\[\{]*[\)\]\}])", r, maxsplit=1)
    first = parts[0].strip()

    # Detect fstring-ish
    if "f\"" in first or "f'" in first or "{" in first and "}" in first:
        return ("fstring", first, None)

    # If it is a quoted string literal (possibly with r prefix)
    m = re.match(r"^[rubfRUBF]*(['\"])(.*)\1$", first)
    if m:
        s = m.group(2)
        # Normalize backslashes for checking
        s_norm = s.replace("\\\\", "\\")
        # Absolute Windows drive?
        if re.match(r"^[A-Za-z]:\\", s_norm) or re.match(r"^[A-Za-z]:/", s_norm):
            return ("absolute", s_norm, None)
        # Absolute POSIX?
        if s_norm.startswith("/"):
            return ("absolute", s_norm, None)

        # Relative path: resolve against script_dir then repo_root
        # Prefer repo_root if it looks repo-relative (starts with data/, outputs/, scripts/, etc.)
        as_path = Path(s_norm)
        if not as_path.is_absolute():
            # if it already begins with common repo folders, interpret as repo-relative
            common_prefixes = ("data/", "data\\", "outputs/", "outputs\\", "scripts/", "scripts\\", "tools/", "tools\\")
            if s_norm.startswith(common_prefixes):
                cand = (repo_root / as_path).resolve()
            else:
                cand = (script_dir / as_path).resolve()
            return ("relative", s_norm, cand)

    # Expression with Path / join / variables
    if any(tok in first for tok in ("/", "Path(", "os.path.join", ".join(", "DATA_DIR", "OUT_DIR", "OUTPUT", "ROOT")):
        return ("expr", first, None)

    return ("unknown", first, None)


def find_absolute_paths_in_text(text: str) -> List[str]:
    hits: List[str] = []
    for rgx in ABSOLUTE_PATH_REGEXES:
        hits.extend(rgx.findall(text))
    # de-dup while keeping order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def ensure_outdir(root: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # helpful git placeholder
    gitkeep = outdir / ".gitkeep"
    if not gitkeep.exists():
        try:
            gitkeep.write_text("", encoding="utf-8")
        except Exception:
            pass


def check_must_have_files(root: Path) -> Dict[str, bool]:
    candidates = {
        "README.md": (root / "README.md").exists(),
        "LICENSE": (root / "LICENSE").exists() or (root / "LICENSE.txt").exists() or (root / "LICENSE.md").exists(),
        "requirements.txt_or_environment.yml": (root / "requirements.txt").exists() or (root / "environment.yml").exists() or (root / "environment.yaml").exists(),
        "CITATION.cff": (root / "CITATION.cff").exists(),
        ".gitignore": (root / ".gitignore").exists(),
    }
    return candidates


# -----------------------------
# Main scan
# -----------------------------
def scan_repo(root: Path, outdir: Path) -> None:
    scripts = iter_py_files(root)
    io_records: List[IORecord] = []
    abs_hits_rows: List[Dict[str, str]] = []

    # I/O scan
    for sp in scripts:
        rel_script = str(sp.relative_to(root)).replace("\\", "/")
        text = read_text(sp)
        lines = text.splitlines()
        script_dir = sp.parent

        # Absolute path scan (anywhere in file)
        abs_hits = find_absolute_paths_in_text(text)
        for h in abs_hits:
            abs_hits_rows.append({"script": rel_script, "absolute_path_hit": h})

        # IO pattern scan, line by line for lineno
        for lineno, line in enumerate(lines, start=1):
            for op, pat in IO_PATTERNS:
                m = re.search(pat, line)
                if not m:
                    continue
                raw_target = m.group(1).strip()
                rtype, rpath_str, cand = classify_and_resolve_target(raw_target, script_dir, root)
                exists: Optional[bool] = None
                resolved_path = rpath_str

                if rtype == "relative" and cand is not None:
                    exists = cand.exists()
                    # store as repo-relative when possible
                    try:
                        resolved_path = str(cand.relative_to(root)).replace("\\", "/")
                    except Exception:
                        resolved_path = str(cand)

                io_records.append(
                    IORecord(
                        script=rel_script,
                        op=op,
                        raw_target=raw_target,
                        lineno=lineno,
                        hint=pat,
                        resolved_type=rtype,
                        resolved_path=resolved_path,
                        exists=exists,
                    )
                )

    # Summaries
    must_have = check_must_have_files(root)
    missing_known = [r for r in io_records if r.exists is False]  # only resolvable relative paths
    checks = RepoChecks(
        root=str(root),
        must_have=must_have,
        io_total=len(io_records),
        io_missing_known_paths=len(missing_known),
        abs_path_hits=len(abs_hits_rows),
        scripts_scanned=len(scripts),
    )

    # Write outputs
    ensure_outdir(root, outdir)

    # 1) io_manifest.csv
    io_csv = outdir / "io_manifest.csv"
    with io_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["script", "op", "lineno", "raw_target", "resolved_type", "resolved_path", "exists", "hint"])
        for r in io_records:
            w.writerow([r.script, r.op, r.lineno, r.raw_target, r.resolved_type, r.resolved_path, r.exists, r.hint])

    # 2) absolute_paths.csv
    abs_csv = outdir / "absolute_paths.csv"
    with abs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["script", "absolute_path_hit"])
        w.writeheader()
        for row in abs_hits_rows:
            w.writerow(row)

    # 3) missing_files.csv (only those we could resolve & check)
    missing_csv = outdir / "missing_files.csv"
    with missing_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["script", "op", "lineno", "resolved_path"])
        for r in missing_known:
            w.writerow([r.script, r.op, r.lineno, r.resolved_path])

    # 4) audit_report.json
    report_json = outdir / "audit_report.json"
    report = {
        "checks": asdict(checks),
        "missing_repo_files": [k for k, ok in must_have.items() if not ok],
        "io_manifest_csv": str(io_csv.relative_to(root)).replace("\\", "/"),
        "absolute_paths_csv": str(abs_csv.relative_to(root)).replace("\\", "/"),
        "missing_files_csv": str(missing_csv.relative_to(root)).replace("\\", "/"),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print("\n================ Repo Audit Summary ================")
    print(f"Repo root: {root}")
    print(f"Scripts scanned: {checks.scripts_scanned}")
    print(f"I/O records found: {checks.io_total}")
    print(f"Missing files (resolvable relative paths): {checks.io_missing_known_paths}")
    print(f"Absolute path hits: {checks.abs_path_hits}")

    missing_repo = [k for k, ok in must_have.items() if not ok]
    if missing_repo:
        print("\n[Repo metadata missing]")
        for k in missing_repo:
            print(f"  - {k}")
    else:
        print("\n[Repo metadata] OK (README/LICENSE/requirements/CITATION/.gitignore present)")

    if checks.io_missing_known_paths > 0:
        print("\n[Missing files referenced by code] (first 20)")
        for r in missing_known[:20]:
            print(f"  - {r.script}:{r.lineno}  {r.op}  {r.resolved_path}")
        if len(missing_known) > 20:
            print(f"  ... and {len(missing_known) - 20} more")

    if checks.abs_path_hits > 0:
        print("\n[Absolute paths found] (first 20)")
        for row in abs_hits_rows[:20]:
            print(f"  - {row['script']}: {row['absolute_path_hit']}")
        if len(abs_hits_rows) > 20:
            print(f"  ... and {len(abs_hits_rows) - 20} more")

    print("\n[Outputs written]")
    print(f"  - {io_csv}")
    print(f"  - {abs_csv}")
    print(f"  - {missing_csv}")
    print(f"  - {report_json}")
    print("====================================================\n")


def main():
    ap = argparse.ArgumentParser(description="Audit a repo for missing files / absolute paths / reproducibility readiness.")
    ap.add_argument("root", help="Repository root directory to scan (e.g., .)")
    ap.add_argument("--outdir", default=None, help="Output directory (default: <root>/outputs/audit)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repo root not found: {root}")

    outdir = Path(args.outdir).resolve() if args.outdir else (root / "outputs" / "audit")
    scan_repo(root, outdir)


if __name__ == "__main__":
    main()
