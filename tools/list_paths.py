import re
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


PATTERNS = [
    r"[A-Za-z]:\\[^\"'\n]+",          # C:\...
    r"[A-Za-z]:/[^\"'\n]+",           # C:/...
    r"/user/[^\"'\n]+",               # /user/...
    r"/home/[^\"'\n]+",               # /home/...
    r"/mnt/[^\"'\n]+",                # /mnt/...
    r"OneDrive[^\"'\n]+",             # OneDrive...
]

regex = re.compile("|".join(f"({p})" for p in PATTERNS))

rows = []
for py in SCRIPTS_DIR.rglob("*.py"):
    text = py.read_text(encoding="utf-8", errors="ignore")
    for m in regex.finditer(text):
        hit = m.group(0)
        hit_short = hit if len(hit) <= 180 else hit[:180] + "..."
        rows.append([str(py.relative_to(ROOT)), hit_short])

out_csv = ROOT / "outputs" / "path_audit.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "path_snippet"])
    w.writerows(rows)

print(f"[OK] Found {len(rows)} path hits")
print(f"[OK] Saved: {out_csv}")
