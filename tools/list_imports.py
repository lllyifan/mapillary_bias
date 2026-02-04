from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = list((ROOT / "scripts").rglob("*.py"))

imports = set()
pat = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)")

for f in SCRIPTS:
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            imports.add(m.group(1))

# common stdlib to ignore
stdlib = {
    "os","sys","re","math","json","time","datetime","pathlib",
    "itertools","functools","collections","subprocess","warnings",
    "typing","logging","hashlib"
}

third_party = sorted(i for i in imports if i not in stdlib)

print("Third-party imports detected:\n")
for i in third_party:
    print(i)
