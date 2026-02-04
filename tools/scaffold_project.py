#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
from pathlib import Path
from datetime import datetime


AUTHOR_NAME = "Research Author"
AFFILIATION = "University of Bristol"
PROJECT_NAME = Path.cwd().name.replace("_", " ").title()


def get_imports_from_file(path):
    imports = set()
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        root = ast.parse(content)
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception:
        pass
    return imports


def generate_tree(root_path):
    lines = []
    ignore = {".git", ".idea", "__pycache__", ".venv", "venv", ".DS_Store"}

    items = sorted([p for p in root_path.iterdir() if p.name not in ignore and not p.name.startswith(".")])

    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        lines.append(f"{prefix}{item.name}{'/' if item.is_dir() else ''}")

        if item.is_dir():
            sub_items = sorted([p for p in item.iterdir() if p.name not in ignore and not p.name.startswith(".")])
            sub_prefix = "    " if is_last else "│   "
            for j, sub in enumerate(sub_items):
                is_sub_last = (j == len(sub_items) - 1)
                sub_pointer = "└── " if is_sub_last else "├── "
                lines.append(f"{sub_prefix}{sub_pointer}{sub.name}")

    return "\n".join(lines)


def get_gitignore_content(root):
    lines = [
        "__pycache__/",
        "*.py[cod]",
        ".venv/",
        "venv/",
        ".DS_Store",
        "Thumbs.db",
        ".vscode/",
        ".idea/",
        "",
        "*.log",
    ]

    print("[Config] Data folder will be INCLUDED in the repository.")

    return "\n".join(lines)


def get_readme_content(root):
    all_imports = set()
    for py in root.rglob("*.py"):
        if py.name == "scaffold_project.py":
            continue
        all_imports.update(get_imports_from_file(py))

    std_lib = {"os", "sys", "pathlib", "datetime", "json", "math", "re", "ast", "csv", "argparse", "typing", "dataclasses", "joblib", "shutil"}
    third_party = sorted(list(all_imports - std_lib))

    dep_str = ", ".join(third_party) if third_party else "No third-party libraries detected."
    tree_str = generate_tree(root)

    content_lines = [
        f"# {PROJECT_NAME}",
        "",
        "## Project Overview",
        "Research code for analyzing Mapillary data bias.",
        "",
        "## Directory Structure",
        "```text",
        f"{root.name}/",
        tree_str,
        "```",
        "",
        "## Requirements",
        f"Detected dependencies: {dep_str}",
        "",
        "## Setup & Usage",
        "1. Place input data in the `data/` directory.",
        "2. Run the main analysis scripts located in `scripts/`.",
        "3. Check `outputs/` for generated results.",
        "",
        "## License",
        "MIT License. See [LICENSE](LICENSE) for details."
    ]
    return "\n".join(content_lines)


def get_license_content():
    year = datetime.now().year
    return f"""MIT License

Copyright (c) {year} {AUTHOR_NAME} ({AFFILIATION})

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def get_citation_content():
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Author"
    given-names: "Research"
    affiliation: "{AFFILIATION}"
title: "{PROJECT_NAME}"
date-released: {today}
"""


def main():
    root = Path.cwd()
    print(f"Scanning root: {root}")

    tasks = {
        ".gitignore": get_gitignore_content(root),
        "README.md": get_readme_content(root),
        "LICENSE": get_license_content(),
        "CITATION.cff": get_citation_content()
    }

    for filename, content in tasks.items():
        path = root / filename
        if path.exists():
            print(f"[SKIP] {filename} exists. (Delete it manually to regenerate)")
        else:
            try:
                path.write_text(content, encoding="utf-8")
                print(f"[CREATE] {filename}")
            except Exception as e:
                print(f"[ERROR] Could not create {filename}: {e}")


if __name__ == "__main__":
    main()
