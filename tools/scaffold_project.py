#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
from pathlib import Path
from datetime import datetime

# ==========================================
# 配置 (可在此修改)
# ==========================================
AUTHOR_NAME = "Research Author" 
AFFILIATION = "University of Bristol"
# 默认取当前文件夹名称作为项目名
PROJECT_NAME = Path.cwd().name.replace("_", " ").title()

# ==========================================
# 核心功能
# ==========================================

def get_imports_from_file(path):
    """提取 Python 文件中的 import 库"""
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
    """生成简单的目录树文本"""
    lines = []
    # 忽略列表
    ignore = {".git", ".idea", "__pycache__", ".venv", "venv", ".DS_Store"}
    
    # 获取根目录下的一级内容
    items = sorted([p for p in root_path.iterdir() if p.name not in ignore and not p.name.startswith(".")])
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        lines.append(f"{prefix}{item.name}{'/' if item.is_dir() else ''}")
        
        # 如果是文件夹，简单显示其子内容（仅下一级）
        if item.is_dir():
            sub_items = sorted([p for p in item.iterdir() if p.name not in ignore and not p.name.startswith(".")])
            sub_prefix = "    " if is_last else "│   "
            for j, sub in enumerate(sub_items):
                is_sub_last = (j == len(sub_items) - 1)
                sub_pointer = "└── " if is_sub_last else "├── "
                lines.append(f"{sub_prefix}{sub_pointer}{sub.name}")
                
    return "\n".join(lines)

# ==========================================
# 文件内容生成器
# ==========================================
def get_gitignore_content(root):
    lines = [
        "# Python & System (这些必须忽略，否则会污染仓库)",
        "__pycache__/",
        "*.py[cod]",
        ".venv/",
        "venv/",
        ".DS_Store",
        "Thumbs.db",
        ".vscode/",
        ".idea/",
        "",
        "# Data & Outputs (当前设置为：允许上传数据和结果)",
        "# 如果你以后改变主意不想上传某个大文件，可以在下面添加",
        "",
        "# 依然建议忽略掉极其巨大的临时日志或缓存（如果有的话）",
        "*.log",
    ]
    
    # 注意：这里不再添加 "data/" 到忽略列表
    print("[Config] Data folder will be INCLUDED in the repository.")
    
    return "\n".join(lines)

def get_readme_content(root):
    # 1. 分析依赖
    all_imports = set()
    for py in root.rglob("*.py"):
        if py.name == "scaffold_project.py": continue
        all_imports.update(get_imports_from_file(py))
    
    std_lib = {"os", "sys", "pathlib", "datetime", "json", "math", "re", "ast", "csv", "argparse", "typing", "dataclasses", "joblib", "shutil"}
    third_party = sorted(list(all_imports - std_lib))
    
    dep_str = ", ".join(third_party) if third_party else "No third-party libraries detected."
    tree_str = generate_tree(root)

    # 使用列表拼接，避免长字符串错误
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

# ==========================================
# 主程序
# ==========================================
def main():
    root = Path.cwd()
    print(f"Scanning root: {root}")
    
    # 定义要生成的文件
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