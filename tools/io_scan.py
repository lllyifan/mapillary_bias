#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import ast
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any


FILE_SUFFIX_RE = re.compile(
    r".*\.(csv|tsv|parquet|feather|pkl|pickle|joblib|json|geojson|shp|gpkg|tif|tiff|png|jpg|jpeg|pdf|txt|md|npz|npy|xlsx)$",
    re.I,
)

def looks_like_file(s: str) -> bool:
    s = s.strip().strip("'\"")
    if FILE_SUFFIX_RE.match(s):
        return True
    lowered = s.lower()
    return any(ext in lowered for ext in [
        ".csv", ".parquet", ".png", ".pdf", ".pkl", ".json", ".shp", ".tif", ".xlsx", ".npz", ".npy"
    ])


@dataclass
class IORecord:
    script: str
    op: str
    target: str
    lineno: int
    hint: str


@dataclass
class ModuleEnv:
    consts: Dict[str, str]
    imports: Dict[str, str]
    from_imports: Dict[str, Tuple[str, str]]


def module_name_from_path(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def load_all_modules(root: Path) -> Dict[str, Path]:
    mods = {}
    for p in root.rglob("*.py"):
        if any(part.startswith(".") for part in p.parts):
            continue
        if ".venv" in p.parts or "site-packages" in p.parts or "__pycache__" in p.parts:
            continue
        mods[module_name_from_path(root, p)] = p
    return mods


def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def is_path_ctor_call(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"Path", "PurePath"}
    return False


def join_expr(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    return f"{a}/{b}"


def expr_to_str(node: ast.AST, env: ModuleEnv, resolver: "Resolver") -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s

    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.Str):
                parts.append(v.s)
            else:
                parts.append("{...}")
        return "".join(parts)

    if isinstance(node, ast.Name):
        name = node.id
        if name in env.consts:
            return env.consts[name]
        if name in env.from_imports:
            mod, orig = env.from_imports[name]
            resolved = resolver.resolve_imported_constant(mod, orig)
            if resolved:
                return resolved
        return None

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            base = node.value.id
            attr = node.attr
            if base in env.imports:
                mod = env.imports[base]
                resolved = resolver.resolve_imported_constant(mod, attr)
                if resolved:
                    return resolved
                return f"{mod}.{attr}"
        return None

    if is_path_ctor_call(node):
        if node.args:
            inner = expr_to_str(node.args[0], env, resolver)
            return inner
        return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l = expr_to_str(node.left, env, resolver)
        r = expr_to_str(node.right, env, resolver)
        if l is not None and r is not None:
            return l + r
        if r and looks_like_file(r):
            return r
        if l and looks_like_file(l):
            return l
        return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        l = expr_to_str(node.left, env, resolver)
        r = expr_to_str(node.right, env, resolver)
        if l is not None and r is not None:
            return join_expr(l, r)
        if r and looks_like_file(r):
            return r
        return None

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "join" and isinstance(node.func.value, ast.Attribute):
            if node.func.value.attr == "path" and isinstance(node.func.value.value, ast.Name):
                if node.func.value.value.id == "os":
                    parts = []
                    for a in node.args:
                        s = expr_to_str(a, env, resolver)
                        if s:
                            parts.append(s)
                        else:
                            parts.append("{...}")
                    if parts:
                        out = parts[0]
                        for p in parts[1:]:
                            out = join_expr(out, p)
                        return out

    return None


class ConstCollector(ast.NodeVisitor):
    def __init__(self, env: ModuleEnv, resolver: "Resolver"):
        self.env = env
        self.resolver = resolver

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name.split(".")[-1]
            self.env.imports[asname] = name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not node.module:
            return
        mod = node.module
        for alias in node.names:
            orig = alias.name
            asname = alias.asname or orig
            self.env.from_imports[asname] = (mod, orig)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            return
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            return
        name = tgt.id
        val = expr_to_str(node.value, self.env, self.resolver)
        if val is not None:
            self.env.consts[name] = val

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
            val = expr_to_str(node.value, self.env, self.resolver)
            if val is not None:
                self.env.consts[name] = val


class Resolver:
    def __init__(self, root: Path, modules: Dict[str, Path]):
        self.root = root
        self.modules = modules
        self.env_cache: Dict[str, ModuleEnv] = {}
        self.ast_cache: Dict[str, ast.AST] = {}

    def get_env(self, mod: str) -> ModuleEnv:
        if mod in self.env_cache:
            return self.env_cache[mod]
        env = ModuleEnv(consts={}, imports={}, from_imports={})
        self.env_cache[mod] = env
        p = self.modules.get(mod)
        if not p:
            return env
        try:
            tree = ast.parse(safe_read_text(p))
        except SyntaxError:
            return env
        self.ast_cache[mod] = tree
        ConstCollector(env, self).visit(tree)
        return env

    def resolve_imported_constant(self, mod: str, name: str) -> Optional[str]:
        env = self.get_env(mod)
        if name in env.consts:
            return env.consts[name]
        if name in env.from_imports:
            mod2, orig2 = env.from_imports[name]
            return self.resolve_imported_constant(mod2, orig2)
        return None


READ_ATTRS = {
    "read_csv", "read_table", "read_parquet", "read_feather", "read_pickle",
    "read_json", "read_excel", "read_file", "load", "open",
}
WRITE_ATTRS = {
    "to_csv", "to_parquet", "to_feather", "to_pickle", "to_json", "to_excel",
    "to_file", "save", "savez", "savez_compressed", "dump", "savefig",
    "write_text", "write_bytes",
}
READ_METHOD_ATTRS = {"read_text", "read_bytes"}


def first_path_arg(call: ast.Call) -> Optional[ast.AST]:
    if call.args:
        return call.args[0]
    for kw in call.keywords or []:
        if kw.arg in {"path", "filepath", "fname", "filename", "file", "path_or_buf"}:
            return kw.value
    return None


class IOScanner(ast.NodeVisitor):
    def __init__(self, mod: str, script_rel: Path, env: ModuleEnv, resolver: Resolver):
        self.mod = mod
        self.script_rel = script_rel
        self.env = env
        self.resolver = resolver
        self.records: list[IORecord] = []

    def add(self, op: str, target: str, lineno: int, hint: str):
        if target and looks_like_file(target):
            self.records.append(IORecord(str(self.script_rel), op, target, lineno, hint))

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            arg = first_path_arg(node)
            if arg is not None:
                path = expr_to_str(arg, self.env, self.resolver)
                mode = None
                if len(node.args) >= 2:
                    mode = expr_to_str(node.args[1], self.env, self.resolver)
                for kw in node.keywords or []:
                    if kw.arg == "mode":
                        mode = expr_to_str(kw.value, self.env, self.resolver)
                if path:
                    op = "read"
                    if mode and any(m in mode for m in ["w", "a", "x", "+"]):
                        op = "write"
                    self.add(op, path, node.lineno, "open()")

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr

            if attr == "savefig":
                arg = first_path_arg(node)
                if arg is not None:
                    path = expr_to_str(arg, self.env, self.resolver)
                    if path:
                        self.add("write", path, node.lineno, ".savefig()")

            if attr in READ_ATTRS:
                arg = first_path_arg(node)
                if arg is not None:
                    path = expr_to_str(arg, self.env, self.resolver)
                    if path:
                        self.add("read", path, node.lineno, f".{attr}()")

            if attr in WRITE_ATTRS:
                if attr == "dump" and len(node.args) >= 2:
                    path = expr_to_str(node.args[1], self.env, self.resolver)
                    if path:
                        self.add("write", path, node.lineno, ".dump()")
                else:
                    arg = first_path_arg(node)
                    if arg is not None:
                        path = expr_to_str(arg, self.env, self.resolver)
                        if path:
                            self.add("write", path, node.lineno, f".{attr}()")

            if attr in READ_METHOD_ATTRS:
                recv = expr_to_str(node.func.value, self.env, self.resolver)
                if recv:
                    self.add("read", recv, node.lineno, f".{attr}()")

        if isinstance(node.func, ast.Name) and node.func.id in {"dump", "load"}:
            if node.func.id == "dump" and len(node.args) >= 2:
                path = expr_to_str(node.args[1], self.env, self.resolver)
                if path:
                    self.add("write", path, node.lineno, "dump()")
            if node.func.id == "load" and node.args:
                path = expr_to_str(node.args[0], self.env, self.resolver)
                if path:
                    self.add("read", path, node.lineno, "load()")

        self.generic_visit(node)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--outdir", default="_io_report")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    modules = load_all_modules(root)
    resolver = Resolver(root, modules)

    all_records: list[IORecord] = []

    for mod, path in modules.items():
        script_rel = path.relative_to(root)
        env = resolver.get_env(mod)
        try:
            tree = resolver.ast_cache.get(mod) or ast.parse(safe_read_text(path))
        except SyntaxError:
            continue
        sc = IOScanner(mod, script_rel, env, resolver)
        sc.visit(tree)
        all_records.extend(sc.records)

    csv_path = outdir / "io_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["script", "op", "target", "lineno", "hint"])
        w.writeheader()
        for r in all_records:
            w.writerow(asdict(r))

    md_path = outdir / "io_manifest.md"
    by_script: Dict[str, list[IORecord]] = {}
    for r in all_records:
        by_script.setdefault(r.script, []).append(r)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# IO Manifest (static scan, strong resolver)\n\n")
        for s in sorted(by_script):
            f.write(f"## {s}\n\n")
            reads = [r for r in by_script[s] if r.op == "read"]
            writes = [r for r in by_script[s] if r.op == "write"]
            f.write("**Reads**\n\n")
            for r in sorted(reads, key=lambda x: x.lineno):
                f.write(f"- L{r.lineno}: `{r.target}` ({r.hint})\n")
            f.write("\n**Writes**\n\n")
            for r in sorted(writes, key=lambda x: x.lineno):
                f.write(f"- L{r.lineno}: `{r.target}` ({r.hint})\n")
            f.write("\n")

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {md_path}")


if __name__ == "__main__":
    main()
