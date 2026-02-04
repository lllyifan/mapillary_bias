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

# ---------------------------------------------
# Config: what counts as a "file-like" path
# ---------------------------------------------
FILE_SUFFIX_RE = re.compile(
    r".*\.(csv|tsv|parquet|feather|pkl|pickle|joblib|json|geojson|shp|gpkg|tif|tiff|png|jpg|jpeg|pdf|txt|md|npz|npy|xlsx)$",
    re.I,
)

def looks_like_file(s: str) -> bool:
    s = s.strip().strip("'\"")
    if FILE_SUFFIX_RE.match(s):
        return True
    # also allow patterns containing common extensions even if has {var}
    lowered = s.lower()
    return any(ext in lowered for ext in [
        ".csv", ".parquet", ".png", ".pdf", ".pkl", ".json", ".shp", ".tif", ".xlsx", ".npz", ".npy"
    ])

# ---------------------------------------------
# IO Records
# ---------------------------------------------
@dataclass
class IORecord:
    script: str
    op: str          # read/write
    target: str      # resolved-ish path expression
    lineno: int
    hint: str

# ---------------------------------------------
# Module environment: constants + import mapping
# ---------------------------------------------
@dataclass
class ModuleEnv:
    # name -> resolved string expression (e.g., "outputs", "/abs/path", "OUT_DIR/fig2.png")
    consts: Dict[str, str]
    # alias -> module_path (for "import x as y")
    imports: Dict[str, str]
    # name -> (module_path, original_name) for "from x import A as B"
    from_imports: Dict[str, Tuple[str, str]]

# ---------------------------------------------
# 1) Parse modules and collect "constants"
# ---------------------------------------------
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
    # Path("x") / PurePath("x")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"Path", "PurePath"}
    return False

def join_expr(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    # keep as path-like expression with /
    return f"{a}/{b}"

def expr_to_str(node: ast.AST, env: ModuleEnv, resolver: "Resolver") -> Optional[str]:
    """
    Best-effort resolve expression to a readable string.
    - literals
    - f-strings => keep constants + "{...}"
    - Name => env.consts or imported const if resolvable
    - Attribute => module.attr if module is imported alias and attr resolvable
    - BinOp: + for strings, / for path join
    - os.path.join(...)
    - Path("a") => "a"
    """
    # literal string
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s

    # f-string
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

    # Name
    if isinstance(node, ast.Name):
        name = node.id
        if name in env.consts:
            return env.consts[name]
        # from-import constant?
        if name in env.from_imports:
            mod, orig = env.from_imports[name]
            resolved = resolver.resolve_imported_constant(mod, orig)
            if resolved:
                return resolved
        return None

    # Attribute: e.g., paths.OUT_DIR or cfg.OUTDIR
    if isinstance(node, ast.Attribute):
        # module alias?
        if isinstance(node.value, ast.Name):
            base = node.value.id
            attr = node.attr
            # imported module alias
            if base in env.imports:
                mod = env.imports[base]
                resolved = resolver.resolve_imported_constant(mod, attr)
                if resolved:
                    return resolved
                # fallback to "mod.attr"
                return f"{mod}.{attr}"
        # could be nested Attribute; too hard
        return None

    # Path("a")
    if is_path_ctor_call(node):
        if node.args:
            inner = expr_to_str(node.args[0], env, resolver)
            return inner
        return None

    # "a" + "b"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l = expr_to_str(node.left, env, resolver)
        r = expr_to_str(node.right, env, resolver)
        if l is not None and r is not None:
            return l + r
        # salvage file-like side
        if r and looks_like_file(r):
            return r
        if l and looks_like_file(l):
            return l
        return None

    # Path join: a / b
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        l = expr_to_str(node.left, env, resolver)
        r = expr_to_str(node.right, env, resolver)
        if l is not None and r is not None:
            return join_expr(l, r)
        if r and looks_like_file(r):
            return r
        return None

    # os.path.join(...)
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
    """
    Collect simple constant assignments:
      X = "..."
      X = Path("...") / "..."
      X = OUT_DIR / f"..."
    plus import aliases.
    """
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
        # only handle simple Name targets
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

# ---------------------------------------------
# Resolver: cross-module constant resolving
# ---------------------------------------------
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
        # if that constant itself is imported, recurse a bit
        if name in env.from_imports:
            mod2, orig2 = env.from_imports[name]
            return self.resolve_imported_constant(mod2, orig2)
        return None

# ---------------------------------------------
# 2) Scan for IO calls using resolved expressions
# ---------------------------------------------
READ_ATTRS = {
    # pandas
    "read_csv", "read_table", "read_parquet", "read_feather", "read_pickle", "read_json", "read_excel",
    # geopandas
    "read_file",
    # numpy / joblib
    "load",
    # rasterio
    "open",
}
WRITE_ATTRS = {
    # pandas
    "to_csv", "to_parquet", "to_feather", "to_pickle", "to_json", "to_excel",
    # geopandas
    "to_file",
    # numpy
    "save", "savez", "savez_compressed",
    # joblib
    "dump",
    # matplotlib
    "savefig",
    # pathlib
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
        # open(...)
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

        # attribute call: X.read_csv / df.to_csv / fig.savefig / joblib.dump etc.
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr

            # any .savefig(...) (covers plt.savefig, fig.savefig, ax.figure.savefig)
            if attr == "savefig":
                arg = first_path_arg(node)
                if arg is not None:
                    path = expr_to_str(arg, self.env, self.resolver)
                    if path:
                        self.add("write", path, node.lineno, ".savefig()")

            # reads
            if attr in READ_ATTRS:
                arg = first_path_arg(node)
                if arg is not None:
                    path = expr_to_str(arg, self.env, self.resolver)
                    if path:
                        self.add("read", path, node.lineno, f".{attr}()")

            # writes
            if attr in WRITE_ATTRS:
                # special case: joblib.dump(obj, path) where path is 2nd arg
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

            # Path(...).read_text() / read_bytes()
            if attr in READ_METHOD_ATTRS:
                recv = expr_to_str(node.func.value, self.env, self.resolver)
                if recv:
                    self.add("read", recv, node.lineno, f".{attr}()")

        # direct name calls: dump/load imported directly
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

# ---------------------------------------------
# main
# ---------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="project root directory")
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

    # write csv
    csv_path = outdir / "io_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["script", "op", "target", "lineno", "hint"])
        w.writeheader()
        for r in all_records:
            w.writerow(asdict(r))

    # write md
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
