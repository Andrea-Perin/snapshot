#!/usr/bin/env python3
"""
Snapshot an experiment script and ALL of its local (project-relative) dependencies.
Only files that actually get imported are copied — nothing else.

Usage:
    python snapshot_experiment.py my_experiment.py
    python snapshot_experiment.py my_experiment.py --out snapshots/ --root .
"""

import ast
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque


# ---------------------------------------------------------------------------
# 1. AST-based import extraction
# ---------------------------------------------------------------------------

def extract_imports(file_path: Path) -> list[tuple[str | None, str, int]]:
    """
    Parse a .py file and return a list of (from_module, import_name, level) tuples.

    - Plain `import foo.bar`         → (None,      "foo.bar", 0)
    - `from foo import bar`          → ("foo",     "bar",     0)
    - `from . import bar`            → ("",        "bar",     1)   ← relative
    - `from ..utils import helper`   → ("utils",   "helper",  2)   ← relative
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((None, alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                # For `from pkg import name`, we want to try both:
                #   pkg/name.py  and  pkg.py  (name might be a symbol, not a module)
                imports.append((module, alias.name, node.level))
    return imports


# ---------------------------------------------------------------------------
# 2. Resolving an import to a local file path
# ---------------------------------------------------------------------------

def _candidates(dotted: str, root: Path) -> list[Path]:
    """Given 'foo.bar.baz', yield candidate paths under root."""
    parts = dotted.replace(".", "/")
    return [
        root / f"{parts}.py",
        root / parts / "__init__.py",
    ]


def resolve_import(
    from_module: str | None,
    name: str,
    level: int,
    from_file: Path,
    root: Path,
) -> list[Path]:
    """
    Try to map an import statement to one or more local Path objects.
    Returns a (possibly empty) list of resolved paths inside `root`.
    """
    resolved = []

    if level > 0:
        # Relative import: anchor is the package directory of from_file
        anchor = from_file.parent
        for _ in range(level - 1):          # each extra dot goes up one more level
            anchor = anchor.parent

        # Build the full dotted module from anchor
        parts = []
        if from_module:
            parts.append(from_module)
        parts.append(name)
        combined = ".".join(parts)

        for p in _candidates(combined, anchor):
            if p.exists() and _is_inside(p, root):
                resolved.append(p)

        # Also try just the from_module part (name may be a symbol)
        if from_module:
            for p in _candidates(from_module, anchor):
                if p.exists() and _is_inside(p, root):
                    resolved.append(p)
    else:
        # Absolute import — try from project root
        candidates_to_try = []

        if from_module:
            full = f"{from_module}.{name}"
            candidates_to_try += _candidates(full, root)
            candidates_to_try += _candidates(from_module, root)  # name may be a symbol
        else:
            candidates_to_try += _candidates(name, root)

        for p in candidates_to_try:
            if p.exists() and _is_inside(p, root):
                resolved.append(p)

    return list(dict.fromkeys(resolved))  # deduplicate, preserve order


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# 3. BFS over the local dependency graph
# ---------------------------------------------------------------------------

def collect_dependencies(entry: Path, root: Path) -> set[Path]:
    """
    Starting from `entry`, follow all local imports transitively.
    Returns the full set of local .py files (including entry itself).
    """
    visited: set[Path] = set()
    queue: deque[Path] = deque([entry.resolve()])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        for from_mod, name, level in extract_imports(current):
            for resolved in resolve_import(from_mod, name, level, current, root):
                rr = resolved.resolve()
                if rr not in visited:
                    queue.append(rr)

    return visited


# ---------------------------------------------------------------------------
# 4. Snapshot: copy files, preserve relative structure, write metadata
# ---------------------------------------------------------------------------

def snapshot(entry_script: str, out_base: str = "snapshots", root: str = ".") -> Path:
    entry = Path(entry_script).resolve()
    root_path = Path(root).resolve()

    if not entry.exists():
        sys.exit(f"[snapshot] Entry file not found: {entry}")
    if not _is_inside(entry, root_path):
        sys.exit(f"[snapshot] Entry file must be inside the project root ({root_path})")

    print(f"[snapshot] Tracing dependencies of {entry.name} ...")
    deps = collect_dependencies(entry, root_path)

    # Name the snapshot after the script + timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_name = f"{entry.stem}__{ts}"
    snap_dir = Path(out_base).resolve() / snap_name
    snap_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for dep in sorted(deps):
        rel = dep.relative_to(root_path)
        dest = snap_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dep, dest)
        copied.append(str(rel))
        print(f"  + {rel}")

    # Write a small metadata sidecar
    meta = {
        "entry": str(entry.relative_to(root_path)),
        "root": str(root_path),
        "timestamp": ts,
        "python": sys.version,
        "files_captured": sorted(copied),
        "file_count": len(copied),
    }
    (snap_dir / "_snapshot_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n[snapshot] Saved {len(copied)} file(s) → {snap_dir}")
    return snap_dir


# ---------------------------------------------------------------------------
# 5. In-script convenience function
# ---------------------------------------------------------------------------

def snap(out_base: str = "snapshots", root: str = ".") -> Path:
    """
    Call this from inside your experiment script to snapshot itself.

    Usage:
        from snapshot_experiment import snap
        snap()                            # at the top, or just before the run
        snap(out_base="my_snaps", root="/path/to/project")

    The caller's __file__ is detected automatically via the call stack.
    """
    import inspect

    # Walk up the call stack to find the first frame that isn't this file.
    this_file = Path(__file__).resolve()
    caller_file = None
    for frame_info in inspect.stack():
        candidate = Path(frame_info.filename).resolve()
        if candidate != this_file:
            caller_file = candidate
            break

    if caller_file is None:
        sys.exit("[snap] Could not determine the calling script.")

    return snapshot(str(caller_file), out_base=out_base, root=root)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Snapshot an experiment script and its local dependencies."
    )
    parser.add_argument("entry", help="The experiment script to snapshot (e.g. run_exp.py)")
    parser.add_argument(
        "--out", default="snapshots",
        help="Output base directory for snapshots (default: ./snapshots)"
    )
    parser.add_argument(
        "--root", default=".",
        help="Project root — only imports inside this directory are considered local "
             "(default: current working directory)"
    )
    args = parser.parse_args()
    snapshot(args.entry, args.out, args.root)


if __name__ == "__main__":
    main()
