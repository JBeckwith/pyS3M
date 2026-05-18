#!/usr/bin/env python3
"""
Audit requirements.txt against actual codebase imports.

Scans:
  - src/, unit_tests/, scripts/, claude/  (AST import extraction)
  - src/ImportManager.py                   (string-based get_module() calls)
  - notebooks/**/*.ipynb                   (JSON cell source, no temp files)

Uses importlib.metadata.packages_distributions() to map dist names → import names.

Usage:
    python claude/audit_requirements.py           # dry run
    python claude/audit_requirements.py --apply   # rewrite requirements.txt
"""

import ast
import json
import re
import warnings
import argparse
from pathlib import Path
from importlib.metadata import packages_distributions

REPO_ROOT   = Path(__file__).resolve().parent.parent
REQS_FILE   = REPO_ROOT / 'requirements.txt'
IMPORT_MGR  = REPO_ROOT / 'src' / 'ImportManager.py'

SCAN_PY_DIRS = ['src', 'unit_tests', 'scripts', 'claude']
SCAN_NB_DIRS = ['notebooks']

# Packages that are never directly imported but are required at runtime as
# backends for other packages.  Always kept regardless of import scan results.
ALWAYS_KEEP: set[str] = {
    'tables',       # PyTables — pandas requires it at runtime for read_hdf / to_hdf
}

# Packages whose pip name differs from their top-level import name.
# These supplement packages_distributions() for cases it gets wrong or misses.
IMPORT_TO_PKG_OVERRIDE: dict[str, str] = {
    'PIL':                'pillow',
    'cv2':                'opencv-python',
    'sklearn':            'scikit-learn',
    'skimage':            'scikit-image',
    'yaml':               'pyyaml',
    'bs4':                'beautifulsoup4',
    'dateutil':           'python-dateutil',
    'attr':               'attrs',
    'mpl_toolkits':       'matplotlib',
    'colour_demosaicing': 'colour-demosaicing',
    'fast_hdbscan':       'fast-hdbscan',
    'frc':                'frc',
    'fpbase':             'fpbase',
    'diplib':             'diplib',
}


# ── normalisation ─────────────────────────────────────────────────────────────

def norm(name: str) -> str:
    """Lowercase and collapse separators so numpy, Numpy, numpy_ all match."""
    return re.sub(r'[-_.]+', '_', name).lower()


# ── import collection ─────────────────────────────────────────────────────────

def imports_from_source(source: str) -> set[str]:
    """Return top-level module names (first dotted component) from Python source."""
    names: set[str] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split('.')[0])
    return names


def imports_from_notebook(path: Path) -> set[str]:
    """Parse a .ipynb as JSON and extract imports from code cells directly."""
    try:
        nb = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return set()
    names: set[str] = set()
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            names |= imports_from_source(source)
    return names


def string_modules_from_import_manager(path: Path) -> set[str]:
    """
    Extract module names passed as string literals to get_module() in
    ImportManager.py, which AST import extraction cannot see.
    """
    if not path.exists():
        return set()
    # Grab every quoted string that looks like a dotted Python identifier
    raw = path.read_text(encoding='utf-8', errors='ignore')
    candidates = re.findall(r'"([a-zA-Z][a-zA-Z0-9_.]*)"', raw)
    # Keep only plausible module names (not single words that are clearly not packages)
    skip = {'available', 'missing', 'error', 'np', 'pd', 'ds', 'cc',
            'plt', 'sns', '__main__'}
    return {c.split('.')[0] for c in candidates if c not in skip}


def all_used_imports() -> set[str]:
    used: set[str] = set()

    for d in SCAN_PY_DIRS:
        p = REPO_ROOT / d
        if p.exists():
            for f in p.rglob('*.py'):
                try:
                    used |= imports_from_source(
                        f.read_text(encoding='utf-8', errors='ignore')
                    )
                except Exception:
                    pass

    for d in SCAN_NB_DIRS:
        p = REPO_ROOT / d
        if p.exists():
            for f in p.rglob('*.ipynb'):
                used |= imports_from_notebook(f)

    used |= string_modules_from_import_manager(IMPORT_MGR)

    return used


# ── dist → import-name mapping ────────────────────────────────────────────────

def build_dist_to_imports() -> dict[str, set[str]]:
    """
    Build {norm(dist_name): {import_name, ...}} from packages_distributions(),
    then add manual overrides.
    """
    mapping: dict[str, set[str]] = {}

    for import_name, dist_names in packages_distributions().items():
        for dist in dist_names:
            mapping.setdefault(norm(dist), set()).add(import_name)

    for import_name, pkg in IMPORT_TO_PKG_OVERRIDE.items():
        mapping.setdefault(norm(pkg), set()).add(import_name)

    return mapping


# ── requirements.txt parsing ──────────────────────────────────────────────────

def pkg_name_from_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('-'):
        return None
    return re.split(r'[=<>!;@\[]', line)[0].strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true',
                        help='Rewrite requirements.txt, removing unused packages.')
    args = parser.parse_args()

    print('Scanning codebase for imports...')
    used = all_used_imports()
    used_norm = {norm(n) for n in used}
    nb_count = sum(1 for d in SCAN_NB_DIRS
                   for _ in (REPO_ROOT / d).rglob('*.ipynb')
                   if (REPO_ROOT / d).exists())
    py_count = sum(1 for d in SCAN_PY_DIRS
                   for _ in (REPO_ROOT / d).rglob('*.py')
                   if (REPO_ROOT / d).exists())
    print(f'  {py_count} .py files, {nb_count} notebooks → '
          f'{len(used)} unique top-level names\n')

    dist_to_imports = build_dist_to_imports()
    lines = REQS_FILE.read_text(encoding='utf-8').splitlines()

    keep_lines: list[str] = []
    removable:  list[str] = []
    unknown:    list[str] = []

    for line in lines:
        pkg = pkg_name_from_line(line)
        if pkg is None:
            keep_lines.append(line)
            continue

        key = norm(pkg)
        provided = dist_to_imports.get(key, None)

        if norm(pkg) in {norm(k) for k in ALWAYS_KEEP}:
            keep_lines.append(line)
        elif provided is None:
            # Package not found in installed metadata — can't judge, keep it
            unknown.append(pkg)
            keep_lines.append(line)
        elif provided & used or key in used_norm:
            keep_lines.append(line)
        else:
            removable.append(pkg)

    # ── report ────────────────────────────────────────────────────────────────
    print('=' * 60)
    print(f'SAFE TO REMOVE ({len(removable)} packages)')
    print('=' * 60)
    if removable:
        for p in sorted(removable, key=str.lower):
            print(f'  - {p}')
    else:
        print('  (none)')

    print()
    print('=' * 60)
    print(f'COULD NOT DETERMINE — kept ({len(unknown)} packages)')
    print('=' * 60)
    if unknown:
        for p in sorted(unknown, key=str.lower):
            print(f'  ? {p}')
    else:
        print('  (none)')

    print()
    print('NOTE: "safe to remove" includes transitive dependencies that pip installs')
    print('automatically when their parent package is installed. Removing them from')
    print('requirements.txt is correct — they will still be present in the environment.')
    print()
    if args.apply:
        REQS_FILE.write_text('\n'.join(keep_lines) + '\n', encoding='utf-8')
        print(f'✅  Rewrote {REQS_FILE.name} — removed {len(removable)} line(s).')
    else:
        kept = len(lines) - len(removable)
        print(f'Dry run: {len(removable)} removable, {kept} to keep. '
              f'Pass --apply to rewrite {REQS_FILE.name}.')


if __name__ == '__main__':
    main()
