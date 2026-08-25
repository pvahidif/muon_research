from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str) -> str:
    """Resolve a config-supplied path against REPO_ROOT; an already-absolute
    path passes through unchanged. Use this for every path a config file
    names (data_source, another run's own run_path, ...) so configs stay
    portable across machines/checkouts instead of baking in one machine's
    absolute layout. To point data at a different disk/mount, symlink
    <REPO_ROOT>/data there rather than hardcoding an absolute path."""
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)
