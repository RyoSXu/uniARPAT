"""Repository locations and authenticated lookup of pre-migration evidence.

Built-in locations are independent of the caller's working directory. Explicit
user paths stay relative to that working directory unless they use an old
repository location recorded in the migration manifest.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "data/reference"
Q1_DATA_ROOT = PROJECT_ROOT / "data/train4ARPAT"
CRYSTAL_Q1_DATA_ROOT = PROJECT_ROOT / "data/crystal_q1_v1"
REPORTS_ROOT = PROJECT_ROOT / "results/reports"
RUNS_ROOT = PROJECT_ROOT / "results/runs"
FIGURES_ROOT = PROJECT_ROOT / "results/figures"
DEFAULT_CONFIG = PROJECT_ROOT / "tools/training-defaults.yaml"
MIGRATION_MANIFEST = "archive/metadata/repository-migration-2026-10-04.json"


@lru_cache(maxsize=8)
def _migration(root: Path) -> dict:
    path = root / MIGRATION_MANIFEST
    if not path.is_file():
        path = PROJECT_ROOT / MIGRATION_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_name(value: str | Path, root: Path) -> str | None:
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            return None
    return path.as_posix()


def repository_path(value: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve a repository path, including recorded old locations, once."""
    base = Path(root) if root is not None else PROJECT_ROOT
    name = _relative_name(value, base)
    if name is None:
        return Path(value).expanduser()
    migration = _migration(base)
    visited = set()
    while name not in visited:
        visited.add(name)
        # Exact aliases take priority: some first-layout artifacts are archived.
        translated = migration["path_moves"].get(name)
        if translated is None:
            if any(name == prefix or name.startswith(prefix + "/") for prefix in (
                "results/reports", "results/runs", "results/figures", "archive",
            )):
                break
            for old, new in sorted(migration["prefix_moves"].items(), key=lambda item: -len(item[0])):
                if name.startswith(old):
                    translated = new + name[len(old):]
                    break
        if translated is None or translated == name:
            break
        name = translated
    else:
        raise ValueError(f"cyclic repository path migration: {value}")
    return base / name


def input_path(value: str | Path, *, root: str | Path | None = None) -> Path:
    """Keep custom CLI paths; translate known repository inputs/default Q1."""
    base = Path(root) if root is not None else PROJECT_ROOT
    path = Path(value).expanduser()
    name = _relative_name(path, base)
    if name is not None and (
        name in {"data/train4ARPAT", "data/crystal_q1_v1"} or name in {
            "configs/default.yaml", "default.yaml", "tools/training-defaults.yaml",
        }
        or any(name.startswith(prefix) for prefix in (
            "output/", "index/", "data/grids_c2b/", "results/",
            "data/reference/", "data/archive/", "archive/",
        ))
    ):
        return repository_path(path, root=base)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_name(name: str | None, migration: dict) -> str | None:
    return migration.get("source_aliases", {}).get(name, name)


def _git_source(base: Path, identity: dict, expected_digest: str) -> Path:
    """Materialize a verified original in an ignored cache; never execute it."""
    commit, name = identity["commit"], identity["path"]
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("invalid recorded Git source identity")
    result = subprocess.run(["git", "-C", str(base), "cat-file", "blob", f"{commit}:{name}"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected_digest:
        raise ValueError(f"recorded Git source unavailable or mismatched: {name}")
    cache = base / "results/runs/repository_sources" / (expected_digest + ".txt")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        if file_sha256(cache) != expected_digest:
            raise ValueError(f"historical source cache was modified: {cache}")
    else:
        cache.write_bytes(result.stdout)
    return cache


def historical_path(
    value: str | Path, expected_digest: str, *, root: str | Path | None = None,
) -> Path:
    """Find the exact recorded bytes; missing/mismatched evidence raises.

    A renamed artifact can still satisfy its original hash. A source file edited
    during migration must instead resolve to an authenticated saved version.
    This lookup is for verification, never a choice of code to execute.
    """
    if len(expected_digest) != 64 or any(c not in "0123456789abcdef" for c in expected_digest):
        raise ValueError("expected_digest must be a complete lowercase SHA-256")
    base = Path(root) if root is not None else PROJECT_ROOT
    candidate = repository_path(value, root=base)
    if candidate.is_file() and file_sha256(candidate) == expected_digest:
        return candidate
    migration = _migration(base)
    name = _source_name(_relative_name(value, base), migration)
    snapshot = migration["source_versions"].get(name, {}).get(expected_digest)
    if snapshot is not None:
        archived = base / snapshot
        if archived.is_file() and file_sha256(archived) == expected_digest:
            return archived
    identity = migration.get("source_git_versions", {}).get(name, {}).get(expected_digest)
    if identity is not None:
        return _git_source(base, identity, expected_digest)
    raise ValueError(f"historical SHA-256 cannot be verified: {value} ({expected_digest})")


def protected_path(
    value: str | Path, expected_digest: str, *, root: str | Path | None = None,
) -> Path:
    """Check an input to a live tool, accepting only the recorded path migration.

    Historical verification alone cannot authorize execution of edited code.
    If current bytes differ, require both the authenticated original and the
    exact post-migration hash. Subsequent source edits still fail the guard.
    """
    base = Path(root) if root is not None else PROJECT_ROOT
    current = repository_path(value, root=base)
    actual = file_sha256(current)
    if actual == expected_digest:
        return current
    migration = _migration(base)
    name = _source_name(_relative_name(value, base), migration)
    before = migration.get("source_before_sha256", {}).get(name)
    after = migration.get("source_after_sha256", {}).get(name)
    if before == expected_digest and after == actual:
        historical_path(value, expected_digest, root=base)
        return current
    raise ValueError(f"protected input changed beyond the recorded migration: {value}")
