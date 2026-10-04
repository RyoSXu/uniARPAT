#!/usr/bin/env python3
"""Reject ignored payloads and oversized blobs before committing or pushing.

The .gitignore reviewed evidence exceptions are the single policy source.
Historical paths are translated by the authenticated migration registry so a
new remote cannot receive old raw data merely because the current tree is clean.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import repository_path

MAX_BLOB_BYTES = 2 * 1024 * 1024


def git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *arguments], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def ignored_names(root: Path, names: list[str]) -> set[str]:
    if not names:
        return set()
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "-z", "--stdin"],
        input=b"".join(os.fsencode(name) + b"\0" for name in names),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return {os.fsdecode(name) for name in result.stdout.split(b"\0") if name}


def blob_sizes(root: Path, object_ids: list[str]) -> dict[str, int]:
    if not object_ids:
        return {}
    output = git(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                 data=("\n".join(sorted(set(object_ids))) + "\n").encode())
    sizes = {}
    for line in output.decode().splitlines():
        oid, kind, size = line.split()
        if kind != "blob":
            raise ValueError(f"expected a blob: {oid}")
        sizes[oid] = int(size)
    return sizes


def tree_entries(root: Path, revision: str | None = None) -> list[tuple[str, str]]:
    entries = []
    if revision is None:
        for item in git(root, "ls-files", "--stage", "-z").split(b"\0"):
            if not item:
                continue
            header, name = item.split(b"\t", 1)
            mode, oid, stage = header.decode().split()
            if stage != "0":
                raise ValueError("resolve unmerged index entries before committing")
            if mode == "160000":
                continue
            entries.append((os.fsdecode(name), oid))
    else:
        for item in git(root, "ls-tree", "-r", "-z", revision).split(b"\0"):
            if not item:
                continue
            header, name = item.split(b"\t", 1)
            _, kind, oid = header.decode().split()
            if kind == "blob":
                entries.append((os.fsdecode(name), oid))
    return entries


def check_entries(root: Path, entries: list[tuple[str, str]]) -> list[str]:
    mapped = {}
    for name, oid in entries:
        mapped[name, oid] = repository_path(name, root=root).relative_to(root).as_posix()
    forbidden = ignored_names(root, list(set(mapped.values())))
    sizes = blob_sizes(root, [oid for _, oid in entries])
    failures = []
    for name, oid in entries:
        target = mapped[name, oid]
        if target in forbidden:
            failures.append(f"ignored payload is versioned: {name} -> {target}")
        if sizes[oid] > MAX_BLOB_BYTES:
            failures.append(f"blob exceeds 2 MiB: {name} ({sizes[oid]} bytes)")
    return failures


def check_index(root: Path) -> list[str]:
    policy = root / ".gitignore"
    if policy.is_file() and git(root, "show", ":.gitignore") != policy.read_bytes():
        return ["stage the reviewed .gitignore before checking the commit"]
    return check_entries(root, tree_entries(root))


def check_worktree(root: Path) -> tuple[list[str], int]:
    files = []
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = [name for name in children if name != ".git"]
        for name in names:
            path = Path(directory) / name
            files.append(path)
    relative = [path.relative_to(root).as_posix() for path in files]
    ignored = ignored_names(root, relative)
    candidates = [path for path, name in zip(files, relative) if name not in ignored]
    failures = [f"file exceeds 2 MiB: {path.relative_to(root)} ({path.lstat().st_size} bytes)"
                for path in candidates if path.lstat().st_size > MAX_BLOB_BYTES]
    return failures, len(candidates)


def push_ranges(root: Path, remote: str, updates: str) -> list[tuple[str, tuple[str, ...]]]:
    ranges = []
    for line in updates.splitlines():
        if not line.strip():
            continue
        _, local_oid, _, remote_oid = line.split()
        if set(local_oid) == {"0"}:
            continue
        if set(remote_oid) == {"0"}:
            arguments = (local_oid, "--not", f"--remotes={remote}")
        else:
            exists = subprocess.run(["git", "-C", str(root), "cat-file", "-e", remote_oid],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            arguments = (f"{remote_oid}..{local_oid}",) if exists.returncode == 0 else (
                local_oid, "--not", f"--remotes={remote}",
            )
        ranges.append((local_oid, arguments))
    return ranges


def push_revisions(root: Path, remote: str, updates: str) -> list[str]:
    revisions = set()
    for _, arguments in push_ranges(root, remote, updates):
        revisions.update(git(root, "rev-list", *arguments).decode().splitlines())
    return sorted(revisions)


def check_revisions(root: Path, revisions: list[str]) -> list[str]:
    entries = set()
    for revision in revisions:
        entries.update(tree_entries(root, revision))
    return check_entries(root, sorted(entries))


def check_push(root: Path, remote: str, updates: str) -> tuple[list[str], int]:
    """Check final trees and objects absent from the advertised remote base.

    Shared historical blobs do not need uploading again. Every final tree must
    still obey today's policy, while new payloads in intermediate commits remain
    forbidden even if a later commit deletes them. A new remote checks history.
    """
    ranges = push_ranges(root, remote, updates)
    revisions = set()
    new_objects = set()
    final_entries = set()
    for local_oid, arguments in ranges:
        revisions.update(git(root, "rev-list", *arguments).decode().splitlines())
        new_objects.update(line.split(b" ", 1)[0].decode() for line in
                           git(root, "rev-list", "--objects", *arguments).splitlines())
        final_entries.update(tree_entries(root, local_oid))
    entries = set(final_entries)
    for revision in revisions:
        entries.update((name, oid) for name, oid in tree_entries(root, revision)
                       if oid in new_objects)
    return check_entries(root, sorted(entries)), len(revisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worktree", action="store_true", help="check prospective unignored files")
    mode.add_argument("--index", action="store_true", help="check every blob in the staged tree")
    mode.add_argument("--revision", help="check a historical tree against today's policy")
    mode.add_argument("--pre-push", metavar="REMOTE", help="read Git pre-push updates from stdin")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.index:
            failures = check_index(root)
            detail = "index"
        elif args.revision:
            failures = check_revisions(root, [args.revision])
            detail = "revision"
        elif args.pre_push:
            failures, count = check_push(root, args.pre_push, sys.stdin.read())
            detail = f"{count} outgoing commits and final trees"
        else:
            failures, count = check_worktree(root)
            detail = f"{count} prospective files"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"repository policy check failed: {error}", file=sys.stderr)
        return 1
    if failures:
        print("Repository payload policy rejected the operation:", file=sys.stderr)
        for failure in failures[:30]:
            print(f"  {failure}", file=sys.stderr)
        if len(failures) > 30:
            print(f"  ... {len(failures) - 30} more violations", file=sys.stderr)
        print("Keep raw/intermediate data local. Historical blobs require a separately approved cleanup.",
              file=sys.stderr)
        return 1
    print(json.dumps({"status": "pass", "checked": detail, "max_blob_bytes": MAX_BLOB_BYTES}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
