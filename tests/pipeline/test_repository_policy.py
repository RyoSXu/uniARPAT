"""Exercise real Git objects in temporary repositories, without remote writes."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.checks.check_repository import (
    MAX_BLOB_BYTES, ROOT, check_index, check_push, check_revisions, check_worktree,
)


class TestRepositoryPolicy(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.git("init", "--quiet")
        (self.root / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
        self.git("add", ".gitignore")

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.root), *arguments],
                                       stderr=subprocess.DEVNULL).decode().strip()

    def commit_tree(self, parent=None):
        arguments = ["-c", "user.name=Repository Policy Test", "-c",
                     "user.email=policy@example.invalid", "commit-tree",
                     self.git("write-tree"), "-m", "policy fixture"]
        if parent is not None:
            arguments.extend(("-p", parent))
        return self.git(*arguments)

    def updates(self, local, remote):
        return f"refs/heads/main {local} refs/heads/main {remote}\n"

    def remote_payload(self):
        (self.root / "predictions.npy").write_bytes(b"already remote predictions")
        self.git("add", "-f", "predictions.npy")
        return self.commit_tree()

    def test_force_added_small_array_is_rejected(self):
        (self.root / "cache.npy").write_bytes(b"local array")
        self.git("add", "-f", "cache.npy")
        self.assertTrue(any("ignored payload" in failure for failure in check_index(self.root)))

    def test_compressed_git_blob_still_checks_original_size(self):
        (self.root / "oversized.py").write_bytes(b"#" + b"x" * MAX_BLOB_BYTES)
        self.git("add", "oversized.py")
        self.assertTrue(any("exceeds 2 MiB" in failure for failure in check_index(self.root)))

    def test_required_q1_metadata_can_be_versioned(self):
        directory = self.root / "data/train4ARPAT"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text('{"splits": {}}\n')
        self.git("add", "data/train4ARPAT/manifest.json")
        self.assertEqual(check_index(self.root), [])

    def test_removed_worktree_payload_remains_blocked_in_old_tree(self):
        (self.root / "predictions.npy").write_bytes(b"local predictions")
        self.git("add", "-f", "predictions.npy")
        old_tree = self.git("write-tree")
        self.git("rm", "--cached", "predictions.npy")
        self.assertEqual(check_index(self.root), [])
        self.assertTrue(check_revisions(self.root, [old_tree]))

    def test_worktree_ignores_local_payload_but_rejects_large_source(self):
        (self.root / "cache.npy").write_bytes(b"x" * (MAX_BLOB_BYTES + 1))
        self.assertEqual(check_worktree(self.root)[0], [])
        (self.root / "large.py").write_bytes(b"#" + b"x" * MAX_BLOB_BYTES)
        self.assertTrue(check_worktree(self.root)[0])

    def test_shared_remote_payload_does_not_block_clean_final_tree(self):
        remote = self.remote_payload()
        (self.root / "README.md").write_text("documentation-only change\n")
        self.git("add", "README.md")
        intermediate = self.commit_tree(remote)
        self.git("rm", "--cached", "predictions.npy")
        local = self.commit_tree(intermediate)
        failures, count = check_push(self.root, "origin", self.updates(local, remote))
        self.assertEqual(failures, [])
        self.assertEqual(count, 2)

    def test_new_intermediate_payload_is_rejected_after_final_deletion(self):
        remote = self.commit_tree()
        (self.root / "predictions.npy").write_bytes(b"new local predictions")
        self.git("add", "-f", "predictions.npy")
        intermediate = self.commit_tree(remote)
        self.git("rm", "--cached", "predictions.npy")
        local = self.commit_tree(intermediate)
        failures, _ = check_push(self.root, "origin", self.updates(local, remote))
        self.assertTrue(any("ignored payload" in failure for failure in failures))

    def test_shared_payload_in_final_tree_is_still_rejected(self):
        remote = self.remote_payload()
        (self.root / "README.md").write_text("payload remains versioned\n")
        self.git("add", "README.md")
        local = self.commit_tree(remote)
        failures, _ = check_push(self.root, "origin", self.updates(local, remote))
        self.assertTrue(any("ignored payload" in failure for failure in failures))

    def test_new_remote_rejects_old_payload_removed_from_final_tree(self):
        remote = self.remote_payload()
        self.git("update-ref", "refs/remotes/other/main", remote)
        self.git("rm", "--cached", "predictions.npy")
        local = self.commit_tree(remote)
        failures, _ = check_push(self.root, "new-remote", self.updates(local, "0" * 40))
        self.assertTrue(any("ignored payload" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
