import tempfile
import unittest
from pathlib import Path

from stage1.utils import resolve_wandb_run_id


class Stage1CheckpointPolicyTest(unittest.TestCase):
    def test_resolve_wandb_run_id_persists_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_id = resolve_wandb_run_id(str(output), "", lambda: "generated")
            self.assertEqual(run_id, "generated")
            self.assertEqual((output / "wandb_run_id.txt").read_text().strip(), "generated")
            resumed = resolve_wandb_run_id(str(output), "", lambda: "unused")
            self.assertEqual(resumed, "generated")

    def test_requested_wandb_run_id_overrides_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_id = resolve_wandb_run_id(str(output), "manual", lambda: "unused")
            self.assertEqual(run_id, "manual")
            self.assertEqual((output / "wandb_run_id.txt").read_text().strip(), "manual")


if __name__ == "__main__":
    unittest.main()
