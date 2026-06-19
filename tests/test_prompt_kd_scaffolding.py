import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stage_prompt_kd.checkpointing import (
    load_training_checkpoint,
    resolve_wandb_run_id,
    save_training_checkpoint,
)
from stage_prompt_kd.losses import box_l1_loss, feature_mse_loss, mask_bce_dice_loss
from stage_prompt_kd.manifest import bbox_from_mask, point_from_mask, read_jsonl, write_jsonl
from stage_prompt_kd.shape_check import expected_tinyvit21_shapes, tinyvit_final_resolution


class PromptKDScaffoldingTest(unittest.TestCase):
    def test_tinyvit21_expected_shapes_match_sam3_target(self):
        self.assertEqual(tinyvit_final_resolution(1008), 32)
        shapes = expected_tinyvit21_shapes(
            batch_size=2,
            img_size=1008,
            embed_dim=1024,
            embed_size=72,
        )
        self.assertEqual(shapes["tinyvit_raw"], [2, 576, 32, 32])
        self.assertEqual(shapes["student_projected"], [2, 1024, 72, 72])

    def test_losses_are_finite(self):
        pred = torch.zeros(2, 1, 8, 8)
        target = torch.ones(2, 1, 8, 8)
        self.assertTrue(torch.isfinite(feature_mse_loss(pred, target)))
        self.assertTrue(torch.isfinite(mask_bce_dice_loss(pred, target)))
        self.assertTrue(torch.isfinite(box_l1_loss(torch.zeros(2, 4), torch.ones(2, 4))))

    def test_manifest_helpers(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        self.assertEqual(bbox_from_mask(mask), [3, 2, 7, 5])
        self.assertIsNotNone(point_from_mask(mask))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl([{"source_id": "a", "image": "a.jpg"}], Path(tmp) / "m.jsonl")
            self.assertEqual(read_jsonl(path), [{"source_id": "a", "image": "a.jpg"}])

    def test_checkpoint_and_wandb_run_id_resume(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            latest = save_training_checkpoint(
                out,
                epoch=3,
                global_step=9,
                model=model,
                optimizer=optimizer,
                wandb_run_id="run-a",
            )
            loaded = load_training_checkpoint(latest, model=model, optimizer=optimizer)
            self.assertEqual(loaded["epoch"], 3)
            self.assertEqual(loaded["global_step"], 9)
            self.assertEqual(loaded["wandb_run_id"], "run-a")
            run_id = resolve_wandb_run_id(out, None, generate_id=lambda: "new-run")
            self.assertEqual(run_id, "new-run")
            self.assertEqual(resolve_wandb_run_id(out, None, generate_id=lambda: "unused"), "new-run")


if __name__ == "__main__":
    unittest.main()
