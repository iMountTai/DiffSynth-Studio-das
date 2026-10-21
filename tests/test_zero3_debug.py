"""CPU-only checks for diagnostic routing; no DeepSpeed/HCU validation."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "zero3_debug", Path(__file__).resolve().parents[1] / "diffsynth/diffusion/zero3_debug.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeZero:
    def __init__(self):
        self.param = torch.nn.Parameter(torch.ones(4))
        self.param.ds_id = 7
        self.param.ds_tensor = torch.ones(4)
        self.param.grad = torch.ones(4)
        self.fp16_groups = [[self.param]]
        self.fp32_partitioned_groups_flat = [torch.nn.Parameter(torch.ones(4))]
        self.fp32_partitioned_groups_flat[0].grad = torch.ones(4)
        self.optimizer = types.SimpleNamespace(state={})
        self.loss_scale = 1.0
        self.clip_grad = 1.0
        self.swap_optimizer = False

    def reduce_independent_p_g_buckets_and_remove_grads(self, param):
        return param

    def partition_grads(self, params_to_release, grad_partitions):
        self.fp32_partitioned_groups_flat[0].grad = grad_partitions[0].clone()

    def _prepare_sub_group(self, sub_group_id, timer_names):
        pass

    def unscale_and_clip_grads(self, sub_group_id, total_norm):
        self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(0.5)

    def _optimizer_step(self, sub_group_id):
        p = self.fp32_partitioned_groups_flat[sub_group_id]
        with torch.no_grad():
            p.add_(p.grad, alpha=-0.1)
        self.optimizer.state[p] = {"exp_avg": p.grad.clone()}

    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        self.param.ds_tensor.copy_(self.fp32_partitioned_groups_flat[sub_group_id].detach())

    def step(self):
        return "step"


class DebugTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"DIFFSYNTH_ZERO3_DEBUG": "full",
                             "DIFFSYNTH_ZERO3_DEBUG_SYNC": "none",
                             "DIFFSYNTH_ZERO3_DEBUG_DIR": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        ds = patch.dict("sys.modules", {"deepspeed": types.SimpleNamespace(__version__="mock")})
        ds.start()
        self.addCleanup(ds.stop)
        self.zero = FakeZero()
        self.engine = types.SimpleNamespace(optimizer=self.zero, named_parameters=lambda: [("weight", self.zero.param)])
        plugin = types.SimpleNamespace(deepspeed_config={"zero_optimization": {"stage": 3}})
        self.acc = types.SimpleNamespace(process_index=0, device=torch.device("cpu"),
                                         state=types.SimpleNamespace(deepspeed_plugin=plugin))

    def create(self):
        debug = module.create_zero3_debug(self.acc, self.engine, self.tmp.name)
        self.addCleanup(debug.close)
        return debug

    def run_step(self, debug, bad=False):
        debug.begin(0, 0)
        debug.loss(torch.tensor(1.0))
        self.zero.reduce_independent_p_g_buckets_and_remove_grads(self.zero.param)
        reduced = torch.tensor([float("nan"), 1, 1, 1]) if bad else torch.ones(4)
        self.zero.partition_grads([self.zero.param], [reduced])
        self.zero._prepare_sub_group(0, set())
        self.zero.unscale_and_clip_grads(0, torch.tensor(2.0))
        self.zero._optimizer_step(0)
        self.zero._reassign_or_swap_out_partitioned_parameters(0)
        debug.boundary("after_update")

    def test_finite_path_preserves_update_and_restores_hooks(self):
        debug = self.create()
        self.run_step(debug)
        torch.testing.assert_close(self.zero.param.ds_tensor, torch.full((4,), 0.95))
        records = [json.loads(line) for line in debug.path.read_text().splitlines()]
        self.assertTrue(any(r["stage"] == "optimizer.after.state" for r in records))
        debug.close()
        self.assertNotIn("partition_grads", vars(self.zero))

    def test_nonfinite_reduced_gradient_is_first_and_stops_at_boundary(self):
        debug = self.create()
        with self.assertRaisesRegex(RuntimeError, "after_update"):
            self.run_step(debug, bad=True)
        self.assertEqual(debug.first_bad["stage"], "grad.reduced")
        self.assertEqual(debug.first_bad["ds_id"], 7)
        records = [json.loads(line) for line in debug.path.read_text().splitlines()]
        self.assertTrue(records[-1]["any_rank_nonfinite"])

    def test_bad_loss_stops_before_backward(self):
        debug = self.create()
        with self.assertRaisesRegex(RuntimeError, "before_backward"):
            debug.loss(torch.tensor(float("inf")))

    def test_remote_rank_failure_propagates_at_boundary(self):
        debug = self.create()
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.all_reduce", side_effect=lambda t, op: t.fill_(1)):
            with self.assertRaisesRegex(RuntimeError, "after_update"):
                debug.boundary("after_update")
        self.assertIsNone(debug.first_bad)

    def test_loss_mode_sync_is_independent_of_full_checks(self):
        os.environ["DIFFSYNTH_ZERO3_DEBUG"] = "loss"
        os.environ["DIFFSYNTH_ZERO3_DEBUG_SYNC"] = "optimizer"
        debug = self.create()
        self.assertNotIn("partition_grads", vars(self.zero))
        with patch("torch.cuda.synchronize") as sync:
            self.assertEqual(self.zero.step(), "step")
            sync.assert_called_once_with(self.acc.device)

    def test_disabled_does_not_require_deepspeed_engine(self):
        os.environ["DIFFSYNTH_ZERO3_DEBUG"] = "off"
        self.assertIsNone(module.create_zero3_debug(None, None, None))

    def test_startup_nonfinite_metadata_is_strict_json_and_not_training_failure(self):
        config = self.acc.state.deepspeed_plugin.deepspeed_config
        config["nested"] = {"limits": [float("inf"), float("-inf"), (float("nan"), 3.5)],
                            "enabled": True, "empty": None}
        self.zero.reduce_bucket_size = float("inf")
        debug = self.create()

        def reject_constant(value):
            raise AssertionError(f"Invalid JSON constant: {value}")

        record = json.loads(debug.path.read_text().splitlines()[0], parse_constant=reject_constant)
        self.assertEqual(record["config"]["nested"]["limits"], ["inf", "-inf", ["nan", 3.5]])
        self.assertEqual(record["effective"]["reduce_bucket_size"], "inf")
        self.assertIs(record["config"]["nested"]["enabled"], True)
        self.assertIsNone(record["config"]["nested"]["empty"])
        self.assertEqual(config["nested"]["limits"][0], float("inf"))
        self.assertFalse(debug.bad)
        self.run_step(debug)

    def test_unsupported_signature_rolls_back_installed_hooks(self):
        self.zero._optimizer_step = lambda different_arg: None
        with self.assertRaisesRegex(RuntimeError, "Unsupported DeepSpeed signature"):
            module.create_zero3_debug(self.acc, self.engine, self.tmp.name)
        self.assertNotIn("partition_grads", vars(self.zero))


if __name__ == "__main__":
    unittest.main()
