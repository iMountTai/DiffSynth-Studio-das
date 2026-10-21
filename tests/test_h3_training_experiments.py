"""CPU routing/gradient tests. These do not validate HIP kernels or DMA overlap."""
import contextlib
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fa = load("h3_fa_cache_test", "diffsynth/core/attention/h3_fa_cache.py")
cpu = load("h3_cpu_pipeline_test", "diffsynth/diffusion/zero3_cpu_pipeline.py")


def attention(q, k, v, scale):
    probs = torch.softmax(torch.einsum("ihd,jhd->hij", q, k) * scale, -1)
    return torch.einsum("hij,jhd->ihd", probs, v)


class FakeFA:
    def __init__(self):
        self.forward_calls = self.backward_calls = 0

    def _wrapped_flash_attn_varlen_forward(self, q, k, v, *args, **kwargs):
        self.forward_calls += 1
        scale = args[10]
        out = attention(q, k, v, scale)
        return out, torch.zeros(1), None, torch.zeros(2, dtype=torch.int64)

    def _wrapped_flash_attn_varlen_backward(self, dout, q, k, v, out, lse, dq, dk, dv,
                                          cuq, cuk, maxq, maxk, dropout, scale, *args, **kwargs):
        self.backward_calls += 1
        with torch.enable_grad():
            inputs = [x.detach().requires_grad_() for x in (q, k, v)]
            grads = torch.autograd.grad(attention(*inputs, scale), inputs, dout)
        for dst, src in zip((dq, dk, dv), grads):
            dst.copy_(src)


class HostCache(fa.FACheckpointCache):
    """Replace only transfers; exercise real cache/autograd/checkpoint routing."""
    def save(self, signature, tensors, device):
        self.signature = signature
        self.host = tuple((t.detach().clone(), t.device) for t in tensors)

    def _prefetch(self):
        self.prefetched = tuple(t for t, _ in self.host)

    def take(self, signature):
        if signature != self.signature:
            raise RuntimeError("FA replay metadata differs")
        return self.prefetched


class FACacheTests(unittest.TestCase):
    def test_reentrant_parameter_gradients_and_interleaved_graphs(self):
        backend = FakeFA()
        cu = torch.tensor([0, 5], dtype=torch.int32)
        weight = torch.randn(4, 24, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        xs = [torch.randn(5, 4, requires_grad=True) for _ in range(2)]
        reference_xs = [x.detach().clone().requires_grad_() for x in xs]
        outputs, caches = [], []
        with patch.object(fa, "_backend", return_value=backend):
            for x in xs:
                cache = HostCache()
                caches.append(cache)
                def block(x):
                    q, k, v = (x @ weight).view(5, 3, 2, 4).unbind(1)
                    current, replay = fa._SCOPE.get()
                    current.seen += 1
                    return fa._CachedFA.apply(q, k, v, cu, 5, 0.5, current, replay).square()
                outputs.append(checkpoint(cache.wrap(block), x, use_reentrant=True))
            sum(y.sum() for y in outputs).backward()
        expected = []
        for x in reference_xs:
            q, k, v = (x @ reference_weight).view(5, 3, 2, 4).unbind(1)
            expected.append(attention(q, k, v, 0.5).square())
        sum(y.sum() for y in expected).backward()
        for actual, ref in zip(outputs, expected):
            torch.testing.assert_close(actual, ref)
        torch.testing.assert_close(weight.grad, reference_weight.grad)
        for x, ref in zip(xs, reference_xs):
            torch.testing.assert_close(x.grad, ref.grad)
        self.assertEqual(backend.forward_calls, 2)  # Not 4: recompute skipped FA.
        self.assertEqual(backend.backward_calls, 2)
        self.assertTrue(all(c.calls == 2 for c in caches))
        self.assertTrue(all(c.host is None for c in caches))
        self.assertIsNone(fa._SCOPE.get())

    def test_scope_resets_after_exception_and_rejects_missing_attention(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            with HostCache().scope():
                pass
        self.assertIsNone(fa._SCOPE.get())
        with self.assertRaisesRegex(ValueError, "test"):
            with HostCache().scope():
                raise ValueError("test")
        self.assertIsNone(fa._SCOPE.get())

    def test_replay_metadata_mismatch(self):
        cache = HostCache()
        cache.signature = (1,)
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            cache.take((2,))

    def test_default_bypass(self):
        self.assertIsNone(fa.cached_attention(None, None, None, None, None, None))
        with patch.dict(os.environ, {"DIFFSYNTH_CPU_ADAM_PIPELINE": "off"}):
            self.assertIsNone(cpu.create_cpu_adam_pipeline(None))


class Event:
    def __init__(self, log): self.log = log
    def record(self, stream): self.log.append("record")
    def synchronize(self): self.log.append("host_wait")


class Stream:
    def __init__(self, log): self.log = log
    def wait_stream(self, other): self.log.append("stream_wait")
    def synchronize(self): self.log.append("drain")


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, fail=False):
        log = []
        params = [torch.nn.Parameter(torch.ones(7)) for _ in range(3)]
        for p in params: p.grad = torch.full_like(p, 2.)
        dst = [torch.zeros(7, dtype=torch.bfloat16) for _ in params]
        z = types.SimpleNamespace(fp32_partitioned_groups_flat=params,
                                  fp16_partitioned_groups_flat=dst, loss_scale=2., clip_grad=1.)
        def scale(i, norm): params[i].grad.mul_(0.5)
        def update(i):
            if fail and i == 1: raise RuntimeError("update failed")
            if p.mode == "fused" and i in p.cpu_ids:
                params[i].grad.mul_(p.active[1])
            with torch.no_grad(): params[i].add_(params[i].grad, alpha=-0.1)
            if p.mode == "fused" and i in p.cpu_ids:
                p.active[0].copy_(params[i].detach())
        def reassign(i): dst[i].copy_(params[i].detach())
        def post(timer_names):
            log.append("post")
            for p, d in zip(params, dst): torch.testing.assert_close(d, p.to(torch.bfloat16))
        def step():
            for i in range(3):
                z.unscale_and_clip_grads(i, torch.tensor(1.))
                z._optimizer_step(i)
                z._reassign_or_swap_out_partitioned_parameters(i)
            z._post_step([])
        z.unscale_and_clip_grads, z._optimizer_step = scale, update
        z._reassign_or_swap_out_partitioned_parameters, z._post_step, z.step = reassign, post, step
        z._unflatten_partitioned_parameters = lambda i: log.append("unflatten")
        p = cpu.CPUAdamPipeline.__new__(cpu.CPUAdamPipeline)
        p.zero, p.mode, p.originals, p.pending, p.active, p.native = z, "writeback", {}, {}, None, None
        p.stats = dict(cpu_subgroups=0, h2d_bytes=0, steps=0)
        p.cpu_ids = [0, 1, 2]
        p.device = torch.device("cpu")
        p.buffers = [torch.empty(7, dtype=torch.bfloat16) for _ in range(2)]
        p.events, p.cursor, p.stream = [None, None], 0, Stream(log)
        p._install()
        return p, z, log, step

    @contextlib.contextmanager
    def transfers(self, log):
        with patch.object(torch.cuda, "current_stream", return_value=Stream(log)), \
             patch.object(torch.cuda, "stream", side_effect=lambda s: contextlib.nullcontext()), \
             patch.object(torch.cuda, "Event", side_effect=lambda: Event(log)), \
             patch.object(torch.Tensor, "record_stream", return_value=None):
            yield

    def test_ring_reuse_post_dependency_and_restoration(self):
        p, z, log, original = self.make_pipeline()
        with self.transfers(log):
            z.step()
            self.assertEqual(p.stats["cpu_subgroups"], 3)
            self.assertIn("host_wait", log)  # Third subgroup reuses first slot.
            self.assertEqual(log[log.index("post") - 1], "stream_wait")
            self.assertFalse(p.pending)
            p.close()
        self.assertIs(z.step, original)

    def test_exception_drains_dma(self):
        p, z, log, _ = self.make_pipeline(fail=True)
        with self.transfers(log):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                z.step()
            self.assertEqual(log[-1], "drain")
            self.assertFalse(p.pending)
            p.close()

    def test_fused_deferred_scale_and_unmodified_non_cpu_route(self):
        p, z, log, _ = self.make_pipeline()
        p.mode = "fused"
        p.cpu_ids = [0, 2]  # Exercise the untouched original route for subgroup 1.
        with self.transfers(log):
            z.step()
            for master in z.fp32_partitioned_groups_flat:
                torch.testing.assert_close(master.grad, torch.ones_like(master))
                torch.testing.assert_close(master, torch.full_like(master, 0.9))
            self.assertEqual(p.stats["cpu_subgroups"], 2)
            self.assertIsNone(p.active)
            self.assertFalse(p.pending)
            p.close()

    def test_active_gradient_clip_uses_original_combined_scale(self):
        p, z, log, _ = self.make_pipeline()
        p.mode = "fused"
        norm = torch.tensor(8.)
        expected_scale = float(1. / (torch.clamp(((norm / z.loss_scale) + 1e-6) / z.clip_grad,
                                                min=1.) * z.loss_scale))
        with self.transfers(log):
            z.unscale_and_clip_grads(0, norm)
            self.assertEqual(p.pending[0], expected_scale)
            torch.testing.assert_close(z.fp32_partitioned_groups_flat[0].grad, torch.full((7,), 2.))
            z._optimizer_step(0)
            z._reassign_or_swap_out_partitioned_parameters(0)
            torch.testing.assert_close(z.fp32_partitioned_groups_flat[0].grad,
                                       torch.full((7,), 2. * expected_scale))
            p.close()


if __name__ == "__main__":
    unittest.main()
