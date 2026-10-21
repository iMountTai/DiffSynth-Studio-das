# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Remote preflight: native CPUAdam and real FA replay. Does not run training."""
import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_check():
    from deepspeed.ops.op_builder import CPUAdamBuilder
    module = load("h3_cpu_pipeline_preflight", "diffsynth/diffusion/zero3_cpu_pipeline.py")
    fused = module.load_fused_cpu_adam()
    original = CPUAdamBuilder().load()
    torch.manual_seed(7)
    cases = 0
    for size in (17, 1031, 100003):  # Exercise SIMD bodies and all scalar tails.
        for adamw in (False, True):
            for bias_correction in (False, True):
                baseline = [torch.randn(size), torch.zeros(size), torch.zeros(size)]
                candidate = [t.clone() for t in baseline]
                output = torch.empty(size, dtype=torch.bfloat16)
                for backend in (original, fused):
                    backend.create_adam(991, 1e-4, 0.9, 0.999, 1e-8, 0.01, adamw, False)
                try:
                    # Include a nonconsecutive step (checkpoint resume behavior).
                    for step in (1, 2, 3, 17, 18):
                        scale = 0.137 if step % 2 else 1.0
                        grad = torch.randn(size)
                        native_grad, fused_grad = grad.mul(scale), grad.clone()
                        beta1 = 0.85 if step >= 17 else 0.9
                        args = (991, step, 1e-4, beta1, 0.999, 1e-8, 0.01, bias_correction)
                        original.adam_update(*args, baseline[0], native_grad, baseline[1], baseline[2])
                        fused.adam_update_fused(*args, candidate[0], fused_grad,
                                                candidate[1], candidate[2], output, scale)
                        torch.testing.assert_close(fused_grad, native_grad, rtol=0, atol=0)
                        for a, b in zip(candidate, baseline):
                            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
                        torch.testing.assert_close(output, candidate[0].to(torch.bfloat16), rtol=0, atol=0)
                        cases += 1
                finally:
                    original.destroy_adam(991)
                    fused.destroy_adam(991)
    print(f"PASS native CPUAdam: {cases} updates against installed DeepSpeed CPUAdam")


def fa_check(mode="cpu"):
    import flash_attn
    if not torch.cuda.is_available():
        raise RuntimeError("FA cache preflight requires an HCU/CUDA device")
    module = load("h3_fa_cache_preflight", "diffsynth/core/attention/h3_fa_cache.py")
    interface = module._backend()
    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.current_device())
    cu = torch.tensor([0, 17, 64], dtype=torch.int32, device=device)
    for dtype in (torch.bfloat16, torch.float16):
        native_inputs = [torch.randn(64, 4, 128, dtype=dtype, device=device, requires_grad=True) for _ in range(3)]
        cached_inputs = [t.detach().clone().requires_grad_() for t in native_inputs]
        dout = torch.randn_like(native_inputs[0])
        expected = flash_attn.flash_attn_varlen_func(
            *native_inputs, cu, cu, 47, 47, softmax_scale=128 ** -0.5, causal=False)
        expected = expected[0] if isinstance(expected, tuple) else expected
        expected.backward(dout)
        cache = module.FACheckpointCache(mode=mode)
        def block(q, k, v):
            return module.cached_attention(q * 1, k * 1, v * 1, cu, 47, 128 ** -0.5)
        with patch.object(interface, "_wrapped_flash_attn_varlen_forward",
                          wraps=interface._wrapped_flash_attn_varlen_forward) as fwd, \
             patch.object(interface, "_wrapped_flash_attn_varlen_backward",
                          wraps=interface._wrapped_flash_attn_varlen_backward) as bwd:
            actual = checkpoint(cache.wrap(block), *cached_inputs, use_reentrant=True)
            actual.backward(dout)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(cached_inputs, native_inputs):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-2, atol=2e-2)
        assert fwd.call_count == 1 and bwd.call_count == 1, (fwd.call_count, bwd.call_count)
        assert cache.calls == 2
        assert (cache.cpu_bytes > 0) == (mode == "cpu")
        assert (cache.gpu_bytes > 0) == (mode == "gpu")
        print(f"PASS FA {dtype} mode={mode}: forward=1, backward=1, replay=1, "
              f"cpu_bytes={cache.cpu_bytes} gpu_bytes={cache.gpu_bytes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--fa", action="store_true")
    parser.add_argument("--fa-gpu", action="store_true")
    args = parser.parse_args()
    if not args.native and not args.fa and not args.fa_gpu:
        parser.error("Choose --native, --fa, and/or --fa-gpu")
    if args.native:
        native_check()
    if args.fa:
        fa_check()
    if args.fa_gpu:
        fa_check(mode="gpu")
