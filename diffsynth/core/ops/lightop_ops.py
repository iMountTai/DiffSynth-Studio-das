# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Automatically dispatch supported HCU tensors to optional lightop kernels."""

import importlib
import warnings
from collections import Counter
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


_CALLS = Counter()
_FALLBACKS = Counter()
_IMPORT_ERROR = None


@lru_cache(maxsize=1)
def _load_op():
    global _IMPORT_ERROR
    try:
        return importlib.import_module("lightop").op
    except ModuleNotFoundError as exc:
        if exc.name != "lightop":
            raise
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def get_lightop_info():
    """Inspect dispatch without importing lightop or initializing a device.

    Counts describe Python dispatches, not compiled/graph replay kernel counts.
    """
    return {
        "policy": "auto",
        "import_attempted": bool(_load_op.cache_info().currsize),
        "import_error": _IMPORT_ERROR,
        "kernel_calls": dict(_CALLS),
        "fallbacks": dict(_FALLBACKS),
    }


def _fallback(name, reason):
    _FALLBACKS[f"{name}:{reason}"] += 1


def _candidate(name, x):
    # Unregistered extension calls are deliberately kept out of compiled graphs.
    if torch.compiler.is_compiling():
        return False
    if not torch.version.hip or not x.is_cuda:
        _fallback(name, "requires HIP device")
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or x.numel() == 0:
        _fallback(name, "dtype or empty tensor")
        return False
    return True


def _operator(name, *symbols):
    op = _load_op()
    if op is None:
        _fallback(name, "import failed")
        return None
    if not all(callable(getattr(op, symbol, None)) for symbol in symbols):
        reason = "missing " + ", ".join(symbols)
        if not _FALLBACKS[f"{name}:{reason}"]:
            warnings.warn(f"lightop {name}: {reason}; kernel unavailable", RuntimeWarning, stacklevel=2)
        _fallback(name, reason)
        return None
    return op


def _aligned(x, elements=8):
    # Row starts and storage offsets must satisfy the kernel's vector alignment.
    return x.storage_offset() % elements == 0


@lru_cache(maxsize=16)
def _zero_bias(width, dtype, device):
    # Normal tensors can be saved for backward even after an inference warmup.
    with torch.inference_mode(False):
        return torch.zeros(width, dtype=dtype, device=device)


class _BiasSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, op):
        ctx.op = op
        ctx.save_for_backward(x, bias)
        with torch.cuda.device(x.device):
            return op.FusedBiasSwiGLU_forward(x, bias).view(*x.shape[:-1], x.shape[-1] // 2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, bias = ctx.saved_tensors
        # contiguous() may retain a misaligned storage offset on a slice.
        grad = grad.contiguous()
        if not _aligned(grad):
            grad = grad.clone()
        with torch.cuda.device(x.device):
            gx, gb = ctx.op.FusedBiasSwiGLU_backward(grad, x, bias)
        return gx.view_as(x), gb if ctx.needs_input_grad[1] else None, None


def fuse_bias_swiglu(x, bias=None):
    if _candidate("SWIGLU", x):
        valid = (
            x.ndim >= 2 and x.shape[-1] % 16 == 0
            and x.numel() < 2**31 and x.is_contiguous() and _aligned(x)
            and (bias is None or (
                bias.shape == (x.shape[-1],) and bias.dtype == x.dtype
                and bias.device == x.device and bias.is_contiguous() and _aligned(bias)
            ))
        )
        if valid:
            needs_grad = torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad))
            if bias is None and not needs_grad:
                op = _operator("SWIGLU", "fuse_silu_and_mul")
                if op is not None:
                    out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
                    with torch.cuda.device(x.device):
                        op.fuse_silu_and_mul(x, out)
                    _CALLS["SWIGLU:no_bias"] += 1
                    return out
            # A build may expose only the bias kernel. It also handles the
            # no-bias case using a zero vector, including inference-only builds.
            symbols = ("FusedBiasSwiGLU_forward", "FusedBiasSwiGLU_backward") if needs_grad else ("FusedBiasSwiGLU_forward",)
            op = _operator("SWIGLU", *symbols)
            if op is not None:
                b = bias if bias is not None else _zero_bias(x.shape[-1], x.dtype, x.device)
                result = _BiasSwiGLU.apply(x, b, op)
                _CALLS["SWIGLU:bias_autograd"] += 1
                return result
        else:
            _fallback("SWIGLU", "shape, layout or bias")
    hidden = x if bias is None else x + bias
    gate, up = hidden.chunk(2, dim=-1)
    return F.silu(gate) * up


def _native_rmsnorm(x, weight, eps):
    # Match nn.RMSNorm instead of substituting a hand-written GPU graph.
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    # Match models.general_modules.RMSNorm on older PyTorch versions.
    variance = x.to(torch.float32).square().mean(dim=-1, keepdim=True)
    normalized = (x * torch.rsqrt(variance + eps)).to(x.dtype)
    return normalized if weight is None else normalized * weight


def can_use_rmsnorm_add(x, residual, weight):
    """Check support before choosing the model's residual/norm computation path."""
    if not _candidate("RMSNORM_ADD", x):
        return False
    valid = (
        x.ndim >= 2 and x.shape == residual.shape
        and x.dtype == residual.dtype and x.device == residual.device
        and 64 <= x.shape[-1] <= 16384 and x.shape[-1] % 16 == 0
        and x.numel() < 2**30 and x.is_contiguous() and residual.is_contiguous()
        and _aligned(x, 16) and _aligned(residual, 16)
        and weight is not None and weight.shape == (x.shape[-1],)
        and weight.dtype == x.dtype and weight.device == x.device
        and weight.is_contiguous() and _aligned(weight, 16)
    )
    if not valid:
        _fallback("RMSNORM_ADD", "shape, layout or weight")
        return False
    return _operator("RMSNORM_ADD", "rn_add_forward_autograd") is not None


def rmsnorm_add(x, residual, weight=None, eps=1e-5, *, return_residual=False):
    """RMSNorm((x + residual).to(residual.dtype)), optionally returning the sum.

    Never mutates inputs. The native sum is still needed by the residual branch;
    lightop's public add/norm API returns only the normalized output.
    """
    summed = None
    if return_residual:
        summed = (residual + x).to(residual.dtype)
    if can_use_rmsnorm_add(x, residual, weight):
        op = _load_op()
        training = torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight))
        with torch.cuda.device(x.device):
            result = op.rn_add_forward_autograd(x, residual, weight, eps, training, False)
        _CALLS["RMSNORM_ADD"] += 1
        return (result, summed) if return_residual else result
    if summed is None:
        summed = (residual + x).to(residual.dtype)
    result = _native_rmsnorm(summed, weight, eps)
    return (result, summed) if return_residual else result


class _Cat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, op):
        ctx.width = a.shape[-1]
        out = a.new_empty((*a.shape[:-1], a.shape[-1] + b.shape[-1]))
        # Flatten rows as a VIEW into decode layout: prefill assumes broadcast B.
        # mode 6: 16 A rows and 8 B rows per block, four warps.
        with torch.cuda.device(a.device):
            op.ds_concat(a.view(1, -1, a.shape[-1]), b.view(1, -1, b.shape[-1]), out.view(1, -1, out.shape[-1]), 6)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad[..., :ctx.width], grad[..., ctx.width:], None


def da_cat(a, b, dim=-1):
    """Feature concat with a conservative contract for the local ds_concat kernel."""
    if _candidate("CONCAT", a):
        # Widths must tile a 64-lane warp in 8-element vectors. For mode 6,
        # A needs >=16 threads per row and B needs >=8 to prevent block overlap.
        valid = (
            a.ndim == b.ndim == 3 and dim in (-1, 2)
            and a.shape[:-1] == b.shape[:-1] and a.device == b.device and a.dtype == b.dtype
            and a.shape[-1] in (128, 256, 512) and b.shape[-1] in (64, 128, 256, 512)
            and a.shape[0] * a.shape[1] % 16 == 0
            and (a.shape[0] * a.shape[1] // 16 + a.shape[0] * a.shape[1] // 8) <= 65535
            and a.stride(-1) == b.stride(-1) == 1
            and a.stride(0) == a.shape[1] * a.stride(1)
            and b.stride(0) == b.shape[1] * b.stride(1)
            and a.stride(1) % 8 == b.stride(1) % 8 == 0
            and _aligned(a) and _aligned(b)
            and max(a.numel(), b.numel(), *a.stride(), *b.stride()) < 2**31
            and all(sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) < 2**31 for t in (a, b))
        )
        if valid:
            op = _operator("CONCAT", "ds_concat")
            if op is not None:
                result = _Cat.apply(a, b, op)
                _CALLS["CONCAT"] += 1
                return result
        else:
            _fallback("CONCAT", "unsupported ds_concat layout or tiling")
    return torch.cat((a, b), dim=dim)


ds_cat = da_cat
