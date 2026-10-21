# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Experimental FA2 O/LSE cache scoped to one reentrant checkpoint.

Q/K/V are recomputed. No detached output is substituted for an autograd node:
replay constructs a new node using the original FA backward implementation.
Only H3's BF16/FP16, non-causal, dropout=0 packed self attention is supported.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import lru_cache
import importlib
import os

import torch
from torch.autograd.function import once_differentiable


_SCOPE = ContextVar("h3_fa_cache", default=None)
_STREAMS = {}


@lru_cache(None)
def _backend():
    interface = importlib.import_module("flash_attn.flash_attn_interface")
    for name in ("_wrapped_flash_attn_varlen_forward", "_wrapped_flash_attn_varlen_backward"):
        if not callable(getattr(interface, name, None)):
            raise RuntimeError(f"FA cache needs flash-attention-cutlass interface {name}")
    return interface


def _stream(device):
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device=device)
    return _STREAMS[device]


def _signature(q, k, v, cu, max_seqlen, scale):
    return (tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in (q, k, v)),
            cu.data_ptr(), cu._version, tuple(cu.shape), int(max_seqlen), scale)


class FACheckpointCache:
    def __init__(self, mode="cpu"):
        if mode not in ("cpu", "gpu"):
            raise ValueError("FA cache mode must be cpu or gpu")
        self.mode = mode
        self.calls = 0
        self.signature = None
        self.host = None
        self.gpu = None
        self.prefetched = None
        self.event = None
        self.device = None
        self.cpu_bytes = 0
        self.gpu_bytes = 0

    def _prefetch(self):
        stream = _stream(self.device)
        # Ensure all producer copies have completed before any host-buffer read.
        stream.wait_event(self.event)
        with torch.cuda.stream(stream):
            self.prefetched = tuple(t.to(device, non_blocking=True) if device is not None and device.type == "cuda" else t
                                    for t, device in self.host)
            self.event = torch.cuda.Event()
            self.event.record(stream)

    @contextmanager
    def scope(self):
        replay = self.calls > 0
        self.calls += 1
        if replay:
            if (self.host if self.mode == "cpu" else self.gpu) is None:
                raise RuntimeError("FA checkpoint has no cached result; repeated backward/retain_graph is unsupported")
            if self.mode == "cpu":
                self._prefetch()  # Start H2D before the block's QKV/Norm/RoPE recompute.
            else:
                self.prefetched = self.gpu
        token = _SCOPE.set((self, replay))
        self.seen = 0
        try:
            yield
            if self.seen != 1:
                raise RuntimeError("FA cache expects exactly one FA2 call per selected DiT block")
        finally:
            _SCOPE.reset(token)
            self.prefetched = None
            # All copies complete before host storage can be released on error
            # or checkpoint graph destruction. No tensor depends on GC timing.
            if self.mode == "cpu" and self.event is not None:
                self.event.synchronize()
            if replay:
                # Recomputed autograd owns the cached values now.
                self.host = None
                self.gpu = None
        if replay and os.environ.get("DIFFSYNTH_FA_CACHE_LOG") == "1":
            print(f"H3 FA cache rank={os.environ.get('RANK', '0')} replay_hit=1 "
                  f"mode={self.mode} cpu_bytes={self.cpu_bytes} gpu_bytes={self.gpu_bytes} "
                  "fa_forward_skipped=1", flush=True)

    def wrap(self, function):
        def run(*args, **kwargs):
            with self.scope():
                return function(*args, **kwargs)
        return run

    def save(self, signature, tensors, device):
        if self.host is not None or self.gpu is not None:
            raise RuntimeError("Cannot record the same checkpoint twice")
        self.signature, self.device = signature, device
        if self.mode == "gpu":
            # The checkpoint's original forward runs under no_grad. Retain the
            # exact FA output/LSE/RNG storage until replay, without a DMA copy.
            self.gpu = tuple(t.detach() if t is not None else None for t in tensors)
            self.gpu_bytes = sum(t.numel() * t.element_size() for t in self.gpu
                                 if t is not None and t.device.type == "cuda")
            return
        stream = _stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        host = []
        with torch.cuda.stream(stream):
            for tensor in tensors:
                if tensor is None:
                    host.append((None, None))
                    continue
                if tensor.device.type == "cuda":
                    cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    cpu.copy_(tensor.detach(), non_blocking=True)
                    tensor.record_stream(stream)
                else:
                    cpu = tensor.detach().clone()
                host.append((cpu, tensor.device))
            self.event = torch.cuda.Event()
            self.event.record(stream)
        self.host = tuple(host)
        self.cpu_bytes = sum(t.numel() * t.element_size() for t, _ in host if t is not None)

    def take(self, signature):
        if signature != self.signature:
            raise RuntimeError("FA replay metadata differs from original forward")
        if self.mode == "cpu":
            torch.cuda.current_stream(self.device).wait_event(self.event)
        tensors = self.prefetched
        if self.mode == "cpu":
            for tensor in tensors:
                if tensor is not None and tensor.device.type == "cuda":
                    tensor.record_stream(torch.cuda.current_stream(self.device))
        return tensors


class _CachedFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, max_seqlen, scale, cache, replay):
        signature = _signature(q, k, v, cu, max_seqlen, scale)
        if replay:
            out, lse, rng = cache.take(signature)
        else:
            out, lse, _, rng = _backend()._wrapped_flash_attn_varlen_forward(
                q, k, v, None, cu, cu, None, None, None, None,
                max_seqlen, max_seqlen, 0.0, scale, False, False, -1, -1,
                softcap=0.0, return_softmax=False,
            )
            cache.save(signature, (out, lse, rng), q.device)
        ctx.save_for_backward(q, k, v, out, lse, cu, rng)
        ctx.max_seqlen, ctx.scale = max_seqlen, scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, cu, rng = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _backend()._wrapped_flash_attn_varlen_backward(
            dout, q, k, v, out, lse, dq, dk, dv, cu, cu,
            ctx.max_seqlen, ctx.max_seqlen, 0.0, ctx.scale, False, -1, -1,
            0.0, None, False, rng_state=rng,
        )
        return dq, dk, dv, None, None, None, None, None


def cached_attention(q, k, v, cu_seqlens, max_seqlen, scale):
    scope = _SCOPE.get()
    if scope is None:
        return None
    cache, replay = scope
    cache.seen += 1
    if cache.seen != 1:
        raise RuntimeError("Multiple attention calls in one FA cache scope")
    if (q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16)
            or q.ndim != 3 or q.shape[-1] % 8 or q.shape != k.shape or q.shape != v.shape
            or any(t.dtype != q.dtype or t.device != q.device or t.stride(-1) != 1 for t in (k, v))
            or q.stride(-1) != 1 or cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != q.device or not cu_seqlens.is_contiguous()):
        raise RuntimeError("FA cache requires H3 packed FP16/BF16 self attention with aligned head dim")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    return _CachedFA.apply(q, k, v, cu_seqlens, max_seqlen, scale, cache, replay)


def configure_fa_cache(dit, checkpointing, checkpoint_offload):
    cpu_layers = int(os.environ.get("DIFFSYNTH_FA_CPU_CACHE_LAYERS", "0"))
    gpu_layers = int(os.environ.get("DIFFSYNTH_FA_GPU_CACHE_LAYERS", "0"))
    if cpu_layers and gpu_layers:
        raise ValueError("Enable CPU or GPU FA cache, not both")
    mode, layers = ("cpu", cpu_layers) if cpu_layers else ("gpu", gpu_layers)
    if not layers:
        return
    if not checkpointing or checkpoint_offload:
        raise RuntimeError("FA cache requires block checkpointing and no separate checkpoint offload")
    if not 0 < layers <= len(dit.blocks):
        raise ValueError("FA cache layers must be between 1 and the number of DiT blocks")
    from .attention import ATTENTION_IMPLEMENTATION
    if ATTENTION_IMPLEMENTATION != "flash_attention_2":
        raise RuntimeError("FA cache requires flash_attention_2")
    _backend()
    for block in dit.blocks[:layers]:
        block._h3_fa_cache_mode = mode
    print(f"H3 FA {mode.upper()} cache enabled on first {layers} DiT blocks; "
          "reentrant checkpoint required", flush=True)


def configure_fa_cpu_cache(dit, checkpointing, checkpoint_offload):
    """Keep the existing preflight and caller API available."""
    return configure_fa_cache(dit, checkpointing, checkpoint_offload)
