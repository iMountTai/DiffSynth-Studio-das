# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, intrusive ZeRO-3 diagnostics; never enabled by normal training."""

import functools
import inspect
import json
import math
import os
import time
from pathlib import Path

import torch


def _json_safe(value):
    """Keep nonfinite metadata readable without emitting invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class Zero3Debug:
    def __init__(self, accelerator, engine, output_path):
        self.mode = os.environ.get("DIFFSYNTH_ZERO3_DEBUG", "")
        self.sync = os.environ.get("DIFFSYNTH_ZERO3_DEBUG_SYNC", "none")
        if self.mode not in ("loss", "full"):
            raise ValueError("DIFFSYNTH_ZERO3_DEBUG must be loss or full")
        if self.sync not in ("none", "fetch", "reduce", "optimizer"):
            raise ValueError("DIFFSYNTH_ZERO3_DEBUG_SYNC must be none, fetch, reduce or optimizer")
        self.accelerator = accelerator
        self.engine = engine
        self.zero = getattr(engine, "optimizer", None)
        if not hasattr(self.zero, "fp32_partitioned_groups_flat"):
            raise RuntimeError("ZeRO-3 diagnostics require a prepared DeepSpeed ZeRO-3 engine")
        if getattr(self.zero, "swap_optimizer", False):
            raise RuntimeError("ZeRO-3 diagnostics do not support NVMe optimizer swapping")
        self.step = -1
        self.epoch = -1
        self.bad = False
        self.first_bad = None
        self.sequence = 0
        self.patches = []
        self.names = {id(p): n for n, p in engine.named_parameters()}
        directory = Path(os.environ.get("DIFFSYNTH_ZERO3_DEBUG_DIR", str(Path(output_path) / "zero3_debug")))
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank{accelerator.process_index}-pid{os.getpid()}-{time.time_ns()}.jsonl"
        self.file = self.path.open("w", encoding="utf-8", buffering=1)
        import deepspeed
        self.write("startup", mode=self.mode, sync=self.sync, torch=torch.__version__,
                   hip=torch.version.hip, deepspeed=deepspeed.__version__,
                   zero_source=inspect.getfile(type(self.zero)),
                   config=accelerator.state.deepspeed_plugin.deepspeed_config,
                   effective={k: getattr(self.zero, k, None) for k in (
                       "overlap_comm", "offload_optimizer", "partial_offload", "reduce_scatter",
                       "contiguous_gradients", "reduce_bucket_size", "communication_data_type",
                       "gradient_accumulation_dtype")},
                   env={k: os.environ.get(k) for k in (
                       "OMP_NUM_THREADS", "HIP_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING",
                       "DIFFSYNTH_ALLOW_HIP_ZERO3_OVERLAP")},
                   warning="Tensor checks synchronize and can hide races; no cross-rank tensor equality is implied.")
        try:
            self.install()
        except Exception:
            self.close()
            raise
        print(f"[zero3-debug rank={accelerator.process_index}] mode={self.mode} sync={self.sync} log={self.path}", flush=True)

    def write(self, stage, **fields):
        record = dict(stage=stage, rank=self.accelerator.process_index, epoch=self.epoch,
                      step=self.step, sequence=self.sequence,
                      microstep=getattr(self.zero, "micro_step_id", None), **fields)
        self.sequence += 1
        # default=str does not handle builtin floats: JSON encodes them directly.
        # DeepSpeed metadata may legitimately contain infinity as a limit.
        self.file.write(json.dumps(_json_safe(record), default=str, allow_nan=False) + "\n")

    def check(self, stage, tensor, **fields):
        if tensor is None:
            self.write(stage, missing=True, **fields)
            return
        with torch.no_grad():
            flat = tensor.detach().reshape(-1)
            # Bound temporary allocations for very large optimizer subgroups.
            counts = torch.zeros((), dtype=torch.int64, device=flat.device)
            maximum = torch.zeros((), dtype=torch.float64, device=flat.device)
            for chunk in flat.split(1048576):
                if chunk.numel():
                    counts += (~torch.isfinite(chunk)).sum()
                    maximum = torch.maximum(maximum, chunk.abs().amax().double())
            nonfinite = int(counts.item())
            absmax = float(maximum.item())
        if nonfinite and self.first_bad is None:
            self.first_bad = dict(stage=stage, **fields)
            print(f"[zero3-debug rank={self.accelerator.process_index}] FIRST NONFINITE: {self.first_bad}", flush=True)
        self.bad |= nonfinite > 0
        self.write(stage, shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                   numel=tensor.numel(), nonfinite=nonfinite,
                   absmax=absmax if math.isfinite(absmax) else None, **fields)

    def patch(self, owner, name, before=None, after=None):
        original = getattr(owner, name, None)
        if not callable(original):
            raise RuntimeError(f"Unsupported DeepSpeed version: missing {name}")
        signature = inspect.signature(original)
        required = {
            "reduce_independent_p_g_buckets_and_remove_grads": ("param",),
            "partition_grads": ("params_to_release", "grad_partitions"),
            "_prepare_sub_group": ("sub_group_id",),
            "_optimizer_step": ("sub_group_id",),
            "unscale_and_clip_grads": ("sub_group_id", "total_norm"),
            "_reassign_or_swap_out_partitioned_parameters": ("sub_group_id",),
        }.get(name, ())
        if any(key not in signature.parameters for key in required):
            raise RuntimeError(f"Unsupported DeepSpeed signature: {name}{signature}")

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if before:
                before(bound.arguments)
            result = original(*args, **kwargs)
            if after:
                after(bound.arguments, result)
            return result

        self.patches.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, wrapped)
        self.write("hook", method=name, signature=str(signature))

    def synchronize(self, stage):
        torch.cuda.synchronize(self.accelerator.device)
        self.write("synchronize", boundary=stage)

    def param_fields(self, p):
        return dict(parameter=self.names.get(id(p), "<unnamed>"), ds_id=getattr(p, "ds_id", None))

    def group(self, bound):
        # Fail explicitly rather than silently inspecting the wrong subgroup.
        return int(bound["sub_group_id"])

    def master(self, group, stage, gradients=False, states=False):
        p = self.zero.fp32_partitioned_groups_flat[group]
        self.check(stage + ".master", p, subgroup=group)
        if gradients:
            self.check(stage + ".grad", p.grad, subgroup=group)
        if states:
            optimizers = [self.zero.optimizer, getattr(self.zero, "backup_optimizer", None)]
            for optimizer in optimizers:
                if optimizer is None:
                    continue
                for key, value in optimizer.state.get(p, {}).items():
                    if torch.is_tensor(value):
                        self.check(stage + ".state", value, subgroup=group, state=key)

    def install(self):
        if self.sync == "fetch":
            coordinator = self.zero._get_param_coordinator()
            self.patch(coordinator, "fetch_sub_module",
                       after=lambda b, r: self.synchronize("fetch.after"))
        if self.sync == "optimizer":
            self.patch(self.zero, "step", before=lambda b: self.synchronize("optimizer.entry"))
        if self.sync == "reduce":
            names = [n for n in dir(self.zero) if n.endswith("__avg_scatter_grads") or n.endswith("__avg_scatter_contiguous_grads")]
            if not names:
                raise RuntimeError("Unsupported DeepSpeed version: no gradient reduction methods")
            for name in names:
                self.patch(self.zero, name,
                           before=lambda b: self.synchronize("reduce.before"),
                           after=lambda b, r: self.synchronize("reduce.after"))
        if self.mode != "full":
            return

        def grad_before(bound):
            p = bound["param"]
            self.check("grad.produced", p.grad, **self.param_fields(p))

        self.patch(self.zero, "reduce_independent_p_g_buckets_and_remove_grads", before=grad_before)

        def reduced(bound):
            for p, grad in zip(bound["params_to_release"], bound["grad_partitions"]):
                self.check("grad.reduced", grad, **self.param_fields(p))

        self.patch(self.zero, "partition_grads", before=reduced)

        def prepared(bound, result):
            group = self.group(bound)
            self.master(group, "optimizer.pre_clip", gradients=True)

        self.patch(self.zero, "_prepare_sub_group", after=prepared)

        def clip_inputs(bound):
            group = self.group(bound)
            for name, value in (("total_norm", bound["total_norm"]),
                                ("loss_scale", self.zero.loss_scale), ("clip_grad", self.zero.clip_grad)):
                self.check("optimizer." + name, value if torch.is_tensor(value) else torch.tensor(value), subgroup=group)

        self.patch(self.zero, "unscale_and_clip_grads", before=clip_inputs)
        self.patch(self.zero, "_optimizer_step",
                   before=lambda b: self.master(self.group(b), "optimizer.before", gradients=True, states=True),
                   after=lambda b, r: self.master(self.group(b), "optimizer.after", states=True))

        def reassigned(bound, result):
            group = self.group(bound)
            for p in self.zero.fp16_groups[group]:
                self.check("parameter.reassigned", p.ds_tensor, subgroup=group, **self.param_fields(p))

        self.patch(self.zero, "_reassign_or_swap_out_partitioned_parameters", after=reassigned)

    def begin(self, epoch, step):
        self.epoch, self.step = epoch, step
        self.write("step.begin")

    def loss(self, loss):
        self.check("loss", loss)
        value = float(loss.detach().item())
        self.write("loss.value", value=value if math.isfinite(value) else None)
        self.boundary("before_backward")

    def boundary(self, stage):
        # All ranks call only at the same outer training-loop boundaries.
        # Never insert collectives inside per-parameter or subgroup hooks.
        flag = torch.tensor(int(self.bad), device=self.accelerator.device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        any_bad = bool(flag.item())
        self.write("boundary", boundary=stage, any_rank_nonfinite=any_bad, first_bad=self.first_bad)
        if any_bad:
            raise RuntimeError(f"ZeRO-3 debug detected nonfinite values at {stage}; collect all rank logs from {self.path.parent}")

    def close(self):
        for owner, name, had_instance_attr, previous in reversed(self.patches):
            if had_instance_attr:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.patches.clear()
        self.file.close()


def create_zero3_debug(accelerator, model, output_path):
    if os.environ.get("DIFFSYNTH_ZERO3_DEBUG", "") in ("", "0", "off"):
        if os.environ.get("DIFFSYNTH_ZERO3_DEBUG_SYNC", "none") != "none":
            raise ValueError("Set DIFFSYNTH_ZERO3_DEBUG=loss to enable synchronization diagnostics")
        return None
    return Zero3Debug(accelerator, model, output_path)
