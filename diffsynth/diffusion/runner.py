# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
import contextlib
import importlib
import json
import os
import time
import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def _set_training_rng_context(accelerator, model, seed, epoch_id, step_id):
    """Publish deterministic per-step RNG inputs to the underlying pipeline."""
    if seed is None:
        return
    unwrapped_model = accelerator.unwrap_model(model)
    pipe = getattr(unwrapped_model, "pipe", None)
    if pipe is not None:
        pipe._diffsynth_training_rng_context = {
            "seed": int(seed),
            "epoch": int(epoch_id),
            "step": int(step_id),
            "rank": int(accelerator.process_index),
        }


@contextlib.contextmanager
def maybe_enable_torch_profiler(accelerator, args):
    if args is None or not getattr(args, "enable_torch_profiler", False):
        yield None
        return

    profile_freq = getattr(args, "profiler_freq", None)
    warmup = getattr(args, "profiler_warmup", 1)
    active = getattr(args, "profiler_active", 1)
    repeat = getattr(args, "profiler_repeat", 1)
    if profile_freq is None:
        wait = getattr(args, "profiler_wait", 0)
    else:
        wait = profile_freq - warmup - active
        if profile_freq < 1 or wait < 0:
            raise ValueError("Profiler frequency must be >= warmup + active and at least 1.")
    if wait < 0 or warmup < 0 or active < 1 or repeat < 1:
        raise ValueError("Profiler schedule requires wait/warmup >= 0 and active/repeat >= 1.")

    ranks_arg = getattr(args, "profiler_target_ranks", "0").strip().lower()
    if ranks_arg == "all":
        target_ranks = set(range(accelerator.num_processes))
    else:
        try:
            target_ranks = {int(rank.strip()) for rank in ranks_arg.split(",") if rank.strip()}
        except ValueError as e:
            raise ValueError("--profiler_target_ranks must be comma-separated integers or `all`.") from e
        invalid_ranks = {rank for rank in target_ranks if rank < 0 or rank >= accelerator.num_processes}
        if not target_ranks or invalid_ranks:
            raise ValueError(f"Invalid profiler target ranks: {sorted(invalid_ranks or target_ranks)}")

    trace_root = getattr(args, "profiler_output_path", None) or os.path.join(args.output_path, "torch_trace")
    os.makedirs(trace_root, exist_ok=True)
    rank = accelerator.process_index
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def trace_handler(prof):
        if rank not in target_ranks:
            return
        trace_dir = os.path.join(trace_root, f"iteration_{prof.step_num}")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"rank{rank}_trace.json.gz")
        begin = time.monotonic()
        prof.export_chrome_trace(trace_path)
        print(f"Rank {rank} exported profiler trace to `{trace_path}` in {time.monotonic() - begin:.2f}s.")

    if accelerator.is_main_process:
        schedule_summary = (
            f"freq={profile_freq}, wait={wait}" if profile_freq is not None else f"wait={wait}"
        )
        print(
            f"PyTorch profiler enabled: {schedule_summary}, warmup={warmup}, "
            f"active={active}, repeat={repeat}, traces=`{trace_root}`."
        )

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=trace_handler,
        record_shapes=getattr(args, "profiler_record_shapes", False),
        profile_memory=getattr(args, "profiler_profile_memory", False),
        with_stack=getattr(args, "profiler_with_stack", False),
        with_modules=getattr(args, "profiler_with_modules", False),
    ) as profiler:
        yield profiler


def _configure_deepspeed_zero3_lora_single_param_all_reduce(accelerator, enabled=False):
    if not enabled:
        return

    deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if deepspeed_plugin is None:
        raise RuntimeError("--deepspeed_zero3_lora_single_param_all_reduce requires DeepSpeed.")

    zero_config = deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
    zero_stage = zero_config.get("stage", getattr(deepspeed_plugin, "zero_stage", None))
    if zero_stage != 3:
        raise RuntimeError("--deepspeed_zero3_lora_single_param_all_reduce requires ZeRO stage 3.")

    if torch.version.hip is None:
        return

    if os.environ.get("DIFFSYNTH_ALLOW_HIP_ZERO3_OVERLAP", "0") in ("1", "true", "True"):
        if accelerator.is_main_process:
            print(
                "DIFFSYNTH_ALLOW_HIP_ZERO3_OVERLAP=1 detected: retaining overlap_comm=True "
                "on HIP for debugging/profiling."
            )
        return

    # HCU/HIP deadlocks when ZeRO-3 parameter fetches run on DeepSpeed's
    # private all-gather stream. Keep the proven default-stream path.
    zero_config["overlap_comm"] = False
    if accelerator.is_main_process:
        print(
            "Configured DeepSpeed ZeRO-3 overlap_comm=False for HIP single-parameter "
            "LoRA all-reduce fetches; parameter fetches use the default device stream."
        )


def _patch_deepspeed_zero3_lora_single_param_all_reduce(accelerator, model, enabled=False):
    if not enabled:
        return

    deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if deepspeed_plugin is None:
        raise RuntimeError("--deepspeed_zero3_lora_single_param_all_reduce requires DeepSpeed.")

    zero_config = deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
    zero_stage = zero_config.get("stage", getattr(deepspeed_plugin, "zero_stage", None))
    if zero_stage != 3:
        raise RuntimeError("--deepspeed_zero3_lora_single_param_all_reduce requires ZeRO stage 3.")

    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.zero.partition_parameters import AllReduceCoalescedHandle, ZeroParamStatus
    from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus

    fetch_work_attr = "_diffsynth_zero3_all_reduce_fetch_work"
    device_accelerator = get_accelerator()
    allow_hip_overlap = os.environ.get("DIFFSYNTH_ALLOW_HIP_ZERO3_OVERLAP", "0") in ("1", "true", "True")
    use_independent_fetch_stream = (
        (torch.version.hip is None or allow_hip_overlap)
        and zero_config.get("overlap_comm") is True
        and not device_accelerator.is_synchronized_device()
    )

    if use_independent_fetch_stream:
        zero_optimizer = getattr(model, "optimizer", None)
        get_param_coordinator = getattr(zero_optimizer, "_get_param_coordinator", None)
        if get_param_coordinator is None:
            raise RuntimeError(
                "Could not locate the DeepSpeed ZeRO-3 parameter coordinator required "
                "for the HCU consumer-stream lifetime fix."
            )
        param_coordinator = get_param_coordinator()
        allgather_stream_attr = "_PartitionedParameterCoordinator__allgather_stream"
        if not hasattr(param_coordinator, allgather_stream_attr):
            raise RuntimeError(
                "The installed DeepSpeed version does not expose the expected ZeRO-3 "
                "parameter coordinator all-gather stream."
            )
        from deepspeed.runtime.zero.partitioned_param_coordinator import iter_params
        from deepspeed.utils import z3_leaf_module

        fetch_stream = getattr(param_coordinator, allgather_stream_attr)
        if fetch_stream is None:
            raise RuntimeError(
                "DeepSpeed ZeRO-3 did not create an asynchronous parameter-fetch stream "
                "despite overlap_comm=true."
            )

        patch_attr = "_diffsynth_record_fetched_params_on_consumer_stream"
        if not getattr(param_coordinator, patch_attr, False):
            original_fetch_sub_module = param_coordinator.fetch_sub_module

            def fetch_sub_module_with_consumer_lifetime(current_submodule, forward):
                consumer_stream = device_accelerator.current_stream()
                # ZeRO initialization, optimizer updates, and the preceding
                # module may produce parameter shards on the compute stream.
                # Publish those writes to the private fetch stream at each
                # fetch boundary before DeepSpeed launches its prefetch work.
                fetch_stream.wait_stream(consumer_stream)
                original_fetch_sub_module(current_submodule, forward)

                # DeepSpeed allocates gathered parameter storage on its private
                # fetch stream, then publishes it to the current compute stream
                # with wait_stream().  The single-parameter AllGatherHandle does
                # not record the consumer stream on that storage.  Under full
                # model memory pressure the caching allocator can consequently
                # recycle the storage across streams and create an event cycle.
                # Recording the actual consumer preserves overlap while making
                # allocation lifetime match the existing stream dependency.
                for param in iter_params(
                    current_submodule,
                    recurse=z3_leaf_module(current_submodule),
                ):
                    if (
                        param.ds_status == ZeroParamStatus.AVAILABLE
                        and param.data.device.type != "cpu"
                    ):
                        param.data.record_stream(consumer_stream)

            param_coordinator.fetch_sub_module = fetch_sub_module_with_consumer_lifetime
            setattr(param_coordinator, patch_attr, True)

    class IndependentStreamAllReduceCoalescedHandle:
        """Publish an all-reduced ZeRO parameter through the private fetch stream.

        HCCL may execute the collective on an internal communication stream.
        ``block_current_stream`` adds only HCCL -> ZeRO-fetch ordering.  DeepSpeed
        then adds ZeRO-fetch -> compute ordering in ``fetch_sub_module``, and the
        wrapper above records the gathered storage on that consumer stream.  No
        compute -> fetch reverse dependency is introduced.
        """

        def __init__(self, work, params):
            self.work = work
            self.params = params
            self.complete = False

            for param in self.params:
                if param.ds_status != ZeroParamStatus.INFLIGHT:
                    raise RuntimeError(f"expected param {param.ds_summary()} to be inflight")

        def wait(self, **kwargs):
            if self.complete:
                return

            self.work.block_current_stream()
            for param in self.params:
                if param.ds_status != ZeroParamStatus.INFLIGHT:
                    raise RuntimeError(f"expected param {param.ds_summary()} to be inflight")
                param.ds_status = ZeroParamStatus.AVAILABLE
            self.complete = True

    def wrap_all_gather_coalesced(original, param_name):
        def all_gather_coalesced(params, safe_mode=False, quantize=False):
            params = list(params)
            if (
                len(params) != 1
                or safe_mode
                or quantize
                or params[0].ds_secondary_tensor is not None
            ):
                return original(params, safe_mode=safe_mode, quantize=quantize)

            param = params[0]
            if param.ds_tensor.status != PartitionedParamStatus.AVAILABLE:
                return original(params, safe_mode=safe_mode, quantize=quantize)
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(
                    f"Expected LoRA parameter {param_name} to be NOT_AVAILABLE before fetch, "
                    f"got {param.ds_summary()}"
                )

            process_group = param.ds_process_group
            partition_rank = dist.get_rank(group=process_group)
            world_size = dist.get_world_size(group=process_group)
            expected_aligned_numel = param.ds_tensor.ds_numel * world_size
            if param.ds_numel_aligned != expected_aligned_numel:
                raise RuntimeError(
                    f"Unexpected ZeRO partition layout for LoRA parameter {param_name}: "
                    f"ds_numel_aligned={param.ds_numel_aligned}, "
                    f"partition_numel={param.ds_tensor.ds_numel}, world_size={world_size}"
                )

            param.ds_status = ZeroParamStatus.INFLIGHT
            flat_tensor = torch.zeros(
                param.ds_numel_aligned,
                dtype=param.ds_tensor.dtype,
                device=get_accelerator().current_device_name(),
                requires_grad=False,
            )
            param.data = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
            partition_start = param.ds_tensor.ds_numel * partition_rank
            flat_tensor.narrow(0, partition_start, param.ds_tensor.ds_numel).copy_(param.ds_tensor)
            work = dist.all_reduce(flat_tensor, group=process_group, async_op=True)
            if not use_independent_fetch_stream:
                return AllReduceCoalescedHandle(handle=work, params=params)

            for param in params:
                # Keep the Work alive through parameter consumption.  Replacing
                # it at the next fetch is safe because release and the next fetch
                # are ordered through the same private fetch stream.
                setattr(param, fetch_work_attr, work)
            return IndependentStreamAllReduceCoalescedHandle(
                work=work,
                params=params,
            )

        return all_gather_coalesced

    patched_params = []
    for name, param in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        if not hasattr(param, "all_gather_coalesced"):
            continue
        if getattr(param, "_diffsynth_single_param_all_reduce", False):
            continue
        param.all_gather_coalesced = wrap_all_gather_coalesced(param.all_gather_coalesced, name)
        param._diffsynth_single_param_all_reduce = True
        patched_params.append(name)

    if not patched_params:
        raise RuntimeError(
            "--deepspeed_zero3_lora_single_param_all_reduce was enabled, "
            "but no partitioned PEFT LoRA parameters were found."
        )
    if accelerator.is_main_process:
        if use_independent_fetch_stream:
            print(
                f"Patched {len(patched_params)} PEFT LoRA parameters to use all-reduce "
                "on DeepSpeed's private fetch stream; parameter producers are published "
                "at each fetch boundary and fetched storage is tracked on its consumer "
                "stream while overlap_comm remains enabled."
            )
        else:
            print(
                f"Patched {len(patched_params)} PEFT LoRA parameters to use all-reduce "
                "on the default parameter-fetch stream."
            )


class PerformanceMeter:
    """Windowed training throughput and model-FLOPs estimates."""

    def __init__(self, accelerator, model, interval=10, peak_tflops=None):
        self.accelerator = accelerator
        self.interval = max(1, int(interval))
        self.peak_tflops = peak_tflops
        # Only rank 0 measures. Every rank running a device synchronize twice
        # per micro-step stalls CPU run-ahead and serializes the ZeRO-3
        # communication overlap, which costs far more than the telemetry is
        # worth. Ranks are already lock-stepped by the optimizer collectives,
        # so rank 0's window is representative.
        self.enabled = bool(accelerator.is_main_process)
        self.optimizer_steps = 0
        self.micro_steps = 0
        self.logical_tokens = 0
        self.tokens_valid = True
        self._window_start = None
        if not self.enabled:
            self.num_params = 0
            self.recomputed_params = 0
            self.flops_per_token = 0
            return
        unwrapped = accelerator.unwrap_model(model)
        dit = getattr(getattr(unwrapped, "pipe", None), "dit", None)
        self.num_params = self._parameter_count(dit)
        self.recomputed_params = self._recomputed_parameter_count(unwrapped, dit)
        self.flops_per_token = 6 * self.num_params + 2 * self.recomputed_params

    @staticmethod
    def _parameter_count(module):
        if module is None:
            return 0
        total = 0
        for name, parameter in module.named_parameters():
            if "lora_" in name.lower():
                continue
            total += int(parameter.ds_numel if hasattr(parameter, "ds_numel") else parameter.numel())
        return total

    def _recomputed_parameter_count(self, model, dit):
        if not getattr(model, "use_gradient_checkpointing", False) or dit is None:
            return 0
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            return self.num_params
        return sum(self._parameter_count(block) for block in blocks)

    def start(self):
        # No device synchronize: the window spans many steps, and rank 0 is
        # bracketed each step by the `loss.item()` device-to-host copy, so
        # per-step queuing noise averages out over the window.
        if self.enabled and self._window_start is None:
            self._window_start = time.perf_counter()

    def end(self, model, optimizer_step=True):
        if not self.enabled or self._window_start is None:
            return None
        logical_tokens = int(getattr(self.accelerator.unwrap_model(model), "last_perf_tokens", 0))
        self.micro_steps += 1
        if logical_tokens > 0:
            self.logical_tokens += logical_tokens
        else:
            self.tokens_valid = False
        if not optimizer_step:
            return None
        self.optimizer_steps += 1
        if self.optimizer_steps < self.interval:
            return None

        window_elapsed = time.perf_counter() - self._window_start
        if window_elapsed <= 0:
            self._reset_window()
            return None

        # Rank 0's own device, extrapolated to the cluster. No collectives.
        world_size = self.accelerator.num_processes
        result = {
            "perf/step_time_s": window_elapsed / self.optimizer_steps,
            "perf/samples_per_sec": world_size * self.micro_steps / window_elapsed,
        }
        if self.tokens_valid and self.flops_per_token > 0:
            tokens_per_sec_per_gpu = self.logical_tokens / window_elapsed
            result["perf/logical_tokens_per_sec"] = tokens_per_sec_per_gpu * world_size
            result["perf/tokens_per_sec_per_gpu"] = tokens_per_sec_per_gpu
            estimated_tflops = self.flops_per_token * tokens_per_sec_per_gpu / 1e12
            result["perf/estimated_tflops_per_gpu"] = estimated_tflops
            if self.peak_tflops and self.peak_tflops > 0:
                result["perf/mfu"] = estimated_tflops / self.peak_tflops

        self._reset_window()
        return result

    def _reset_window(self):
        self.optimizer_steps = 0
        self.micro_steps = 0
        self.logical_tokens = 0
        self.tokens_valid = True
        self._window_start = time.perf_counter()


def _format_performance_metrics(metrics):
    fields = [
        f"{metrics['perf/step_time_s']:.3f}s/step",
        f"{metrics['perf/samples_per_sec']:.3f} samples/s",
    ]
    tflops = metrics.get("perf/estimated_tflops_per_gpu")
    if tflops is not None:
        fields.append(f"{tflops:.1f} TFLOPS/GPU")
    mfu = metrics.get("perf/mfu")
    if mfu is not None:
        fields.append(f"MFU {mfu:.1%}")
    return "Performance: " + ", ".join(fields)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    deepspeed_zero3_lora_single_param_all_reduce = bool(
        getattr(args, "deepspeed_zero3_lora_single_param_all_reduce", False)
    )

    if accelerator.is_main_process:
        save_training_args(args)

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    _configure_deepspeed_zero3_lora_single_param_all_reduce(
        accelerator,
        enabled=deepspeed_zero3_lora_single_param_all_reduce,
    )

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    _patch_deepspeed_zero3_lora_single_param_all_reduce(
        accelerator,
        model,
        enabled=deepspeed_zero3_lora_single_param_all_reduce,
    )

    # W&B is the user-facing switch for training telemetry.  Performance
    # metrics are computed automatically when W&B logging is enabled and are
    # forwarded through ModelLogger together with the loss.  Keep the meter
    # behind the same switch so users do not need two flags for one logging
    # destination, while preserving the interval and peak-TFLOPs knobs.
    enable_wandb_log = bool(getattr(model_logger, "enable_wandb_log", False))
    if args is not None:
        enable_wandb_log = enable_wandb_log or bool(getattr(args, "enable_wandb_log", False))

    perf_meter = None
    if enable_wandb_log:
        perf_meter = PerformanceMeter(
            accelerator, model,
            interval=getattr(args, "performance_log_interval", 10),
            peak_tflops=getattr(args, "hardware_peak_tflops", None),
        )
        if accelerator.is_main_process:
            print(f"W&B performance metrics enabled: interval={perf_meter.interval}.")

    initialize_deepspeed_gradient_checkpointing(accelerator)
    with contextlib.ExitStack() as debug_stack, maybe_enable_torch_profiler(accelerator, args) as torch_profiler:
        from .zero3_cpu_pipeline import create_cpu_adam_pipeline
        cpu_pipeline = create_cpu_adam_pipeline(model)
        if cpu_pipeline is not None:
            debug_stack.callback(cpu_pipeline.close)
        from .zero3_debug import create_zero3_debug
        zero3_debug = create_zero3_debug(accelerator, model, model_logger.output_path)
        if zero3_debug is not None:
            debug_stack.callback(zero3_debug.close)
        for epoch_id in range(num_epochs):
            progress_bar = tqdm(dataloader) if accelerator.is_main_process else dataloader
            for step_id, data in enumerate(progress_bar):
                with accelerator.accumulate(model):
                    if zero3_debug is not None:
                        zero3_debug.begin(epoch_id, step_id)
                    _set_training_rng_context(
                        accelerator,
                        model,
                        getattr(args, "seed", None),
                        epoch_id,
                        step_id,
                    )
                    if perf_meter is not None:
                        perf_meter.start()
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    if zero3_debug is not None:
                        zero3_debug.loss(loss)
                    accelerator.backward(loss)
                    if enable_model_cpu_offload:
                        offload_manager.after_backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if zero3_debug is not None:
                        zero3_debug.boundary("after_update")
                    if accelerator.is_main_process and hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix({"loss": f"{loss.detach().item():.4f}"})
                    perf_metrics = (
                        perf_meter.end(model, optimizer_step=accelerator.sync_gradients)
                        if perf_meter is not None else None
                    )
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                    if perf_metrics is not None and accelerator.is_main_process:
                        print(_format_performance_metrics(perf_metrics))
                        model_logger.log_metrics(accelerator, perf_metrics, model_logger.num_steps)
                if torch_profiler is not None:
                    torch_profiler.step()
            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            deepspeed_plugin = accelerator.state.deepspeed_plugin
            zero_stage = ds_config.get("zero_optimization", {}).get(
                "stage",
                getattr(deepspeed_plugin, "zero_stage", None),
            )
            if torch.version.hip is None or zero_stage != 3:
                print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
                return
            # HCU only: Accelerate's shorthand DeepSpeed config drops unknown
            # nested keys, so the HCU launch scripts carry no
            # activation_checkpointing section. Without configure(),
            # gradient_checkpoint_forward falls back to torch's non-reentrant
            # checkpoint, whose recompute runs while ZeRO-3 parameters are
            # partitioned (numel 0) and fails with a metadata mismatch. Route
            # recompute through DeepSpeed's reentrant implementation instead.
            import deepspeed
            deepspeed.checkpointing.configure(mpu_=None)
