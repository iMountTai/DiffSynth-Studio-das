# MiniMax-H3 全量训练：CPUAdam pipeline / FA CPU cache 实验

两个开关默认关闭，可独立测试。面向 ZeRO-3、BF16、
optimizer CPU partial offload（保留 ratio=0.7），不修改已安装的 DeepSpeed 或 FA。
`writeback` 按运行时 ZeRO-3/CPUAdam 接口检查；`fused` 的 C++ 更新实现仍限定 DeepSpeed **0.18.4**。
不需要同步整个 DeepSpeed/FA 仓库；新 C++ 源码随 DiffSynth 一起同步。
本地仅通过 CPU mock/梯度/静态检查；真实 CPU 扩展构建、HIP DMA、FA 和多卡性能待验证。

## 先做独立预检

在远端现有训练容器中，进入 `DiffSynth-Studio` 根目录，使用同一个训练 Python：

```bash
# 单进程先编译，避免首次多 rank 训练时一起构建。需要 C++ 编译器、OpenMP、ninja。
MAX_JOBS=1 python examples/dev_tools/test_h3_training_experiments.py --native
python examples/dev_tools/test_h3_training_experiments.py --fa
```

`--native` 对比当前安装的 CPUAdam 与新增扩展，覆盖缩放、Adam/AdamW、bias correction、
SIMD 尾部、beta 改变和非连续 step。`--fa` 使用真实 FA，检查输出、梯度和调用次数。
当前 157 容器中的 HCU `flash_attn` 包在导入时会额外导入 `pytest`；缺少它时普通 attention
会回退到 `torch`，不能把能运行的基线当成 FA2 基线。A/B 两组都应确认实际选中 `flash_attention_2`。
它们不代替完整 ZeRO-3 测试或长期收敛验证。预检失败先停止相应实验，保留完整报错。

## A/B 开关

保持原有数据、seed、ratio、sub_group_size、overlap_comm 和训练启动命令不变。
把下列环境变量传到**每个训练 rank**，多节点时也必须一致：

| 实验 | DIFFSYNTH_CPU_ADAM_PIPELINE | DIFFSYNTH_FA_CPU_CACHE_LAYERS |
|---|---|---|
| 基线 / 恢复原路径 | off | 0 |
| 仅异步写回 | writeback | 0 |
| 融合 CPUAdam + 异步写回 | fused | 0 |
| 仅 FA 缓存 | off | 1 |
| 两项组合（单项验证之后） | fused | 1 |

例如，在运行原有 launcher 前：

```bash
export DIFFSYNTH_CPU_ADAM_PIPELINE=fused
export DIFFSYNTH_FA_CPU_CACHE_LAYERS=0
export DIFFSYNTH_CPU_PIPELINE_LOG=1
export DIFFSYNTH_FA_CACHE_LOG=0
# 然后执行原有训练启动命令；确认 launcher 没有清理这些环境变量。
```

FA 单项用 `DIFFSYNTH_CPU_ADAM_PIPELINE=off`、`DIFFSYNTH_FA_CPU_CACHE_LAYERS=1`、
`DIFFSYNTH_FA_CACHE_LOG=1`。先 1 层，再 2/5 层；不要直接设置 50。
日志开关用于确认分派，正式性能测量设为 0。`DIFFSYNTH_CPU_ADAM_BUILD_VERBOSE=1`
可显示扩展编译信息。

## CPUAdam pipeline

- `writeback`：原始 CPUAdam，更新后在 CPU 转 BF16，双 pinned-host buffer 异步 H2D。
- `fused`：基于官方 DeepSpeed v0.18.4 CPUAdam SIMD/标量更新主体，加入 FP32 梯度缩放
  和 BF16 staging 输出；原有 Python optimizer state、step 和 checkpoint 格式保留。
- 仅支持 CPU FP32 master/gradient/M/V、GPU BF16 分片。不支持 NVMe、parameter offload、
  GradScaler、ZenFlow。`writeback` 对其他 DeepSpeed 版本须重新做多卡数值验证；
  `fused` 暂不支持非 0.18.4 版本。显式开启但不满足条件时直接报错。
- GPU subgroup 保留原有实现。CPU staging 占用约 `2 × 最大 CPU subgroup 元素数 × 2 bytes`
  主机 pinned memory，不新建 GPU 参数 staging。
- 重用 host buffer 前等待对应 DMA event；post-step gather 前向提交 stream 建立依赖；
  异常及退出时 drain copy stream。仍须用多卡 trace 确认真实 RCCL stream 依赖与性能。
- `fused` 保持原生 Adam 算法和精度类型，但不能未经数值测试宣称不同编译产物逐位相同。
- 不与 `DIFFSYNTH_ZERO3_DEBUG` 同时启用，避免旧诊断钩子在 deferred scaling/DMA 中间阶段
  读取非最终数据。先分别验证；loss、global grad norm、参数更新仍应通过训练日志比较。

## FA CPU cache

- 只选前 N 个 DiT block，token refiner 不缓存。保留整 block 的 DeepSpeed reentrant checkpoint。
- 首次 forward 缓存 O/LSE/RNG 到 CPU；重算开始时预取到 GPU，同时正常重算 QKV、Norm、RoPE；
  到 FA 位置等待预取，创建真正的 autograd 节点，使用原 FA backward。
- 每次 checkpoint 调用独立缓存，避免不同 step/microbatch 串数据。CPU 缓存生命周期随
  checkpoint graph；记录与回放数量、shape/stride/device、cu_seqlens 标识不匹配时报错。
- 仅支持 FA2、FP16/BF16、head_dim 为 8 的倍数、packed self attention、dropout=0、非 causal。
  不支持 repeated backward/retain_graph、double backward、整 block compile、独立 checkpoint offload
  或换成非 reentrant 路径；首次 replay 后即释放对应 pinned CPU 缓存。
- 使用本地 flash-attention-cutlass 的私有 varlen forward/backward 接口；版本不匹配可能报错。
  必须先跑 `--fa`。不强行兼容不明接口，也不吞掉 kernel 错误后重复执行。
- pinned CPU 缓存按完整 O/LSE/RNG 计算，约 207 MiB/层/卡（14912×56×128 BF16 估算）。
  H2D 预取会把 O 的生命周期提前到 block 重算开始，因此仍可能增加局部 GPU 峰值，不能保证满显存时必不 OOM。
- 为保证异步拷贝源生命周期，当前原型在每个选中 checkpoint 的退出边界完成 event host wait。
  D2H 可与该 block 后续计算重叠，H2D 可与 QKV 等重算重叠，但尚未做跨 block lookahead；
  trace 若显示边界等待抵消收益，应先减少层数或关闭，不能把缓存命中等同于加速。

## 远端验收与反馈

分别保存基线、writeback、fused、FA-1-layer 的有效配置和全部 rank 日志，比较 warmup 后多步
step 时间、loss、global grad norm、峰值 allocated/reserved、主机 RAM/pinned memory。
同时验证保存 checkpoint 后恢复训练。计时排除首次构建、初始化和 profiler 开销。
日志应出现 CPU subgroup 数和 H2D bytes，FA 应出现 `fa_forward_skipped=1`。
若发生 NaN、loss/梯度异常、hang、OOM 或预检不一致，关闭对应开关，保留首次异常前后的日志。

恢复原路径只需：

```bash
export DIFFSYNTH_CPU_ADAM_PIPELINE=off
export DIFFSYNTH_FA_CPU_CACHE_LAYERS=0
```

## 原生扩展来源

`diffsynth/core/ops/csrc/h3_cpu_adam/` 的四个源文件来自
https://github.com/deepspeedai/DeepSpeed/tree/v0.18.4/csrc ，保留 Microsoft 版权与 Apache-2.0
标记并注明修改。增加的入口仅用于此实验，独立 JIT 编译为
`diffsynth_cpu_adam_0184_v1`，不会覆盖安装环境的 `cpu_adam`。

## LoRA：利用显存减少重算

单节点 `examples-hcu/MiniMax-H3/lora/MiniMax-H3-Ref2VA.sh` 和多节点
`examples-hcu/MiniMax-H3/lora/start.sh` 均接受下面的环境变量，默认全为 `0`。
原脚本的数据下载、预处理和本地模型读取流程不变；这些开关只在训练阶段生效。
多节点由 `start.sh` 将开关传到各节点容器，确保所有 rank 一致。

| 实验 | `DIFFSYNTH_FA_CPU_CACHE_LAYERS` | `DIFFSYNTH_FA_GPU_CACHE_LAYERS` | `DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS` |
|---|---:|---:|---:|
| 基线 | 0 | 0 | 0 |
| GPU 留存 FA O/LSE/RNG，跳过前 N 层的重算 FA forward | 0 | N | 0 |
| 后 N 个完整 DiT block 不做 checkpoint，跳过整块重算 | 0 | 0 | N |
| 组合 | 0 | N | M |

GPU cache 和原有 CPU cache 二选一。GPU cache 保留前 N 个 block 首次 forward 的
FA O/LSE/RNG，checkpoint replay 时直接用原 FA backward 构图；没有 D2H/H2D，
但会在 forward 至对应 backward 期间占用 GPU 显存。整块免 checkpoint 选最后 N 个
block，保留其反向所需激活；与 cache 组合时要求 `N + M <= DiT block 数`。
两项都要求原脚本的 `--use_gradient_checkpointing`，GPU cache 还要求 FA2 和
DeepSpeed reentrant checkpoint。先从各 1 层增加，用实际 peak allocated/reserved
决定层数；20 GB 空闲不能直接换算为安全层数。

例如只测 GPU FA cache，在执行原单节点脚本前设置：

```bash
export DIFFSYNTH_FA_GPU_CACHE_LAYERS=1
export DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS=0
bash examples-hcu/MiniMax-H3/lora/MiniMax-H3-Ref2VA.sh
```

只测整块免 checkpoint，将 GPU cache 层数设为 `0`、免 checkpoint 层数设为 `1`；
组合时两者都设为 `1`。
开启 `DIFFSYNTH_FA_CACHE_LOG=1` 可确认每个 cache replay 的命中和 GPU bytes，
正式计时设回 `0`。HCU 上可先执行
`python examples/dev_tools/test_h3_training_experiments.py --fa-gpu`，检查真实 FA
输出、梯度和 forward 调用次数；最终仍以相同训练配置的 trace、峰值显存和长程 loss
比较。禁用实验只需把三个层数变量设回 `0`。
