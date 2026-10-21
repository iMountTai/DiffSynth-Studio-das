# ZeRO-3 overlap NaN 排查

诊断默认关闭，不改 overlap_comm、offload ratio、训练精度或优化器。仅适用于 ZeRO-3；暂不支持 NVMe optimizer swap。修改本地代码后，需要在实际运行的容器中使用相同文件。

在**训练阶段的 accelerate launch 之前**设置环境变量；多机时每个节点都要设置。不要把诊断变量加到数据预处理阶段。保持原来的数据、seed、checkpoint、卡数与其他优化开关相同，只将实际 DeepSpeed JSON 中的 overlap_comm 切换为 true/false 做对照。不要为了 Full 排查启用 LoRA 专用参数。

先用 loss 模式复现：

```bash
export DIFFSYNTH_ZERO3_DEBUG=loss
export DIFFSYNTH_ZERO3_DEBUG_SYNC=none
export DIFFSYNTH_ZERO3_DEBUG_DIR=./zero3-debug/overlap-on-loss
# 在这里执行原来的训练命令，overlap_comm=true
```

然后从同一初始状态运行完整检查：

```bash
export DIFFSYNTH_ZERO3_DEBUG=full
export DIFFSYNTH_ZERO3_DEBUG_SYNC=none
export DIFFSYNTH_ZERO3_DEBUG_DIR=./zero3-debug/overlap-on-full
```

建议 overlap=false 也采集一份 full 日志作为有限值/量级对照。每次运行使用独立目录，不从已经出现 NaN 的 checkpoint 继续。

日志是每 rank 独立的 JSONL，带 pid/时间戳，不会覆盖旧日志。未设置目录时放在训练 output_path/zero3_debug 下。startup 包含版本、DeepSpeed 配置、实际 optimizer 属性和源码路径；hook 记录实际安装版本的方法签名。不支持的方法/签名会显式报错并恢复已安装补丁。

full 检查顺序：

1. loss：backward 前的原始 loss。
2. grad.produced：梯度进入 DeepSpeed bucket 处理之前，包含参数名和 ds_id。
3. grad.reduced：归约函数返回、进入分区/offload 处理时的分片。
4. optimizer.pre_clip：子组准备后、裁剪前的 FP32 主权重和梯度；CPU offload 时这里检查的是 CPU 梯度。
5. optimizer.total_norm/loss_scale/clip_grad：实际传给梯度缩放路径的输入。
6. optimizer.before/after：更新前后的主权重和 Adam state，更新前还检查缩放后的梯度。
7. parameter.reassigned：回写后的低精度参数分片。

每条张量记录包含 shape、dtype、device、非有限值数量、absmax。absmax 遇到非有限结果写 null。不会保存完整模型张量；不能据此证明所有有限值都数值正确，也不能跨 rank 直接比较不同参数分片是否相等。

## 同步定位

若 full 模式反而不再复现，返回 loss 模式，每次只测试一个同步点：

| DIFFSYNTH_ZERO3_DEBUG_SYNC | 额外同步的位置 |
|---|---|
| none | 不增加下面这些设备同步 |
| fetch | 每次参数 fetch_sub_module 返回后 |
| reduce | 两种梯度归约方法的入口和出口 |
| optimizer | ZeRO optimizer.step 入口，在 norm/裁剪/CPUAdam 读取之前 |

例如：

```bash
export DIFFSYNTH_ZERO3_DEBUG=loss
export DIFFSYNTH_ZERO3_DEBUG_SYNC=fetch
export DIFFSYNTH_ZERO3_DEBUG_DIR=./zero3-debug/overlap-on-fetch-sync
```

同步使用整设备 synchronize，是诊断实验，不是修复。full 的统计会读取 GPU 标量，增加同步；即使 loss/none 也会在 loss 和 update 后做一次 rank 间异常标志归约，所以也不是完全无扰动。若 loss/none 就使 NaN 消失，这本身是时序敏感的证据，需保留无诊断运行日志。

发现非有限值时先写 FIRST NONFINITE 和日志，不在参数 hook 内单卡退出。所有 rank 在 before_backward 或 after_update 统一归约异常标志后报错停止。梯度 hook 检出的坏值可能继续经过当前更新，但会在常规 step 日志/保存之前停止。已有通信卡死或设备错误不保证能到达这个边界；本工具不修复 collective 卡死。

请反馈所有节点的所有 rank JSONL、完整训练控制台日志，以及每次实验使用的启动命令/配置。重点保留 FIRST NONFINITE 前后的记录。诊断运行耗时不能用于性能比较。

关闭：

```bash
unset DIFFSYNTH_ZERO3_DEBUG DIFFSYNTH_ZERO3_DEBUG_SYNC DIFFSYNTH_ZERO3_DEBUG_DIR
```
