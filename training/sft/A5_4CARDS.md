# A5 四卡 SFT：当前 HCCL 诊断配置

入口沿用 `scripts/train_sft_npu.sh` → `training/sft/tools/launch_npu.py` →
`training/sft/train/sft_sdar.py`。四卡配置是
`training/sft/configs/chemdiffagent_sft_full_npu_4cards.yaml`。

当前该配置用于复现第 363 步前向故障的对照：卡 4/5/6/7、梯度检查点关闭、
`deepspeed_overlap_comm: false`、32 步累积、阶段和通信入口日志开启，512 个 micro-steps 后退出。
这次停止不执行验证或保存 checkpoint；如需完整训练，设置
`diagnostics.stop_after_micro_steps=null`。本轮在已失败的 overlap=false 基线上补充观测，
不再修改通信执行策略。

本配置从当前仓库的 `chemdiffagent_sft_full_npu.yaml` 派生，不替换模型实现、
算子调用或依赖文件。继续使用你已经验证匹配的 Python、torch_npu 和 CANN 环境，
无需为了这次四卡调整重新安装 requirements。

## 配置差异

| 字段 | 当前八卡全量配置 | 新增四卡配置 |
| --- | --- | --- |
| `runtime.num_processes` | 8 | 4 |
| `runtime.visible_devices` | `0,1,2,3,4,5,6,7` | `4,5,6,7` |
| `training.gradient_accumulation_steps` | 16 | 32 |
| 每卡 micro batch | 1 | 1 |
| 全局 batch | 128 | 128 |
| `training.gradient_checkpointing_enable` | true | false（沿用最近失败运行） |
| `training.deepspeed_overlap_comm` | 未设置，ZeRO-3 默认 true | false |
| `diagnostics.enabled` | 未设置，默认 false | true |
| `diagnostics.stop_after_micro_steps` | 未设置 | 512 |
| `diagnostics.trace_collectives` | 未设置，默认 false | true，从第 320 步记录 |
| `runtime.env.HCCL_ENTRY_LOG_ENABLE` | 未设置 | 1 |

完整模型、3 epochs 的学习率日程、BF16、学习率、验证集切分、validation、profiling，
以及当前的序列长度策略仍沿用八卡全量配置；诊断停止会在完整训练结束前退出。
不减层，不额外截断序列。每个 optimizer update 包含 32 个 micro steps，
profiling 仍按 optimizer update 计数。

现有启动器从 `accelerate_configs/1_npu_zero3.yaml` 读取模板，再根据 YAML
动态设置 rank 数、BF16、累积步数和梯度裁剪。该模板名字中的 `1_npu` 不会把
本次训练限制为单卡；生成的配置位于本次 run 的 `work/accelerate-*.yaml`。
ZeRO-3、无 CPU offload、`zero3_init_flag: false` 和权重聚合保存均沿用现有实现。

## 启动

先在 A5 上激活原来能运行这个项目的环境，在仓库根目录执行。先检查四卡 YAML 的
`model.pretrained_model`、`dataset.path`、`runtime.cann_root`，保留你实际可用的路径。
这些路径沿用当前八卡配置，没有额外指定新的权重下载目录。
`runtime.visible_devices` 应填四个实际可用的逻辑设备 ID。

```bash
bash scripts/train_sft_npu.sh \
  config=training/sft/configs/chemdiffagent_sft_full_npu_4cards.yaml
```

启动脚本默认解析当前 `python`，并把实际解释器路径传给配置中的
`${oc.env:CHEMDIFF_PYTHON}`。如果已经显式设置 `CHEMDIFF_PYTHON`，会使用该解释器；
确认它指向你当前匹配的环境。CANN 仍由 `runtime.cann_root` 指定，不强制设置额外的
HCCL 调度模式。启动脚本和 Python 启动器都不会执行 pip 安装。

先只验证三次 optimizer 更新，可以在同一个配置上覆盖停止条件：

```bash
bash scripts/train_sft_npu.sh \
  config=training/sft/configs/chemdiffagent_sft_full_npu_4cards.yaml \
  training.max_steps=3 training.num_train_epochs=1 \
  validation.eval_steps=3 profiling.active=2
```

这仍然加载完整模型，每个更新仍累积 32 个 micro steps；不是三次单独的前向/反向。
默认输出为 `res/<timestamp>-chemdiffagent_sft_full_npu_4cards/`，其中 `output/`
存训练产物，`work/` 存解析后的 YAML 和实际 Accelerate 配置。

## 验证范围

此次检查针对配置与当前八卡版本的一致性、启动脚本的解释器选择和参数传递。
尚未在目标 A5 上执行四卡全量训练，因此峰值显存、HCCL 通信和收敛需要实跑确认。
先检查启动日志中的 `Num processes: 4`、`gradient_accumulation_steps=32`，
再检查 run 下的训练报告、验证指标和 checkpoint。

## Accelerate 1.15 梯度累积修复与 HCCL 定位

训练循环现在在每个 micro-step 进入 `accelerator.accumulate(model)`。
Accelerate 1.15 会把 `sync_gradients` 传给 DeepSpeed，未进入累积上下文时该值一直为
true，会导致每个 micro-step 都更新参数。外部学习率调度器由训练循环在更新后推进一次，
关闭 Accelerate 按进程数额外推进 scheduler 的行为。模型实现与依赖文件未改变。

使用原四卡配置，在同一个训练进程中保留完整前向/反向和 ZeRO-3，先跑 128 个 micro-steps：

```bash
bash scripts/train_sft_npu.sh \
  config=training/sft/configs/chemdiffagent_sft_full_npu_4cards.yaml \
  runtime.visible_devices=4,5,6,7 \
  diagnostics.enabled=true diagnostics.start_micro_step=85 \
  diagnostics.stop_after_micro_steps=128
```

保持实际模型、数据和 CANN 路径。不要复用旧的 `experiment.run_dir`。
此诊断停止条件不修改原学习率日程；到达 128 个 micro-steps 后退出，不执行验证或保存权重。
未设置 `diagnostics.stop_after_micro_steps` 时仍按原训练条件正常验证、保存。
如果设置了更早的 `training.max_steps` 或 epoch 已结束，原停止条件仍会先触发。
停止在非累积边界会丢弃尚未更新的梯度，故这里选用 32 的倍数 128。

`output/metrics.jsonl` 的预期值（未发生更新跳过时）：

| micro_step | optimizer_step | scheduler_step | optimizer_updated |
| --- | --- | --- | --- |
| 31 | 0 | 0 | false |
| 32 | 1 | 1 | true |
| 64 | 2 | 2 | true |
| 94 | 2 | 2 | false |
| 96 | 3 | 3 | true |
| 128 | 4 | 4 | true |

epoch 末尾不足 32 个 micro-steps 的剩余批次按 Accelerate 默认策略再更新一次。

每个 rank 单独写入 `output/diagnostics/rank_<rank>.jsonl`，每条记录立即关闭落盘。
记录含 epoch、micro-step、阶段、时间、同步边界和 DeepSpeed 的实际更新计数。
启动记录包含库版本、卡号、HCCL 展开模式和 ZeRO 的实际配置。
阶段覆盖数据获取、搬运、mask、模型前向、loss 检查、`engine.backward`、`engine.step`、
学习率推进、耗时 gather、loss reduce、profiling、验证和保存。

首轮 `diagnostics.synchronize=false` 不新增设备同步或集合通信；`host_return` 只代表
Python 调用返回，不代表异步设备任务完成。先比较四个 rank 最后进入的阶段和最早的异常。
如果还不能定位，在同一命令末尾加 `diagnostics.synchronize=true`，在阶段前后做本卡同步：
`device_complete` 才代表同步完成；`where=pre_sync` 可能是之前未完成的任务报错，
`where=body` 表示调用期间异常，`where=post_sync` 表示阶段后的同步发现错误。
同步会改变执行时序、降低性能，若同步后不再复现，应继续调查并发，而非宣告故障已修复。

也可单独对照 `profiling.enabled=false`，或进一步使用
`runtime.env.ASCEND_LAUNCH_BLOCKING=1` 获取更精确的算子调用栈。每轮只改变一个定位条件。
梯度累积修复尚不等于修复 HCCL；独立 AllGather 通过也不覆盖整网中的异步参数预取与重算。

在运行结果目录查看：

```bash
tail -n 5 output/metrics.jsonl
for rank in 0 1 2 3; do
  tail -n 12 "output/diagnostics/rank_${rank}.jsonl"
done
```

本次本地验证包括 11 项控制流程/诊断/profiler 回归测试，以及从 Accelerate 1.15.0
官方源码提取累积、DeepSpeed 包装器和 scheduler 方法的测试（DeepSpeed engine 使用替身）。
修复前模拟 128 个 micro-steps 得到 128 次参数更新、512 次 scheduler 推进；修复后均为 4 次。
这属于 Python 控制流程验证，未执行目标 A5 的真实反向或 HCCL，仍需以上四卡运行确认。

## 第 363 步前向 AllGather 故障：仅关闭通信重叠的对照

`training.deepspeed_overlap_comm` 是可选开关：省略或设为 `null` 沿用 DeepSpeed 默认值，
`false` 关闭通信重叠，`true` 显式开启。训练入口在模型加载和 `accelerator.prepare()` 前，
将它写入实际 DeepSpeed 配置的 `zero_optimization.overlap_comm`，并在 engine 创建后
检查实际值；不匹配会立即报错。此开关不依赖 `diagnostics.enabled`。

当前仓库的四卡 YAML 已配置好以下字段，拉取后无需再次手动添加：

```yaml
training:
  gradient_checkpointing_enable: false
  gradient_accumulation_steps: 32
  deepspeed_overlap_comm: false
diagnostics:
  enabled: true
  synchronize: false
  stop_after_micro_steps: 512
```

直接用原脚本进行 512 个 micro-steps 的对照，设备默认是 4/5/6/7：

```bash
bash scripts/train_sft_npu.sh \
  config=training/sft/configs/chemdiffagent_sft_full_npu_4cards.yaml
```

15:07 的第 363 步失败运行是 checkpoint 关闭、overlap 开启。
随后 15:42 启动的对照已确认 overlap 关闭，但四卡仍在第 363 步前向失败。
因此关闭 overlap 不足以解决故障；当前先补通信入口记录，不继续同时修改多个执行条件。
梯度累积仍为 32，预取、合并聚合、profiling、模型和数据路径保留。
不要同时添加 `ASCEND_LAUNCH_BLOCKING`。
关闭 overlap 不等于关闭预取或使全部 HCCL 调用成为同步调用。
512 是诊断停止条件，不修改学习率日程；到达后不执行验证或保存 checkpoint。
若需要继续完整训练，将 `diagnostics.stop_after_micro_steps` 保持为 `null`。

启动器先打印 `requested_overlap_comm=False`，这只表示读取了请求值。
模型、数据和 DeepSpeed engine 初始化完成后，rank 0 直接打印（不依赖 INFO 日志级别）：

```text
DeepSpeed effective overlap_comm=False; gradient_accumulation_steps=32; gradient_checkpointing=False
```

这条实际值输出在第一个训练 batch 前出现，并非刚启动进程时立即出现。
在新运行目录执行以下命令，核实 engine 的实际配置：

```bash
grep '"phase": "deepspeed"' output/diagnostics/rank_0.jsonl
```

应包含 `overlap_comm=False`，预取上限仍为 `prefetch_bucket_size=50000000`（原默认值）。
若越过第 363 步，先验证到 512 步的 16 次更新；一次通过不能证明已解决长期稳定性问题。
若再次失败，比较四卡阶段日志和同一轮的 HCCL 日志。

## 继续定位：对齐四卡通信入口

2026-11-17 15:42 运行的实际配置为 `overlap_comm=False`，完成 362 micro-steps / 11 updates，
四个 rank 在第 363 步 `model_forward` 报错，输入均为 `[1, 8918]`。CANN 记录四卡
BF16 AllGather，输入各 24 MiB、输出各 96 MiB，但失败记录的 opIndex 分别是
`22102 / 22107 / 22113 / 22117`。这些是不同 rank 的异常快照，不能据此确认同一操作匹配，
也不能直接断言通信错序或某张卡未调用。`RunAllGather Send failed` / `ret[4]` 定位到
AICPU AllGather 的发送路径，尚未提供对端、传输层或资源状态的具体首错。

当前四卡配置已额外启用以下记录，仍在 512 步停止、不保存 checkpoint：

```yaml
diagnostics:
  trace_collectives: true
  collectives_start_micro_step: 320
runtime:
  env:
    HCCL_ENTRY_LOG_ENABLE: "1"
```

仍使用本页原来的 bash 启动命令。DeepSpeed 初始化后，每卡打印 `[SFT collectives]`。
`output/diagnostics/collectives_rank_<rank>.jsonl` 记录通信类型、通信组、该组内调用序号、
tensor 的数量/dtype/地址/stride，以及触发参数获取的模块名。序号从 prepare 后开始累计，
第 320 步开始写入；不是 HCCL opIndex。模块获取过程中也可能预取后续模块，
所以 `fetch_context.module` 不是失败参数的确切归属。直接调用 torch.distributed 的通信不在
此 Python 记录范围内，必须结合 CANN 日志核对。包装器原样返回 Work，不新增 wait、同步或集合通信。
逐行日志写入仍会影响主机时序；加日志后不复现不等于修好。

从仓库根目录执行，RUN 填启动器打印的本次实际运行目录：

```bash
RUN=/实际运行目录
python training/sft/tools/compare_collectives.py "$RUN/output/diagnostics" \
  > "$RUN/output/collective_comparison.json"
grep -R -n -E 'Entry-HcclAllGather|opData information|groupRank information|RunAllGather.*failed|SendRecvBatch|ProcessLayer.*failed' \
  "$RUN/work/logs/ascend" > "$RUN/output/hccl_entries_and_errors.txt"
```

比较脚本只需 Python 标准库。它比较共同序号上的 op、元素数、dtype、shape 等，报告第一个
不一致及缺失尾段。缺失尾段也可能是另一卡先失败导致本卡停止下发，不能直接当作根因。
`host_return` 仅是 Python 返回，不代表设备完成。四卡序列完全一致只能排除已记录范围内的
部分错序/尺寸问题，不能排除 HCCL 执行、资源、内存生命周期或模型算子造成的内存破坏。

`HCCL_ENTRY_LOG_ENABLE` 是 CANN 原生入口记录开关；实际 PR 是否支持及是否生效，以
`work/logs/ascend/run/plog` 中的 `Entry-HcclAllGather` 为准。未出现时请保留完整的
`work/logs/ascend`（包含 run 与 debug），不要只保留 grep 命中的 ERROR 行。

后续按证据单变量测试：先核对四卡序列；一致时比较与日志同布局的 BF16 24 MiB
in-place AllGather 长循环与独立输入缓冲区版本；再独立关闭 ZeRO 参数预取或改变聚合方式。
若脱离模型/DeepSpeed 仍失败，优先交给对应 A5 PR 的 HCCL/CANN 团队核对驱动、固件、
运行库和通信资源。不要直接照搬其他产品的展开模式或环境变量支持范围。

参考：[CANN 故障定位与入口日志](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclug_000024.html)、
[DeepSpeed 0.18.6 参数获取实现](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.6/deepspeed/runtime/zero/parameter_offload.py)。

## Jetty SQ 问题的独立最小复现

16:45 运行进一步定位到 `HcommChannelNotifyRecordOnThread` 的
`CheckOverflow(1, false)` 未通过，返回 HCCL_E_AGAIN(20)，被 AllGather 上层转成内部错误(4)。
这对应 Jetty SQ 的软件容量/反压检查；是否是消费位置更新异常仍需对应 PR 构建确认。
公开相关修订：[共享 Jetty 的 CI 更新](https://api.gitcode.com/api/v5/repos/cann/hcomm/commits/0efba3f5f82e8cb80e1c1072730b6099a7af0cbb)、
[等待可用发送空间](https://api.gitcode.com/api/v5/repos/cann/hcomm/commits/8202d04429ecd5a838c314cd3191cabf22b10b19)。

使用现有 Python/torch/torch_npu/CANN 环境，在仓库根目录执行：

```bash
bash scripts/repro_hccl_allgather.sh
```

若训练使用其他 CANN 安装路径，先设置 `CHEMDIFF_CANN_ROOT` 为原来的路径；
解释器沿用 `CHEMDIFF_PYTHON` 或当前 `python`。没有模型路径、数据集、DeepSpeed、
Accelerate、YAML 或额外 pip 依赖。启动器只创建本次结果目录并启动四个 torchrun worker。

默认条件：物理卡 4/5/6/7；BF16；每卡输入 24 MiB（12,582,912 元素），输出 96 MiB；
输入是输出中 rank 对应的分片，因此输入地址偏移为 0/24/48/72 MiB。
两个四卡 HCCL 通信域按批交替 AllGather，完成一批后才切换通信域，避免跨域在途操作的
流顺序干扰。默认总计每 rank 60,000 次（每域 30,000 次）。
两个域用于覆盖可能共享 Jetty 的 channel 使用方式，不保证每种构建都发生 Jetty 共享。
每批 8 个独立输出缓冲区，保留所有 Work/输入/输出直到 wait 和 NPU synchronize 完成，
随后才复用。默认 tensor 占用约 768 MiB/卡，另有 HCCL/运行时内部资源。

保留原环境的 HCCL 展开模式，不强制切换算法。确认日志实际使用
`AicpuAllGatherSoleMesh` 后才能把结果与本次训练执行路径直接比较。
每 1000 次左右抽查当前批次每个 rank 分片的首、中、末元素，不是全 tensor 精度测试。
子进程默认日志级别为 WARNING（2），保留 overflow 警告并避免全量 INFO 过大；
需要更详细记录时可用 `HCCL_REPRO_LOG_LEVEL=1`，同时开启 HCCL 入口日志。

输出位于启动时打印的 `res/<时间>-hccl_allgather/`，含 `console.log`、
`output/rank_0.jsonl` 到 `rank_3.jsonl` 和 `work/logs/ascend`。
`submit_enter` 是主机准备提交，`device_complete` 才标记整批完成；
检测异常所在的 iteration 不一定就是首次失败的设备操作。

失败后在这个结果目录执行：

```bash
grep -R -n -E 'overflow check failed|jetty SQ overflow|DoSendRecvBatchRx.*20|RunAllGather Send failed|HcclLaunchAicpuKernel run failed' work/logs/ascend
for r in 0 1 2 3; do tail -n 3 "output/rank_${r}.jsonl"; done
```

只有同样出现 `HcommChannelNotifyRecordOnThread` 的 overflow 检查失败，才能判为
该触发点在脱离模型与 DeepSpeed 后复现。单独的 507018/ret4、数据检查失败或超时均不足以确认。
四个 rank 都出现 `passed` 只表示本次独立场景通过，不排除训练中的调度或资源共享触发条件。

每条命令都创建新进程和新结果目录。每轮只改变一个条件：

```bash
# 对照一：只将共享输入/输出布局改成独立输入缓冲区
bash scripts/repro_hccl_allgather.sh --layout separate
# 对照二：只改为一个通信域
bash scripts/repro_hccl_allgather.sh --groups 1
# 对照三：只改为每次完成后再下发下一次
bash scripts/repro_hccl_allgather.sh --window 1
```

前三个控制项分别对应布局、通信域共享、并发深度。不要一次同时修改它们。
本脚本只完成本地语法和控制流程验证，真实 HCCL 复现结果需由 A5 运行确认。
