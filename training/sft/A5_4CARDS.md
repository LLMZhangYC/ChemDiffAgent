# A5 四卡 SFT：使用当前仓库和现有环境

入口沿用 `scripts/train_sft_npu.sh` → `training/sft/tools/launch_npu.py` →
`training/sft/train/sft_sdar.py`。四卡配置是
`training/sft/configs/chemdiffagent_sft_full_npu_4cards.yaml`。

本配置从当前仓库的 `chemdiffagent_sft_full_npu.yaml` 派生，不替换模型实现、
算子调用或依赖文件。继续使用你已经验证匹配的 Python、torch_npu 和 CANN 环境，
无需为了这次四卡调整重新安装 requirements。

## 配置差异

| 字段 | 当前八卡全量配置 | 新增四卡配置 |
| --- | --- | --- |
| `runtime.num_processes` | 8 | 4 |
| `runtime.visible_devices` | `0,1,2,3,4,5,6,7` | `0,1,2,3` |
| `training.gradient_accumulation_steps` | 16 | 32 |
| 每卡 micro batch | 1 | 1 |
| 全局 batch | 128 | 128 |

除运行名称外，其余字段与当前八卡全量配置相同：完整模型、3 epochs、BF16、
梯度检查点、学习率、验证集切分、validation、profiling，以及当前的序列长度策略。
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
