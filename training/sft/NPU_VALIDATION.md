# NPU 一层 SFT 验证记录

日期：2026-11-14。执行位置：`zyc` 容器，项目 `/data/hd/ChemDiffAgent`。

## 已执行的范围

以下训练使用**随机初始化的测试模型**：源模型 3 层，抽取后 1 层，hidden size 64、
intermediate size 128、4 个 attention heads、2 个 KV heads、head dim 16、词表 16。
它用于验证代码链路，不代表公开 SDAR-8B 权重的质量、内存或性能。

环境为既有 SFT venv：PyTorch 2.10.0+cpu / torch-npu 2.10.0.post7 /
Transformers 4.52.4 / Accelerate 1.7.0 / DeepSpeed 0.18.6 / CANN 9.3.0。
所有集成训练均调用原 `train/sft_sdar.py`，使用 NPU BF16 和 DeepSpeed ZeRO-3。

| 验证 | 结果 |
| --- | --- |
| CPU 权重抽取/裁剪测试 | 10 项通过 |
| 分片权重选择性读取 | 未打开只包含后续层的 shard；混合 shard 未调用后续层张量的 get_tensor |
| 参数完整性 | 导出 14 个张量；逐项与源权重完全一致；无其他 decoder 层 |
| 前缀模型等价性 | CPU 上，导出模型与源模型仅执行第一层时 logits 完全一致 |
| 单文件 safetensors | 抽取及 AutoModel 重载通过 |
| 错误输入 | 越界层数、异常 shape、重复输出目录均明确失败；失败不留下半成品 checkpoint |
| 裁剪边界 | padding 总预算、8192 上限、工具段截断、样本索引对齐、无监督样本处理均通过 |
| 短序列原 SFT 入口 | 实际长度 280；完成前向、反向、1 次参数更新和保存 |
| 8192 原 SFT 入口 | P=128，R=4032，展开长度 8192；完成前向、反向、1 次参数更新和保存 |
| 梯度累积 | 2 个 microbatch、accumulation=2，对应 1 次 optimizer 更新和 1 次 scheduler 推进 |
| checkpoint 重载 | 仅 1 层、14 个有限张量；本地 SDAR 类和 AutoModel 类的 CPU logits 差为 0 |
| 实际权重更新 | q_proj 最大变化 0.0010986328125；基线先舍入至保存 dtype，排除了单纯类型转换差异 |

小模型 8192 测试的最后 loss 为 1.3203125；先前通过的运行中 NPU allocated 峰值约
4.02 GiB。此数字仅针对上述小模型，不能用于估算完整 hidden size / 词表的一层 SDAR-8B。

## NPU 数值参照

`tests/verify_npu_reference.py` 使用相同的 BF16 舍入后权重、固定输入、相同加噪位置和
block attention mask，对照 CPU FP32 与 NPU BF16。RoPE 缓冲保持 FP32。
序列长度为 32；预设的 logits/梯度相对 L2 误差门限为 5%，loss 绝对误差门限为 0.02。

| 指标 | 实测 |
| --- | ---: |
| logits 相对 L2 误差 | 0.00419619（约 0.42%） |
| logits 最大绝对误差 | 0.00229904 |
| 梯度相对 L2 误差 | 0.00687727（约 0.69%） |
| CPU loss | 1.37334180 |
| NPU loss | 1.37347925 |
| 带 mask 的单 token KV-cache 解码相对 L2 误差 | 0.00380658（约 0.38%） |

所有参数梯度有限，测试通过。该结果是 CPU 数学参照，不是 CUDA FlashAttention 与 NPU
的逐算子比较；不承诺跨设备逐位一致，也未评价一层模型的化学任务能力。

## 本次运行发现并修复的问题

1. SDAR 无 projection bias 时，原分组可能产生空的 no-decay 参数组；DeepSpeed 去掉空组后，
   PyTorch 2.10 scheduler 的 group 数量不匹配。现在在构造 optimizer 前过滤空组。
2. 原循环每个 microbatch 都推进外部 scheduler。现在以 DeepSpeed 实际 optimizer 更新边界
   为准，避免在梯度累积期间提前推进学习率。未替换 optimizer、scheduler 或训练框架。

## 结果文件

统一目录：`/data/hd/ChemDiffAgent-work/validation`。

- `random-1l/reduction_report.json`：实际选择的张量和读取 shard。
- `trained-8k/preprocessing_report.json`：P/R/展开长度。
- `trained-8k/training_report.json`：设备、层数、loss、更新和调度次数、内存。
- `trained-accum/training_report.json`：梯度累积次数。
- `numerical-reference.json`：CPU/NPU 数值对照。
- `checkpoint-validation.json`：权重更新与重载检查。
- `train-8k-final.log`、`train-accum-final.log`：最终集成运行日志。
- `/data/hd/ChemDiffAgent-work/unit-tests-final.log`：CPU 测试日志。

真实公开权重尚未导入；完整尺寸一层模型的 8k 内存/精度验证、GPU/NPU 实权重对齐、
实际化学数据训练及多 NPU 训练未在本次运行覆盖。代码与配置接口已留好，使用方法见
`NPU_README.md`。
