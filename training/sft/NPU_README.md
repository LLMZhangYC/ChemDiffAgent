# SDAR 一层 SFT / NPU

本实现继续使用 `train/sft_sdar.py`、SDAR 模型类、Accelerate 和 DeepSpeed ZeRO-3。
模型权重先离线抽取，再交给原 SFT 入口训练。项目、输出和缓存位于 `/data/hd`；
复用服务器上已有的 Python 环境与 CANN，不安装或修改其依赖。
下列命令均在 `zyc` 容器中执行。

## 配置与启动

主配置：`/data/hd/ChemDiffAgent/training/sft/configs/chemdiffagent_sft_npu_1l.yaml`。

1. 导入公开权重后，将 `model.reduction.source_model` 设为本地 checkpoint 目录。
   必须包含 SDAR 的 `config.json`、匹配的 tokenizer，以及单文件或分片 safetensors。
2. 将 `model.pretrained_model` 设为一个尚不存在的输出目录（默认 `/data/hd/models/SDAR-1L`）。
3. 抽取权重：

```bash
bash /data/hd/ChemDiffAgent/scripts/train_sft_npu.sh runtime.action=export
```

也可以用 OmegaConf 参数覆盖路径，例如：

```bash
bash /data/hd/ChemDiffAgent/scripts/train_sft_npu.sh \
  runtime.action=export \
  model.reduction.source_model=/data/hd/models/SDAR-8B-Chat
```

4. 根据可用设备设置 `runtime.visible_devices`，运行训练：

```bash
bash /data/hd/ChemDiffAgent/scripts/train_sft_npu.sh runtime.action=train runtime.visible_devices=0
```

可通过 `config=/绝对路径/自定义.yaml` 指定另一份完整配置。
`runtime.visible_devices=null` 时继承外部 `ASCEND_RT_VISIBLE_DEVICES`。
`runtime.num_processes` 控制进程数；启动器将训练配置中的梯度累积和裁剪值写入本次
Accelerate 配置，防止两份配置不一致。不要直接用普通单卡 Accelerator 启动 NPU 路径：
原训练循环依赖 DeepSpeed 处理梯度累积、裁剪和优化器更新。

现有环境默认是 `/data/workspace/zyc/ChemDiffAgent/envs/sft/bin/python`。
如更换环境，同时设置 shell 的 `CHEMDIFF_PYTHON`（启动器解释器）和配置中的
`runtime.python`（训练解释器）。CANN 路径由 `runtime.cann_root` 控制。
默认 wandb 为离线模式，可设置 `wandb.enabled=false` 完全关闭记录。

## 减层含义和权重读取

`model.reduction.num_hidden_layers: 1` 保留 `model.layers.0.*`，以及
`model.embed_tokens.weight`、`model.norm.weight`、`lm_head.weight`。
不缩小 hidden size、词表、注意力头数或 MLP 维度，不随机补齐缺失权重。

抽取器 `tools/export_reduced_sdar.py` 在 CPU 上读取索引，按参数名选择张量；
不会实例化完整模型，不打开只含后续层的 shard，也不通过整文件 `load_file` 后筛选。
同一 shard 内混有后续层时，只对保留参数调用 `get_tensor`。支持带索引的分片
safetensors 和 `model.safetensors`；不支持 `.bin`、量化权重或滑窗 SDAR 变体。
源权重保持不变，目标已存在时明确报错。

输出包含匹配的 tokenizer、局部 NPU 兼容的 SDAR 实现、更新后的模型配置、权重索引，
以及记录读取 shard/参数名/字节数的 `reduction_report.json`。
训练先读取配置检查层数，再实例化模型；误传完整 checkpoint 会提前失败。

## 8192 的定义与数据行为

`training.max_seq_length: 8192` 限制实际送入模型的、包含 padding 的长度：

```text
P + R + noisy_R = P + 2R <= 8192
R_budget = min(max_gen_length, floor((8192 - P) / 2))
```

当前数据管道使用整个数据集共享的 prompt 宽度，故 P 为 padding 后的宽度。
裁剪发生在构造 padded response tensor、加噪副本、位置索引和 attention mask 之前。
未配置 `max_seq_length` 时保留原 prompting 路径。

- prompt 超过 `max_prompt_len` 时沿用原来的丢弃策略，不截掉系统提示。
- response 保留 token 前缀；如果截断发生在 `<tool_response>` 内，回退到该段起点。
- 完整工具响应仍作为上下文，不作为预测目标；原 block attention 和 loss 归一化保留。
- 无有效 response 监督的样本被过滤；全部无效时明确报错。
- 过滤索引同步应用到样本和 `step_map`，防止样本错位。
- attention mask 使用等价的 bool 0/1 表示，避免 int64 方阵的额外内存。

运行目录的 `preprocessing_report.json` 记录实际 P、R、展开长度和裁剪/过滤数量；
`training_report.json` 记录层数、设备、micro step、optimizer step、最后 loss 和 NPU 峰值内存。

## 数值与验证范围

NPU 使用 PyTorch RMSNorm、普通 SwiGLU 和携带原自定义 mask 的 SDPA；CUDA 分支保留原算子。
原 CUDA 单 token / KV-cache 分支没有向 FlashAttention 传入自定义 mask，NPU SDPA 路径会遵循
传入的 mask；带屏蔽位置的推理场景需另行对齐。这次 SFT 关闭 cache，不进入该解码分支。
跨设备不承诺逐位相同。`tests/verify_npu_reference.py` 对随机小型一层模型进行
CPU FP32 / NPU BF16 的 logits、loss、梯度和带 mask 的 KV cache 解码对照；
它不替代 CUDA FlashAttention 对齐或真实公开权重的效果评估。

CPU 测试入口（使用已有 SFT 环境并设置本地缓存）：

```bash
export HF_HOME=/data/hd/ChemDiffAgent-work/cache/hf
export TMPDIR=/data/hd/ChemDiffAgent-work/tmp
export PYTHONDONTWRITEBYTECODE=1
/data/workspace/zyc/ChemDiffAgent/envs/sft/bin/python \
  /data/hd/ChemDiffAgent/training/sft/tests/test_reduction_and_truncation.py
```

`tests/make_smoke_fixture.py --output /data/hd/某个新目录` 生成明确标记为 RANDOM 的
小模型及短序列/8192 序列训练配置，用于执行原 SFT 入口的集成测试。
验证结果和真实公开权重尚未覆盖的部分见 `NPU_VALIDATION.md`。

原代码备份：`/data/hd/ChemDiffAgent-work/backups/before-npu-1l.tar.gz`。
