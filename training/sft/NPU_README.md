# SDAR 一层 SFT / NPU

本实现继续使用 `train/sft_sdar.py`、SDAR 模型类、Accelerate 和 DeepSpeed ZeRO-3。
模型权重先离线抽取，再交给原 SFT 入口训练。每次实验的输出和运行文件放在同一 run 目录；
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
SFT 指标直接写入 `experiment.output_dir/metrics.jsonl`，不再使用 W&B。

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

NPU 使用 `torch_npu.npu_rms_norm`、`torch_npu.npu_swiglu` 和携带原自定义 mask 的
`torch_npu.npu_fusion_attention`；CPU/CUDA 分支保留原行为。
原 CUDA 单 token / KV-cache 分支没有向 FlashAttention 传入自定义 mask，NPU 融合 attention 路径会遵循
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


## Native NPU operator dependencies

The NPU path uses `torch_npu.npu_rms_norm`, `torch_npu.npu_swiglu`, and
`torch_npu.npu_fusion_attention` (training forward/backward, not inference FIA).
Install a `torch-npu` wheel matching the server's PyTorch and CANN versions.
CANN runtime/operator packages and the Ascend driver must also be installed;
`op-plugin` supplies the adapter inside the torch_npu build and is not an extra
Python package to pip-install. No FlashAttention, Liger, Triton, TorchAir,
MindSpeed, or custom Kimi operators are required by these three calls.

Attention uses BNSD with native GQA, `sparse_mode=0`, and an explicit bool mask.
Model masks use True=visible; the wrapper inverts them to NPU True=blocked.
The packed SwiGLU input is `[gate_proj(x), up_proj(x)]`; linear weights and
checkpoint names are unchanged. Missing/incompatible NPU operators raise errors
instead of silently reverting to SDPA.

Source examples: Kimi K3 `models/modeling_dspark_common.py` RMSNorm/SwiGLU;
attention follows the training API rather than Kimi's inference-only FIA.
Run `tests/verify_npu_reference.py --help` for the existing device comparison
entry point. The native operator replacement still requires target-device
forward/backward and peak-memory validation; earlier SDPA results do not certify it.

## Training profiling

The one-layer smoke config enables `profiling.enabled`; other configs default to
disabled when the section is absent. `train/profiling.py` follows the Kimi
Level1/PipeUtilization setup and the training-side optimizer-step scheduling in
`cann-recipes-train/llm_sft/DeepSpec/deepspec/utils/profiler_context.py`
(reviewed commit `e71712609dc70d3046af4b7453b44a515751bd60`).

With `training.max_steps=3`, `warmup=1`, `active=2`, `skip_first=0`, the first
optimizer update warms up and the next two are recorded. All accumulated
micro-batches are included between optimizer boundaries. Only the first epoch
creates a native profiler; it stops immediately after the finite window completes. Later epochs never restart it. If the first epoch ends
early, only a partial window is collected and later epochs do not extend it.
CPU/NPU activities are recorded, and the profiler closes before
checkpoint saving, including on early exit; exceptions also exit the context.

Relative `profiling.output_dir` is anchored to `experiment.output_dir`. Traces
and version/schedule metadata are written to `prof/rank_N/epoch_E/`. Each rank
records independently. The default disables shapes, memory tracing and stacks
to reduce overhead; enable these options only when needed. The trace exporter
does not require TensorBoard for collection.

```bash
bash scripts/train_sft_npu.sh runtime.action=train
# Disable instrumentation for throughput measurements:
bash scripts/train_sft_npu.sh runtime.action=train profiling.enabled=false
# CPU lifecycle/configuration checks (does not validate the native profiler):
python training/sft/tests/test_profiling.py
```

Requires the installed `torch_npu.profiler` and matching CANN profiling support;
no Kimi/DeepSpec training framework is imported. Target-device trace generation
still needs verification. Profiling adds overhead and should be disabled when
estimating end-to-end training throughput.

The NPU recipes disable OpenTelemetry through `runtime.env.OTEL_SDK_DISABLED: "true"`
and `runtime.env.OTEL_METRICS_EXPORTER: "none"`. The launcher forwards `runtime.env`
to child processes, overriding inherited values. These do not disable NPU profiling.


## Full model, exactly three optimizer updates

Use `configs/chemdiffagent_sft_full_npu_3steps.yaml`. It loads the complete 8B
checkpoint with 8 ranks, BF16, ZeRO-3, accumulation 1, max_steps 3, and epoch cap 1.
The original data/sequence limits are preserved; model size and sequence memory
are not reduced. The first update warms up profiling and updates 2/3 are recorded.
A checkpoint is saved before exit. This config does not export reduced weights.

```bash
export CHEMDIFF_PYTHON="$(python -c 'import sys; print(sys.executable)')"
bash scripts/train_sft_npu.sh config=training/sft/configs/chemdiffagent_sft_full_npu_3steps.yaml
```

Check the startup banner for `epochs=1 max_steps=3 gradient_accumulation_steps=1`.
Check the final log for `Reached training.max_steps=3; saving and stopping`.

`chemdiffagent_sft_full_npu_3steps.yaml` additionally
sets `validation.eval_steps: 3`. It runs one held-out evaluation after the third
optimizer update, then saves and exits without evaluating the same state twice.
`chemdiffagent_sft_full_npu.yaml` retains `validation.eval_steps: null` and
`eval_after_epoch: true`, so normal full training validates only at epoch end.


## Checkpoint retention

`experiment.checkpoints_total_limit` must be a positive integer or null (unlimited).
After rank zero successfully saves the model, tokenizer and metadata, retention
removes the oldest direct `ckpt_<integer>` directories by numeric epoch order.
The just-saved directory is always protected. With five epochs and limit 3,
`ckpt_2`, `ckpt_3`, and `ckpt_4` remain. Re-saving an existing directory does not
reserve an extra checkpoint slot. Unnumbered `ckpt`, unrelated directories,
files, symlinks and junctions are excluded. No old checkpoint is removed when
the new save fails; there may briefly be limit+1 checkpoints during saving.
All ranks wait for saving/cleanup before continuing. Existing checkpoint
contents still contain model/tokenizer exports, not full optimizer state.

Run `python training/sft/tests/test_checkpointing.py` for retention tests.


## Experiment directories

All SFT recipes now define a common parent:

```yaml
experiment:
  run_name: chemdiffagent-sft-full-npu-3steps
  run_dir: ./runs/${experiment.run_name}
  output_dir: ${experiment.run_dir}/output
runtime:
  work_dir: ${experiment.run_dir}/work
```

`run_dir` is resolved relative to the directory where the NPU launcher is invoked,
before the subprocess changes cwd. `output/` contains checkpoints, training
reports, split/validation records when enabled, and `prof/`. `work/` contains
resolved launch YAMLs, caches, temporary files and Ascend runtime logs. Existing
artifacts are not moved. Cache storage remains per experiment in this change.

```bash
bash scripts/train_sft_npu.sh config=training/sft/configs/chemdiffagent_sft_full_npu_3steps.yaml
# Change only the common parent:
bash scripts/train_sft_npu.sh experiment.run_dir=/data/hd/runs/experiment-001
```

Reusing a run_name/run_dir reuses its checkpoint paths; choose a new run_name or
run_dir to preserve a separate experiment. No timestamp is added automatically.
For the generic `scripts/train_sft.sh`, use RUN_NAME/RUN_DIR/OUTPUT_DIR to override
the defaults. Checkpoints are exports rather than exact optimizer-state resumes.


## SFT training/validation split

SFT recipes hold out 10% of the input training data for validation. The split is
deterministic and groups identical prompts together, so alternative responses
and trace-expanded states for the same prompt cannot cross the train/validation
boundary. `dataset.max_samples`, when set, applies before splitting.

```yaml
dataset:
  validation_fraction: 0.1  # Set to 0 to disable the holdout.
  split_seed: 10086
  validation_seed: 10087
validation:
  eval_steps: null         # Optional positive interval in optimizer updates.
  eval_after_epoch: true
```

Validation runs on every rank with `eval()` and no gradients, then restores the
previous model mode. Its masking is generated once with a separate fixed seed;
tool responses and padding remain excluded from supervision. Metrics include
the response-normalized loss used for training, masked-token cross entropy, and
masked-token accuracy. Distributed aggregation removes padded duplicate examples.
These are denoising metrics, not generated-answer benchmark accuracy.

The output directory contains `data_split.json` (source indices, seeds and retained
counts), `validation_metrics.jsonl` (one record per evaluation), and validation
history in `training_report.json`. Checkpoints remain saved at epoch boundaries;
validation does not automatically select or overwrite a best checkpoint.
Older external configs without `validation_fraction` retain training-only behavior.
The CPU tests run with `python training/sft/tests/test_validation.py`.


## Local train/validation loss history

SFT no longer imports or requires Weights & Biases. No account, network access,
or binary log reader is needed. Each micro-step appends a `phase: train` record
to `experiment.output_dir/metrics.jsonl`. Every validation appends a
`phase: validation` record to the same file and still writes the existing
`validation_metrics.jsonl`. Logs are closed after each record for immediate reads.

Records include `run_id`, zero-based `epoch`, one-based `micro_step`,
`optimizer_step`, and `loss`. Train records also include `learning_rate`.
Train loss is the mean of rank-local batch losses across ranks; validation loss
retains its existing response-normalized definition and distributed de-duplication.
Only rank zero writes files. Repeated runs in the same output directory append
with distinct `run_id` values; `training_report.json` records the current
`metrics_run_id`. Filter by run_id and phase before plotting.

Existing W&B run files are left untouched. This change does not convert past
binary histories or alter VRPO logging. External YAMLs with obsolete `wandb`
sections are accepted but those sections are ignored by SFT.

```json
{"run_id":"example","phase":"train","epoch":0,"micro_step":3,"optimizer_step":3,"loss":1.15,"learning_rate":0.00001}
{"run_id":"example","phase":"validation","epoch":0,"micro_step":3,"optimizer_step":3,"loss":1.21,"masked_accuracy":0.4}
```
