# NPU inference

## 性能采集配置

```bash
bash scripts/run_inference_npu.sh inference/configs/inference_npu_smoke.yaml
```

smoke 配置按数据集顺序跑 single_test 的前 8 条，每条 1 次 rollout，保留正式
推理的模型、block length、去噪步数和生成长度上限。prefill / denoise 各自先预热
2 次 forward，再分别采集 5 次，之后继续完成这 8 条。若实际执行次数不足，
对应阶段只能产生已执行次数的 trace；性能采集开销不能当作端到端吞吐。
输出位于启动目录的 `res/<datetime>-inference_npu_smoke/`。

完整测试仍使用 `inference_npu.yaml`（max_samples=0），输出在
`res/<datetime>-inference_npu/`。其中 `single_test.jsonl` 每完成一条 rollout 就写入并
flush，包含逐条推理结果；`run.log` 保存终端输出和最终计数，`config.yaml` 和
`commands.txt` 保存启动配置与命令，`prof/` 保存 trace。

## YAML 一键启动

编辑 `inference/configs/inference_npu.yaml` 中的 Python、CANN、模型和数据路径，然后：

```bash
bash scripts/run_inference_npu.sh inference/configs/inference_npu.yaml
# 只查看实际命令，不加载模型或启动 NPU：
bash scripts/run_inference_npu.sh inference/configs/inference_npu.yaml --dry-run
```

启动脚本使用 `${PYTHON:-python3}` 解析 YAML，需要安装 PyYAML；真正推理使用
YAML 的 `runtime.python`。`runtime.cann_env` 会自动 source，已加载环境时可设为 null。
配置中的相对模型、数据路径基于仓库根目录。YAML 启动结果固定写入启动命令时
当前目录的 `res/<YYYYMMDD_HHMMSS_ffffff>-<config文件名>/`。
无需配置顶层 `output_dir`；旧配置须删除该字段。每次运行新建目录，保存
`single_test.jsonl`、`config.yaml`、`commands.txt`、`run.log` 和 `prof/`。
prof 相对路径基于本次结果目录。每次新建目录意味着 `resume` 不会续跑历史目录；
若要续跑已有结果，可直接使用目录内 commands.txt 中的命令并加 `--resume`。
`runtime.env` 可设置工具路径、`ASCEND_RT_VISIBLE_DEVICES` 等环境变量。
工具路径默认相对于仓库。外部安装可通过 `runtime.env` 中的
`RXN_T5CHEM_ROOT`、`RXN_T5CHEM_MODEL_PATH`、`UNICORE_ROOT`、
`CHEMTOOL_AGENT_ROOT`、`CONTROLLED_CHEMICALS_CSV` 覆盖；已有容器环境变量也会保留。
YAML 中的模型、数据、CANN 绝对路径属于部署配置，迁移容器时需按实际目录修改。
正式全量推理将 `inference.max_samples` 设为 0，按需关闭 `profiling.enabled`。

`profiling.phase` 可选 `both`、`prefill`、`denoise`、`cache_commit` 或 `all`，默认
YAML 选择 `both`，一次完整推理中分别采集 prefill 和去噪 decode，各阶段独立
预热、计数和保存，不采 cache commit。每个阶段只计对应的 forward，先运行 skip_first + warmup
次而不记录，再采 active 次。每次 active forward 独立启停 profiler、同步 NPU，
保证其他阶段不会混入 trace；因此会产生采集开销，不适合衡量端到端吞吐。
结果分开保存在 `prof/<dataset>/<phase>/forward_<index>/rank_0/`。
这些独立 trace 的 native warmup 为 0；YAML warmup 表示未记录的模型预热次数。
`both` 模式无需重复运行数据集；也可用 prefill / denoise 单独采集。数据仍完整运行。
prefill 次数按 assistant turn 计，短于一个 block 的 prompt 不产生 prefill。
`all` 保留连续窗口及混合 forward 计数行为。

## 环境变量启动（兼容旧用法）

使用与 SFT 相同的 PyTorch / torch-npu / CANN 配套环境，先加载服务器的
CANN 环境。模型须为 SDAR checkpoint（包含 tokenizer 文件）。

```bash
MODEL_PATH=/path/to/checkpoint DATA_DIR=/path/to/data \
CONDA_ENV=chemdiffagent DEVICE=npu:0 MAX_SAMPLES=1 \
bash scripts/run_inference_npu.sh
```

NPU 入口直接加载仓库 `training/sft/models/sdar`，复用 SFT 的
`npu_rms_norm`、`npu_swiglu`、`npu_fusion_attention` 替换。
Attention 保留 block diffusion 的可见性 mask、GQA 和 KV cache 语义；
prefill、去噪以及 cache commit 均走这套算子。暂未切换到 FIA。
显式请求不可用的设备会报错。反应工具 T5Chem 默认使用 CPU，可通过
`RXN_T5CHEM_DEVICE` 单独配置。

NPU 脚本默认开启采集，`PROFILE=0` 关闭。默认跳过 0 次 forward、预热 1 次、
采集 2 次，然后立即停止。通过 `PROFILE_SKIP_FIRST`、`PROFILE_WARMUP`、
`PROFILE_ACTIVE` 调整。一个 step 是一次完整模型 forward，工具调用不计 step。
建议 smoke run 保证至少完成 warmup + active 次 forward。
CPU/NPU 活动采用 Level1 + PipeUtilization，默认结果位于
`outputs/inference/prof/<dataset>/rank_0/`，含版本及 schedule metadata。
重复运行建议设置不同的 OUTPUT_DIR。异常退出也会关闭 profiler。

直接运行 Python 时使用 `--device npu:0 --profile`；可通过 `--profile-dir`
指定目录，相对路径基于输出 jsonl 所在目录。

本地 CPU contract tests 不代表原生 NPU 性能或数值验证。服务器上仍需完成
上述 smoke run，检查 trace 中三个 NPU 算子、生成结果及多 block KV cache 行为。
