# 最终完成状态（2026-10-28）

A 的90次候选训练、30个结果单元及全部评测已经完成并本地核验。先读 RESULTS_ZH.md；原始完整记录在 formal-results/。以下运行说明保留作为复现与历史记录，不要重复提交已完成的作业。

# 数据与基础方法

状态：这是实验代码与操作说明。真实数据审计和 GPU 结果以远程运行记录为准；不得把测试通过写成模型效果已验证。

## 范围与接口

- 主模型 `FacebookAI/roberta-base`，下载时记录确切 commit SHA；同一个 CLS 线性分类头。
- M1：冻结整个 RoBERTa，只更新线性头，冻结部分关闭 dropout。
- M2：只更新最后两层及同一线性头，其余部分冻结且关闭 dropout。
- 不包含 B 的 M3/M4/M6 或 C 的 LoRA；他们可复用本项目的数据、线性头和运行目录接口。
- 正式预算 400/1600/6400，分别为训练 320/1280/5120、验证 80/320/1280。
- 五个配对种子 13/21/42/87/100。预算内严格平衡；同一种子下训练集与验证集分别嵌套，跨预算也不会互相泄漏。不同种子允许重叠。
- IMDb 官方训练集去除规范化重复、标签冲突文本及与官方测试的文本重合；官方测试 25,000 条保留原样。此清理规则在训练前固定，不使用测试标签或预测。
- Yelp/Amazon 只从官方测试部分固定抽 5,000 条，各类 2,500；规范化去重并排除源领域文本及两目标集合间选中重复。Amazon 固定拼接 `title + 换行 + content`。
- 规范化只用于查重；模型输入保留原始文本。

## 操作

登录节点只做环境、文件与调度操作。数据处理、模型检查、训练及 GPU 查询均放在计算节点。先运行 `bash scripts/diagnose.sh`，根据实际配额修改 sbatch 资源。本包的 partition/QoS 来自提供的手册，必须以账户诊断为准。

实际采用的环境方案：用 conda-pack 标准 tar 从课程共享 CZ4042_v5 完整复制并修正路径，创建个人环境，Python 3.11 / PyTorch 2.5.1 / CUDA 11.8，并把 MKL 修正为 2024.0.0；其他依赖在 pyproject.toml 中锁定。共享环境与手册版本可能不同，先核对实际版本。CUDA 是否兼容必须以计算节点实测为准。无需另装完整 CUDA toolkit。不要在 base 安装依赖。

```bash
bash scripts/rebuild-environment.sh
mkdir -p logs
sbatch scripts/check.sbatch
sbatch scripts/prepare.sbatch
```

若创建过程中失败，先检查 `conda env list` 和报错，不自动删环境。初次复现的 `no such table: shards` 可通过 classic 求解器与独立缓存绕开：`CONDA_PKGS_DIRS="$HOME/.conda/sc4001-pkgs" conda create --solver classic --repodata-fn current_repodata.json -n sc4001-bootstrap --override-channels -c conda-forge python=3.11 pip -y`。本次基础环境保留为 sc4001-bootstrap，慢速官方 torch 2.6.0 下载已停止。共享包缓存已清理，普通 conda 离线 clone 也失败；最终使用 conda-pack 的标准 tar 打包、路径修复及解包方案。no-archive 曾不完整退出，不能使用。每次激活禁用 ~/.local 用户站点包，并优先加载个人环境 lib；共享环境的 MKL 2025 与旧 PyTorch 不兼容，需使用 MKL 2024.0.0。脚本会拒绝覆盖已有 sc4001。

## 流程

准备完成必须有 `artifacts/data-v1/READY`。如果失败，会留下未完成目录；保留日志并把目录改名为失败版本，再重新准备，不能直接当完整数据使用。

```bash
sbatch scripts/train.sbatch --method M1 --budget 400 --seed 13 --lr 0.001 --max-steps 2 --out runs/smoke-M1
sbatch scripts/train.sbatch --method M2 --budget 400 --seed 13 --lr 0.00002 --max-steps 2 --out runs/smoke-M2
```

同一时间只申请账户允许的 GPU 总数。本包不批量提交 90 个任务，也不自动启动全部调参。

正式搜索草案（需三人一起冻结）：每种方法、每个预算、每个种子 3 个候选；M1 LR={0.0001,0.001,0.01}；M2 LR={0.000005,0.00002,0.00005}。每个候选最多 10 epochs，每 epoch 保存与验证，按源验证准确率选检查点（并列取最早），候选并列时按预先列出的 LR 顺序取第一个。AdamW、weight decay=.01、warmup ratio=.1、batch=8、累积=4、FP32。学习率只是预注册起点，需试跑检查学习曲线后冻结，不能声称最优。

长度 256/512 的比较应先在源验证集上独立完成，计入开发成本，再把统一长度传给所有方法；本包默认 256 只是试跑值。不得依据 Yelp/Amazon 测试指标选择长度、LR 或方法。

示例正式候选：

```bash
sbatch scripts/train.sbatch --method M1 --budget 400 --seed 13 --lr 0.001 --length 512 --out runs/M1-b400-s13-lr0.001
```

中断后用完全相同参数及 `--resume` 恢复最新 checkpoint。每 epoch 才保存；首个 epoch 未保存时无法恢复。先完成一次人工中断/恢复检查再扩大实验。`train_wall_seconds_this_invocation` 是本次调用的耗时，恢复多次时必须把各次作业计费时间累计；GPU 峰值是进程统计，不是整张卡的全部占用。

## 测试集开放

先冻结主实验方案、完成所有源验证选择、保存 winner 清单。只对每个单元选中的 winner 运行：

```bash
sbatch scripts/evaluate.sbatch --run runs/M1-b400-s13-lr0.001 --protocol-locked
```

`--protocol-locked` 是操作声明，不能代替团队的预注册记录。smoke 运行禁止生成正式测试结果。测试目录不可覆盖；中途评测失败时保留或改名失败目录再重跑，记录重跑原因。

## 交付格式

`artifacts/data-v1/`：数据版本与模型 SHA、原始行号、15 份固定划分、Yelp/Amazon 固定测试索引、标签统计、长度分位数、256/512 截断比例和清理报告。`audit.json` 包含测试数据分布，只用于固定的数据质量检查，不能据此调模型。

`runs/<method>-b<budget>-s<seed>-lr<lr>/`：

- `run_config.json`：种子、预算、LR、长度、SHA、划分哈希、是否 smoke。
- `parameters.json`：可训练参数总量与完整名称。
- `history.json`、`validation.json`：验证轨迹、最佳 checkpoint、耗时和显存。
- `environment.txt`、`best/`、`checkpoints/`：依赖记录、最佳模型、恢复状态。
- `test/*_predictions.jsonl`：example_id、label、prediction、p_positive；无需复制评论原文。
- `test/metrics.json`：三个领域的绝对准确率和两外部领域准确率的算术平均。

C 可按 `example_id` 配对分析。均值、样本标准差和 M2-M1 差值应按同一预算的五个种子计算；不能把同一种子的候选 LR 当独立重复。每个候选都是实际训练成本；M1/M2 全部 3 候选的主网格是 90 次训练，源验证选中 30 个结果单元。

## 验证与局限

纯 Python 测试：`PYTHONPATH=src python -m unittest discover -s tests -v`。
计算节点模型检查：`python tests/check_model.py`，用随机小模型验证冻结参数、梯度和 dropout，不下载模型，不能替代真实数据 smoke。

没有实现近似语义重复检测；规范化哈希不能发现所有泄漏。官方 IMDb 测试保留重复，报告会披露重复条数。没有实现特征缓存，因此 M1 仍会重复前向传播。主干与分类头虽一致，训练参数不同；按验证集使用不同 LR 是计划中的设计。

来源：用户给定 GPU 手册；https://huggingface.co/FacebookAI/roberta-base ；https://huggingface.co/datasets/stanfordnlp/imdb ；https://huggingface.co/datasets/fancyzhx/yelp_polarity ；https://huggingface.co/datasets/fancyzhx/amazon_polarity ；https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer ；https://pytorch.org/get-started/previous-versions/ 。依赖使用已发布固定版本以控制接口变化，不代表当前最新版。

完成每个单元的 3 个候选后，可在计算节点执行 `python -m sc4001_a.select --length 512` 生成不可覆盖的 `winners.json`。A-v1 已固定为 512。工具只读验证指标，拒绝候选缺失、重复和关键配置不一致；文件列出实际存在的单元，不能把不存在的单元当作已完成。

重建同一数据版本时，用 `python -m sc4001_a.prepare --source-lock remote-records/sources.json --out artifacts/data-rebuilt`；前提是已取得本次 sources.json。实际 pip 依赖另存为 requirements-remote.txt，环境导出记录不能代替数据/模型版本锁定。

## 已冻结并启动的 A-v1 正式协议

统一长度固定为 512。开发依据与限制见 PILOT_RESULTS.md；全部参数和源码哈希在 protocol/A-v1.json。作业使用 protocol/locked_src 的冻结副本，之后修改 src 不会改变已提交实验。

A 共 30 个最终单元，每个单元 3 个 LR 候选，共 90 次训练。第一段 seeds 13/21/42（54 次）为作业 70088；第二段 seeds 87/100（36 次）为 70089，依赖 70088 成功，随后统一选择 30 个 winner、测试并汇总。

```bash
squeue -u "$USER"
sacct -j 70088,70089 --format=JobID,State,ExitCode,Elapsed
```

运行入口是 scripts/formal.sbatch，种子作为参数；最后一段加 --finish。不要在已排队时重复提交。已完成候选会跳过；未完成候选在存在完整 epoch checkpoint 时恢复，否则保留失败目录再重新运行。

完成后的恢复状态会删除；非胜出候选的 best 权重在本单元源验证选择后删除。配置、指标、轨迹、日志保留，30 个赢家的 best 权重保留。清理只针对 runs/A-v1 内本程序生成的文件，避免 90 份完整模型和优化器状态超过空间配额。

最终产物位于 results/A-v1/：winners.json、per_seed.csv、summary.json、summary.md、COMPLETE。逐样本预测位于各 winner 的 test/。五种子均值、样本标准差和 M2-M1 配对差值自动汇总，不自动作显著性声明。若第一段失败，afterok 依赖的第二段不会启动；按候选日志诊断并恢复，不能把等待状态解释成成功。

GPU 恢复测试在 70082 成功：step 2 保存后停止，恢复到 step 4，与不停训练的最终权重在 1e-7 绝对容差内一致。NumPy RNG 恢复使用局部 safe_globals 上下文，仍采用受限的 weights_only 读取。

正式结果完成后，可在 Mac 本地项目目录运行 `bash scripts/collect-results.sh tsun003@10.96.189.11` 收集汇总、配置和逐样本预测；不会复制模型权重。需要有效 SSH/VPN 连接。`best_checkpoint` 记录的是训练时胜出检查点的历史位置，完成后恢复状态会清理；保留模型在 winner 的 best/。
