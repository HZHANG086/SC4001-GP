# 给 B：完整补充运行包 v2

## 这次要完成什么

你只需要用现有环境和权重运行这个包。A/B 原来的准确率、训练代码、模型选择不改；不用把 29GB 权重发给 C，也不用重新训练全部模型。

| 内容 | 规模 | 用途 |
|---|---|---|
| 表示分析与预测复核 | M3/M4/M6/L2SP × 3 个预算 × 5 个种子，共 60 个选中模型 | 补齐强基线与机制对照；检查真实 checkpoint 与归档预测一致 |
| 逐层表示比较 | 每个模型三域各 256 个固定样本，13 层、CLS/mean 两种池化 | CKA、去偏 CKA、乱序控制、余弦距离 |
| 干净实现的成本测试 | M1/M2/M3/M4/M6/L2SP，每种 3 次独立进程 | 训练显存、每步耗时、吞吐、学生模型推理开销 |
| 原实现诊断 teacher 对照 | M3/M4/L2SP，每种 3 次独立进程 | 量化额外 teacher 占用，保留原始成本记录并解释差异 |
| 自动检查、汇总、打包 | 8,190 行 CKA、180 项预测复核、27 次正式成本测试 | 小文件一次回传，C 接手画图、分析和写报告 |

这是**现有六方法实验的补充运行全集**，不包含新方法 LoRA 训练、A 持有的 M1/M2 checkpoint 表示分析或重新搜索超参数。已有完整测试预测和 λ 消融记录由 C 在本地分析，不重复占用 GPU。LoRA 是另一个需要独立协议的新训练任务，不能冒充本包已经完成。

## 1. 下载和放置

推荐用 C 发来的 **`SC4001_B_complete_run_package_v2.zip`**。解压后只有 `member-c/`，把它放到你现有项目根目录，与 `member-a/`、`member-b/` 并列。已有旧版 `member-c/` 可先改名备份，再放入完整新版，避免混合两版脚本；原 `member-b/analysis/cka-C-v1` 输出保留。

GitHub 同一版本在 `member-c-analysis-v1` 分支的 `member-c/` 目录。**不要为了取 C 代码覆盖你的 B 工作目录或权重。**不必下载整个仓库；ZIP 已包含全部 C 代码和说明。

需要你原来的文件：

- `member-b/formal-results/selected_models.tsv` 和归档的选择记录、测试预测。
- `member-b/runs/B-v1/*/best/` 中 60 个选中模型的权重、配置和 tokenizer，以及父目录 `run_config.json`。
- `member-b/artifacts/data-v1/` 数据和 READY、sources、splits、manifest。
- `member-a/remote-records/data-v1/` 的参考哈希，以及 A/B 的 `protocol/locked_src/` 和原激活脚本。
- 本地缓存的固定版本 `FacebookAI/roberta-base`，revision 为 `e2da8e2f811d1448a5b465c236feacd80ffbac7b`。

现有 `sc4001` 环境即可：torch 2.5.1、transformers 4.51.3、datasets 3.5.0、numpy 1.26.4。**不需要安装 scipy、matplotlib、peft，不需要新模型或新数据。**预检查发现版本或文件不符会报出缺项；先发报错，不要随意升级依赖或换模型版本。

## 2. 先提交 smoke

登录 TC1，进入你现有项目的 **`member-b` 目录**。以下所有命令都在这个目录执行。

```bash
bash ../member-c/scripts/start_bundle.sh smoke
```

先检查版本、60 个权重路径、固定数据和模型缓存，再申请 GPU。登录节点只做文件检查，不做推理或训练。

smoke 会运行 budget 400 / seed 13 的四种 B 模型，并对六种方法各做一次缩短成本测试。成功后会出现：

```text
analysis/C-handoff-v2/SMOKE_COMPLETE
```

用 `squeue -u "$USER"` 看任务；用 `logs/c_bundle_<jobid>.out/.err` 看日志。确认 smoke 作业结束、标记存在且无异常，再运行下一步。弃用旧的 `cka.sbatch 13 full` 命令；v1 与 v2 结果不能混用。

## 3. 一条命令启动全部正式任务

```bash
bash ../member-c/scripts/start_bundle.sh full
```

这会提交第 0 项，之后每项成功才提交下一项，不需要一直开着终端。链中最多一个当前任务和一个等待任务；启动器要求账号没有其他排队/运行作业，避免超过两个提交名额。运行期间请不要再为同一账号启动另一条任务链。

| 阶段编号 | 任务 |
|---|---|
| 0–2 | seed 13，依次 budget 400 / 1600 / 6400 |
| 3–5 | seed 21，同上 |
| 6–8 | seed 42，同上 |
| 9–11 | seed 87，同上 |
| 12–14 | seed 100，同上 |
| 15–17 | 成本测试第 1 / 2 / 3 次，每次 9 个独立进程 |

每个 CKA 任务只加载四个选中模型，按顺序运行，减小超时风险。每项申请 1 张 GPU、24GB 主机内存、5小时50分钟上限。**这是排队资源上限，不是预计耗时；真实 TC1 时间尚未实测。**用 smoke 的 `run.json` 估计一个 CKA cell 的时间，正式成本测试比 smoke 多一些计时步。

## 4. 如果失败或队列没有继续

失败不继续提交后续作业，不覆盖权重；输出保留。请发对应日志和 `analysis/C-handoff-v2/jobs.tsv` 给 C。若服务器不允许在计算节点调用 `sbatch`，已完成结果仍有效，可从登录节点手动接续。

排除问题、确认账号没有正在运行/排队的旧链任务后，例如第 7 项失败或没有成功提交：

```bash
bash ../member-c/scripts/start_bundle.sh resume 7
```

恢复编号用失败日志或 `jobs.tsv` 确认；`NEXT_STAGE` 只提示上次计划提交项，不是可靠的失败判定。完成且输入/代码一致的 cell 会跳过；未完成 cell 会重算，不混合不完整激活。代码或输入不同会拒绝沿用旧结果，先反馈，不要删除校验来强行运行。

不要反复提交 `full` 或并行跑多个 `resume`。如某个 CKA cell 本身超过时限，请把日志发来进一步拆分，不要无限重试。

## 5. 完成后回传这一个压缩包

全部正式任务完成后，最后一项会自动核验和打包。回到登录节点、确认队列已经结束，再运行一次打包，纳入最终完整日志和作业账本：

```bash
bash ../member-c/scripts/start_bundle.sh package
```

请发回：

```text
member-b/analysis/C-return-v2.tar.gz
```

也可以把压缩包上传到你们已使用的共享位置，再给 C 下载链接。无须发送权重、缓存数据或隐藏层大数组。

包内应有：

- `C-handoff-v2/cka/b<budget>-s<seed>/`，共 15 个目录；每个包含 `cka.csv`、`prediction_replay.csv`、`manifest.json`、`run.json`、`COMPLETE`。
- `C-handoff-v2/cost/r<repeat>-<method>-<profile>/`，共 27 个正式目录；每个包含 `manifest.json`、`result.json`、`COMPLETE`。
- `collected/cka_all_seeds.csv`（8,190 行）、`prediction_replay_all.csv`（180 行）、`cost_all_repeats.csv`（27 行）、`cost_summary.csv`（9 行）及汇总检查 JSON。
- smoke 输出、预检查记录、`jobs.tsv`、可获得的 SLURM 账本、SHA256 清单和对应日志。

打包器会拒绝缺项、不同配置/样本、错误种子、重复行、失败预测复核和合成测试结果。**只收到某个 COMPLETE、截图或平均分不算完整交付。**暂时失败也可以先把整个 `analysis/C-handoff-v2/` 和 `logs/c_bundle_*.out/.err` 发来排查。

## 6. 实验口径（C 写报告时使用）

- CKA：三域各固定 256 条（正负各 128），同一输入 ID/顺序用于全部模型。四种方法各对比 pretrained，M4/M6/L2SP 另对比同预算同种子的 M3，共七类比较。FP32 eval、关闭 TF32；相似度用 CPU float64。固定乱序是诊断控制，不是显著性检验。第 0 层恒定 CLS 的 CKA 记 NA。
- 预测复核：每个模型、每个域对照归档样本概率，最大绝对差异须 ≤1e-4。测的是固定子集的 checkpoint 一致性；已有全量预测已另做审计，不声称本任务重新跑了全量测试集。
- 成本：统一使用 IMDb **训练集**中固定 32 条样本，padding/truncation 512，microbatch 8、梯度累积 4、FP32；2 个优化器步预热，5 个优化器步计时。输入预先分词并驻留 GPU，计时不包含分词、数据读取或 CPU→GPU 传输。每个 profile 独立进程，三轮轮换运行顺序；训练前重置随机数，使 clean/teacher 对照的 dropout 起点一致。记录预热/计时显存峰值、逐步时间、参数量、推理显存和吞吐。
- 所有成本方法从相同 pretrained 编码器和相同轮次的随机分类头开始；学习率固定 2e-5，M6 λ=.01、L2SP λ=.1。这些是固定成本探针，不是重新选超参数。M4 测的是第二阶段计算代理，第一阶段计算由 M1 代表，**没有重建完整 LP-FT 的初始化或性能**。
- clean M6 保留实际参与损失的 frozen teacher；clean L2SP 保留 pretrained 权重参考。M3/M4/L2SP 另加一个只驻留、不参与训练损失的 teacher，对照原 B 实现的额外诊断显存。两种 profile 都测学生模型单独推理。
- 短程成本与原完整训练的动态 padding、验证、保存、调度开销不同，必须单列。不能把短程显存写成原训练修正值，不能用每步时间直接宣称完整项目耗时，三轮进程重复不是三个新的训练种子。M4 两阶段时间需分别核算，不能把两阶段显存峰值相加。
- 本包未写回原模型、未生成新准确率结果、未用目标域测试挑参数。现阶段代码在 CPU 小模型与模拟队列上验证；真实 GPU 数值由这次运行取得。
