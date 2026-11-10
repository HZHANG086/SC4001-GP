# Member C: audited statistics, error analysis and CKA handoff

本目录完成 C 的前三项工作：接收并核验 A/B 交付、统计与错误分析、准备 B 可运行的 CKA 与统一成本分析包（v2）。
**不包含 LoRA 训练，也不包含尚未执行的真实模型 CKA 结果。** A-v1/B-v1 原始代码、配置、预测和结果保持不变。

优先阅读：

- [结果报告（中文，含图）](results/C-v1/REPORT_ZH.html)
- [交给 B 的运行说明](HANDOFF_B_ZH.md)
- [分析协议与限制](protocol/C-analysis-v1.json)
- [全量核验](results/C-v1/audit.json)、[公开测试文本核对](results/C-v1/text_join_audit.json)
- [统计表](results/C-v1/paired_comparisons.csv)、[错误转移](results/C-v1/error_transitions.csv)
- [测试日志](results/tests.log)

## 范围与状态

| 工作 | 状态 |
|---|---|
| A/B 源提交固定 | `f442a4462364ce0c53cbae45febbfcfc8238988c` |
| 数据划分、冻结代码、源验证选择和训练记录检查 | 已执行；285 次候选训练记录 |
| 逐样本预测复算 | 已执行；450 个预测文件、5,250,000 行；包含 λ 消融 |
| 最终统计样本 | 90 组选定模型；候选模型不充当额外随机种子 |
| 测试原文及标签 | 从固定 Hub 版本重建 35,000 条，ID/标签一致 |
| 权重 | 未下载、未验证实际文件可访问；已核对 B 的 60 个路径与选择清单 |
| CKA | 脚本通过同版本库的小模型端到端测试；正式 TC1 推理待 B 执行 |
| 人工案例标注 | 提供 AI 辅助解释草稿，仍需组员复核；不是人工金标准 |

A 的归档没有逐运行的 `COMPLETE` 和 `environment.txt`；已有全局环境记录、每次配置/完整历史/验证结果和最终核验文件。检查器明确记录此归档差异，没有补造标记。

## 复算统计与图表

以下命令从仓库根目录执行，使用独立 Python 环境。不要在 TC1 登录节点训练或执行 GPU 推理。

```bash
python -m pip install -r member-c/requirements-analysis.txt
PYTHONPATH=member-c/src python -m sc4001_c.analyze
PYTHONPATH=member-c/src python -m sc4001_c.plots
```

原文复建会下载公开测试 parquet 和分词器，不会下载模型权重。缓存放在仓库外。

```bash
PYTHONPATH=member-c/src python -m sc4001_c.texts --cache ../c-test-cache
PYTHONPATH=member-c/src python -m sc4001_c.plots
PYTHONPATH=member-c/src python -m sc4001_c.report
```

完整文本不提交；案例交付文件仅保留短摘录、稳定 ID、文本 SHA256 和指标。重建缓存用于检查原文。复建脚本使用相同 IMDb/Yelp/Amazon 固定版本、A 的测试索引，以及 Amazon `title + newline + content` 拼接方式。

```bash
PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src python -m unittest discover -s member-c/tests -v
```

## 如何读统计结果

1. 主问题是每个预算下 M6−M3 的跨域差值；统计单位为 5 个配对种子。
2. 图中 paired-t 95% 区间是近似且未校正的区间；5 个种子不足以可靠检查分布假设。
3. 同时给出精确符号翻转检验。只有 32 种符号组合，双侧最小 p 值为 0.0625；这不意味着各方法等效。
4. focal 3 个比较和 secondary 69 个比较分别做 Holm 校正。该分析在看到 A/B 结果后制定，**不是事后补写的预注册验证性实验**。
5. bootstrap 保留同一测试样本在不同模型/种子之间的配对。固定模型条件区间与种子×样本交叉重采样区间不同；都不能当作更多独立训练。
6. 同时展示有利与不利的结果。95% 区间跨零不证明零效果；bootstrap 与 t 区间结论不同则披露方法敏感性。

## 成本修正

`cost_accounting.csv` 同时列增量搜索和独立复现整条流程的搜索成本。M6/L2SP 继承 M3 搜索；M4 继承 M1 搜索。共享上游工作不能重复加进整个项目的实际总耗时。

`M6-b6400-s21-lam1` 在作业 72808 超时，72909 续跑。其 `validation.json` 仅有最后一次调用时间，故该 cell 的搜索成本是下界；最终选中的是 lam0.01，不能错误地说它本身续跑了。B 作业账本总分配时长 56.564 小时，包含评测、启动等，不等同纯训练时间。

B 的 M3/M4/L2SP 在训练中保留了分析用 teacher，显存为当前实现的测量，不能直接声称是算法最低需求。**已补齐统一短程成本测试脚本，真实 GPU 测量仍待 B 执行。** 新结果将作为单独的成本探针，不能替换原完整训练测量。

## v2 完整运行与回传

请按 [交接说明](HANDOFF_B_ZH.md)运行 `start_bundle.sh smoke`，通过后运行 `start_bundle.sh full`。正式链包含 15 个 CKA cell（四种方法 × 三个预算 × 五个种子，共 60 个模型），再加三轮成本测试（27 个独立进程）。每项成功才提交下一项；失败停止。

最后运行 `start_bundle.sh package`，返回 `member-b/analysis/C-return-v2.tar.gz`。收集器检查 8,190 行 CKA、180 项预测复核及 27 次成本测试。真实结果返回前，不制作真实 CKA/成本结论。v1 输出不能混入 v2。

独立收集 CKA 的命令（仓库根目录）：

```bash
PYTHONPATH=member-c/src python -m sc4001_c.collect_cka \
  --input member-b/analysis/C-handoff-v2/cka --out member-c/results/CKA-v2
```

完整新增范围、成本口径与排除项见 [v2 协议](protocol/C-handoff-v2.json)。原分析报告已同步描述 v2；统计结果不变。LoRA 若继续开展，仍是独立的新训练任务。

## AI 使用与归因

本部分由 Codex 协助编写核验、统计、可视化和 CKA 代码，并生成解释草稿。已进行记录复算、数学单元测试、小模型端到端测试和图表检查。真实 TC1 推理仍未执行。团队需复核案例、解释和最终报告；不得将 AI 草稿写成组员已人工核验的事实。

模型加载复用 A 的 `LinearRoberta`，源数据与预测来自 A/B；本部分新增审计、配对统计、错误分析和表示比较。CKA 依据 [Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html)；统计实现对照 [SciPy paired permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) 验证。源码以本项目贡献为准，不复制外部实现。
