# C 工作入口（2026-11-11）

**当前尚未全部完成。** 原 `README.md`、`DELIVERY_MANIFEST.json` 和 `results/C-v1` 属于已冻结的历史交接版本，保留原样以便验证哈希。它们关于“尚未运行 CKA / 不包含 LoRA”的描述不是当前进度。

## 已完成与真实缺口

| 项目 | 当前证据 |
|---|---|
| 作业要求和课件映射 | `docs/requirements_audit.md`；原课程手册优先于历史方案 |
| A/B 预测、统计与错误分析 | `results/C-v1`；原始记录不变 |
| B 回传的真实 CKA 与 V100 成本 | `formal-results/return-v2`；244 个文件哈希匹配，60 个模型，180 次预测复核，27 次成本探针 |
| 六方法综合分析和正式风格图 | `results/C-final`、`report/figures`；不是包含 M5 的最终版 |
| C 学术正文草稿 | `report/latex/main.tex`；Arial 10pt、A4、单栏，黑色正文和彩色科研图 |
| M5 LoRA 训练代码 | `src/sc4001_c/lora*.py`，协议 `protocol/M5-v1.json` |
| 无 Git、无权重的独立复算 | `results/standalone_reproduction.json`；450 个预测文件、525 万行记录、七张统计表、21 张综合结果 CSV 及十张彩色/灰度 PNG 完全一致 |
| ZIP 实际交付预检 | `results/zip_preflight/PREFLIGHT.json`；4,132 文件、约 76 MB；Linux 解压后全量预测复算通过。首次舍入差异与失败记录保留；这不是最终 C 包 |
| 本地验证 | 更新后的完整 37 项 CPU 测试通过，受测代码哈希与当前代码一致，见 `results/validation-37-tests`。覆盖 LoRA、诊断、记账服务关闭和安全回传测试。小模型测试不代表正式实验已完成 |
| M5 真实 GPU 实验 | 真实 GPU 两步验证通过；前两批成功结束，12374 第三批正在训练；当前 19/45 个候选完成、0/15 个正式评测完成；见 results/Ada-live-status.json |
| 最终报告与交付包 | 待 M5 完成后合并和验收 |
| 往届公开项目检索 | 按用户要求，在第一版完整交付之后开展，目前尚未到该阶段 |

## 复算现有结果

以下命令从仓库根目录执行；使用 `requirements-final.txt` 所列完整分析环境。无 Git 历史的 ZIP 复现步骤见 `docs/REPRODUCTION.md`；主报告分工、术语与页数整合见 `docs/REPORT_INTEGRATION.md`。

```bash
export PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src
python -m sc4001_c.final_analysis
python -m sc4001_c.final_plots
python -m sc4001_c.final_plots --grayscale --out member-c/report/figures-grayscale
```

第一条会核验回传清单，并从原始 CKA 和成本结果重新收集统计。历史 C-v1 的复算方式保留在原 README；不需要 B 的 29GB 权重来重算现有表格。再次执行真实 B 模型推理才需要对应权重。

## Ada 运行安排

所有 GP 新工作在 `/share/zhangh/sc4001-gp-c`；原 Assignment 环境不改动。数据由 A 的冻结预处理器和固定版本生成，19 个协议/抽样/审计文件与 A 完全一致；原始 RoBERTa 权重已缓存。M5 从预训练基座开始，不继承 B 微调权重。

- `12365`：未启动的数据准备任务，因本地准备并同步成功而取消，不是实验失败。
- `12366`：GPU 两步验证成功；最大合并差异 4.47e-8。
- `12367`：第一批正式训练完成；实时调度记录显示 COMPLETED、ExitCode 0:0，运行 47 分 30 秒。九个正式候选的配置、十轮训练、参数量和权重哈希均复核通过；合并最大误差 9.42e-6。
- `12369`：第二批（seed 21）成功完成；九个候选复核通过，合并最大误差 1.28e-05。
- `12374`：第三批（seed 42）已自动启动。
- 正式训练按五个种子分批，每批 9 个候选；全部训练完才锁定 15 个选择，再统一测试。
- 最后一批自动提交 `diagnostics.sbatch`，完成 M5 预测核验、CKA 和 21 次同硬件训练成本探针。
- 任一批失败，后续链停止；先保留日志、检查完整恢复点，再继续。不能删除不完整证据后假装首次成功。

固定配置和步骤见 `docs/M5_RUNBOOK.md`。训练任务在 Slurm 中持续存在，不依赖聊天窗口保持打开。当前 SSH 控制连接会过期；重新连接属于后续监控条件，不影响已经提交的 Slurm 作业。

## 结论与写作边界

M6 提高了平均池化表示的 CKA，但相对 M3 的 OOD 准确率差值分别为 -0.702、+0.694、+0.038 个百分点，种子间不稳定。不可写成“表示更接近，所以泛化一定更好”。V100 探针显示 M6 的额外训练开销，但该短程结果不能冒充完整训练或已训练模型的部署测试。

所有草稿中 M5 的状态必须保持明确，不能填入估计的准确率。案例中的 AI 解释需要组员复核；不要将未发生的人工审核写成事实。组名/姓名/学号由用户暂缓提供，不编造。

服务器未启用 Slurm 历史记账。已保留实时状态快照，并为后续批次增加开始/退出留档；打包时保留记账失败证据，不伪造分配时长。此运行层修正未改变训练或分析方案；原诊断协议存于 `protocol/M5-diagnostics-v1.initial.json`。

服务器已补装 SciPy 1.15.3，供训练后的统计审计使用；采用不更新其他依赖的安装方式，训练库版本不变。记录见 `results/Ada-scipy-install.txt`。各候选环境清单保留实际状态，不回填早期尚未安装的包。

第一批验收见 `results/Ada-stage0-check.json`；终态调度快照、实际退出日志和下一批启动记录见 `results/Ada-live-evidence/`。这些是实际 GPU 训练证据，不代表尚未执行的目标域测试。

M5 回传验收还会从已核验预测独立重算 18 组配对统计，并将验收记录绑定到完整回传清单；最终分析会重新检查全部回传文件哈希。实现及合成测试已通过，真实 M5 回传尚未到达，不能把测试结果当作正式模型结果。
