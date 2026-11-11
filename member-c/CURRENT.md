# C 当前交付入口

更新：2026-11-11（新加坡时间）。C 的既定实验、分析、图表、学术正文、往届公开材料比较和改进均已完成。首个完整 ZIP 与改进后的一键复算均已实际通过：40 项 CPU 测试、577.5 万条预测、22 张综合 CSV、36 个图表导出。最终交付 ZIP 的具体校验值和独立执行记录随交付验收文件保存；不能用历史预检包替代。

## 优先阅读

- `report/C_analysis.pdf`：可并入小组主报告的学术英文 C 正文，A4、单栏、Arial 10pt，黑色正文、彩色科研图；最终页数和实际渲染验收见 `report/FINAL_QA.json`。
- `report/latex/main.tex`：可编辑正文；`report/figures` 和 `report/figures-grayscale`：六张科研图，均有 PDF/SVG/PNG。
- `docs/EXPERIMENT_GUIDE.md`：报告问题、表格、代码和图表的一一对应；含一键复算命令。
- `docs/PUBLIC_COURSEWORK_REVIEW.md`：完成首版后进行的往届检索及已落实改进。
- `docs/FINAL_ACCEPTANCE.md`：验收清单和团队交接边界。
- `docs/REPRODUCTION.md`：无需 Git 历史、无需模型权重的完整预测与统计复算。
- `docs/REPORT_INTEGRATION.md`：与 A/B 合并、十页正文限制、成员与 AI 声明说明。
- `docs/requirements_audit.md`：原课程要求、团队承诺和实际证据映射。

## 已验收的真实证据

| 范围 | 证据 |
|---|---|
| A/B 原始统计 | `results/C-v1`；285 个候选、450 份预测、525 万条记录，原始结果不变 |
| B GPU 补充 | `formal-results/return-v2`；244 个清单文件、60 个模型、8,190 行 CKA、180 次预测复核、27 次 V100 探针 |
| M5 正式训练 | 45 个候选均完成十轮，15 个 IMDb 验证集选择独立复算一致；`results/Ada-stage4-check.json` |
| M5 回传 | `formal-results/M5-return-v1`；809 个清单文件全部校验，45 份预测、525,000 条记录、18 组配对统计独立复算通过；`results/M5-local-audit` |
| M5 诊断 | 1,170 行 CKA、45 次预测复核、21 次七方法匹配 Ada 探针；回传中的 `M5-diagnostics/audit.json` |
| 七方法综合结果 | `results/C-final`；105 个方法/预算/种子单元，9,360 行 CKA，硬件分开统计 |
| 代码验证 | `results/complete_zip_reproduction` 和 `results/wrapper_validation`：完整 40 项 CPU 测试及真实数据复算通过；不把合成测试当作正式实验 |

## 主要结论

M6 相对 M3 的 OOD 差值为 -0.702、+0.694、+0.038 个百分点，种子间不稳定。更高的平均池化 CKA 不代表更好泛化，也不能解释为因果关系。

M5 的 OOD 均值为 91.064%、93.152%、94.056%。同硬件 Ada 探针中，M5 相对 M3 少用约 49.3% 峰值分配显存、每步耗时少约 25.3%；但比 M2 更耗资源。M5 最小预算的标准差为 2.669 个百分点，不能只汇报有利种子或把最高均值当作普遍最优。

## 证据边界与团队后续

所有 M5 配置在测试前锁定，未因测试结果扩充搜索。五个训练种子是统计重复单位，域、层、样本或成本进程不能充当额外种子。短程成本探针不等于完整训练、搜索或训练后部署测量。

训练从固定原始 RoBERTa-base 开始，不使用 B 微调权重。15 个 M5 选中权重保留在 Ada 服务器；交付包不含大权重、缓存和系统字体。恢复连接后的调度历史已过期，完整输出、脚本退出码和运行时快照见 `results/Ada-postrun-recovery.json`；不伪造未观察到的调度终态。原连接中断记录保留作为历史证据。

用户暂缓提供组号、姓名和学号。组员需审阅正文、实际贡献及 AI 使用声明，并将 A/B/C 整合为不超过十页正文的最终报告；没有发生的人工审阅不得写成事实。本工作不包含向课程系统提交或替组员发送消息。

原 `README.md`、`DELIVERY_MANIFEST.json`、`results/C-v1` 和标有 DRAFT 的旧 PDF 均为冻结历史版本，不能替代此入口。A/B 冻结代码与历史 C 清单保留不变。
