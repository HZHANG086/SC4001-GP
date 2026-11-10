"""Build a standalone, black-and-white Chinese audit report from measured tables."""
import argparse
import base64
import html
import json
from pathlib import Path
import pandas as pd
from .common import load


def table(df):
    return df.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:.3f}')


def figure(path,caption):
    data=base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('member-c/results/C-v1'))
    a=p.parse_args();out=a.results
    audit=load(out/'audit.json');joined=load(out/'text_join_audit.json')
    paired=pd.read_csv(out/'paired_comparisons.csv')
    focal=paired[(paired.comparison=='M6-M3')&(paired.metric=='ood_accuracy')]
    summary=pd.read_csv(out/'accuracy_summary.csv')
    accuracy=summary[summary.metric=='ood_accuracy'].pivot(index='method',columns='budget',values='mean').reindex(['M1','M2','M3','M4','M6','L2SP']).reset_index()
    errors=pd.read_csv(out/'error_transitions.csv');errors=errors[errors.comparison=='M6-M3']
    error_summary=errors.groupby(['budget','domain'])[['fixed_by_left','broken_by_left','net_change_pp']].mean().reset_index()
    annotations=load(Path('member-c/docs/case_annotations.json'))
    cases=pd.DataFrame(annotations['cases'])
    costs=load(out/'cost_caveats.json')
    fallacies=[
      ['Simpson / 聚合掩盖','分别报告预算、域与类别；不把不同预算混成一个总增益。'],
      ['生态谬误','种子均值不能说明每条评论受益；另列逐样本修正与破坏。'],
      ['选择偏差','固定平衡测试集、单一英语情感任务；不能推广到自然类别比例或所有任务。'],
      ['碰撞变量','未进行控制变量回归；错误类别条件分析仅作描述，不作因果估计。'],
      ['基率忽视','三个测试集均正负平衡；自然部署先验不同，校准指标不直接外推。'],
      ['均值回归','不只挑最佳种子或极端失败案例；展示全部五种子。'],
      ['幸存者偏差','候选训练矩阵完整；记录了超时续跑，不把耗时缺口当作零成本。'],
      ['多重检验','3 个 focal 与 69 个 secondary 比较分组 Holm 校正；区间未调整。'],
      ['分析自由度','C 是事后探索分析，公开全部选择与结果；不是预注册因果检验。'],
      ['相关不等于因果','漂移、CKA、长度和错误关系仅作关联描述。'],
      ['反向因果','没有用相关图确定漂移到准确率的作用方向。']]
    sections=[]
    sections.append('<h1>SC4001 GP：A/B 核验与 C 前三项交付</h1><p class="lead">统计与错误分析已实际执行；CKA 分析包已通过离线测试，真实模型推理待 B 在 TC1 执行。</p>')
    sections.append(f'<p>2026-11-10 · 源提交 <code>{audit["source_commit"]}</code> · A/B 原始记录保持不变。</p>')
    sections.append('<h2>1. 交付与核验范围</h2>'+table(pd.DataFrame([
      ['训练记录',audit['candidate_runs'],'配置、10 epoch 历史、最佳验证 checkpoint 与冻结代码哈希'],
      ['选中模型结果',audit['selected_cells'],'6 方法 × 3 预算 × 5 种子'],
      ['测试记录',audit['tested_runs'],'选中模型 + 预先规定的 λ 消融候选'],
      ['逐样本预测',audit['prediction_rows'],'450 文件；ID、标签、概率、预测和重算准确率'],
      ['原始测试评论',35000,'固定 Hub 版本和测试索引重建，标签全部对应'],
      ['B 模型路径',60,'与选择清单一致；实际权重未下载、尚未加载'],
      ['真实 CKA 输出',0,'尚未运行，不能写入实验结论']
    ],columns=['项目','数量','检查内容'])))
    sections.append('<p>核验的是已归档证据和其内部一致性，不是重新训练验证。A 的逐运行 COMPLETE/environment 文件未归档，已明确记录；A 有全局环境与完整配置历史。训练/验证索引隔离已检查；原始训练文本的去重流程未在本次完整重跑。历史文件内容也不能单独证明当时访问测试集的真实时间顺序。</p>')
    sections.append('<h2>2. 核心结果：漂移减少不等于稳定提高跨域准确率</h2><p>下表是 Yelp 与 Amazon 等权平均准确率（%，五种子均值）。不同域单列结果保存在 CSV 中。</p>'+table(accuracy))
    sections.append(figure(out/'figures/01_accuracy.png','所有方法与三个标签预算；误差线为种子间样本标准差。'))
    sections.append('<p>M2 在 1,600/6,400 标签下的 OOD 均值接近 M3，值得讨论参数与计算成本取舍；这不是正式等效性证明。M4 在较大预算下均值较高，但不能仅凭均值排序宣称统计优势。</p>')
    cols=['budget','mean','ci95_low','ci95_high','p_signflip_exact','p_signflip_exact_holm','p_t_approx_holm','wins','losses']
    sections.append('<h3>M6 − M3 的配对差值（百分点）</h3>'+table(focal[cols]))
    sections.append('<p>三个预算的近似 paired-t 95% 区间都跨零；该区间未做多重比较调整。5 个种子不足以可靠确认差值分布，故同时给出精确符号翻转结果。只有 32 种翻转，双侧 p 最小为 0.0625。不能从“不显著”推出“没有效果”或“完全等效”。</p>')
    sections.append(figure(out/'figures/02_paired_differences.png','空心点为各个种子，方点为平均差值；误差线为近似、未校正的 95% 配对区间。'))
    sections.append('<h3>Bootstrap 敏感性：推断对象不同会产生不同区间</h3>'+table(pd.read_csv(out/'bootstrap_sensitivity.csv')))
    sections.append('<p>固定模型条件区间只反映当前模型的测试样本抽样不确定性，不能证明换个训练种子仍有同样收益。交叉 bootstrap 同时重采样种子和共同测试样本，但仅有五种子，仍属探索性敏感性分析。400 标签的交叉 percentile 区间偏负且不跨零，而 paired-t 区间跨零，说明判定依赖统计方法；不能只保留一种。没有稳定跨预算的 M6 正向收益。</p>')
    sections.append('<h2>3. 错误分析：改对与改错必须一起看</h2><p>以下数字是每个种子上的样本数量，再对五种子取均值；不是额外的独立重复。</p>'+table(error_summary))
    sections.append(figure(out/'figures/04_error_tradeoffs.png','M6 修正的错误与新引入的错误同时展示。'))
    sections.append('<p>例如 400 标签的 Amazon：M6 平均修正 49.8 条，但破坏 98.2 条，净下降 0.968 个百分点；1,600 标签时则修正 104.2 条、破坏 69.4 条，净提升 0.696 个百分点。方向随数据预算变化。按类别看，400 标签 Amazon 的下降主要落在负类（−2.008 pp）；1,600 标签的提升也更多来自负类。它是可观察的类别差异，不等于已证实的机制。</p>')
    token_table=pd.DataFrame([{'domain':d,'test_n':r['n'],'tokens_gt_512':r['truncated_n'],
                              'truncation_percent':100*r['truncated_n']/r['n']} for d,r in joined.items()])
    sections.append('<h3>长度与截断</h3>'+table(token_table))
    sections.append('<p>长度使用固定 RoBERTa 分词器，包含特殊 token。Amazon 长文本只有 1 条，不能用该分组估计可靠规律；图中省略 n&lt;30 的分组，完整原始数值仍在 CSV。否定词分组只是正则表达式标记，不是人工语义标注。</p>')
    sections.append(figure(out/'figures/07_length_strata.png','探索性长度分组，展示均值与种子标准差。'))
    sections.append('<h3>可追踪案例，而非只挑成功例子</h3><p>事先按稳定 ID 哈希确定 36 个案例，覆盖三个预算、三个域、四种对错组合，均来自 seed 13。以下是其中全部 18 个模型分歧案例的 AI 辅助解释草稿，组员仍须复核。仅描述文本现象；未做反事实实验，不归因模型内部机制。</p>'+table(cases))
    sections.append('<h2>4. 消融与成本边界</h2>'+figure(out/'figures/05_lambda_ablation.png','λ=0 使用已选定的 M3；消融曲线不能用于重新选择测试集最优配置。'))
    sections.append('<p>L2-SP 的 15 个 winner 中有 14 个选到 λ=0.1 搜索下界；强正则失败并不能代表所有 L2-SP 配置。M6 的 pooled-feature 漂移是其直接优化目标，下降本身不证明保留了更有利于泛化的知识。</p>')
    sections.append(figure(out/'figures/06_drift_vs_accuracy.png','每个预算分开展示漂移与准确率；点之间有结构性依赖，不做混合因果推断。'))
    sections.append(figure(out/'figures/03_search_cost.png','独立复现整条流程的成本拆分；共享上游搜索不能重复累计为实际项目总耗时。'))
    sections.append(f'<p>B 的 18 个作业总分配时长为 {costs["B_allocation_hours_including_eval"]:.3f} GPU 小时，含评测与其他开销。M6-b6400-s21-lam1 超时续跑，候选搜索时间低估；最终选中 lam0.01 的自身训练时间不受这次续跑影响。B 普通方法还保留诊断 teacher，其显存属于当前实现；本轮没有生成未经测量的修正显存数值。</p>')
    sections.append('<h2>5. 补充运行包 v2 已准备什么，仍缺什么</h2><p>CKA 扩展到 M3/M4/M6/L2SP 的全部 60 个选中模型，三域各 256 个固定平衡样本，逐层比较 CLS 与平均池化特征。四种方法均对照预训练模型，M4/M6/L2SP 另对比配对 M3；共 8,190 行表示比较。保留去偏 CKA、固定乱序控制和恒定 CLS 的 NA 处理。</p><p>新增六方法的统一短程成本测试，另对 M3/M4/L2SP 增加诊断 teacher 驻留对照，三轮共 27 个独立进程。M6 的必要 teacher 和 L2SP 的参考权重均保留。固定训练样本、长度、批量和精度；M4 是第二阶段计算代理，不能替代完整 LP-FT 成本或准确率实验。</p><p><strong>真实 GPU 结果仍待 B 执行。</strong>先通过四模型与六方法 smoke，再启动 18 项自动接续任务；最后回传 C-return-v2.tar.gz。代码已通过小模型与模拟队列检查，实际服务器时间尚未实测。不需要转移 29GB 权重，也不重训完整模型。详见 HANDOFF_B_ZH.md 和 C-handoff-v2.json。</p>')
    sections.append('<h2>6. 统计与可复现性检查（11/11 项已覆盖）</h2>'+table(pd.DataFrame(fallacies,columns=['检查项','处理与限制'])))
    sections.append('<p>Material Passport：Origin Skill = academic-research-suite / experiment-agent；Mode = validate + authorized implementation；Verification Status = ANALYZED，归档指标已独立复算，训练未重跑，真实 CKA 待执行。模型权重、真实 TC1 速度、人工案例解释均不冒充已验证。</p>')
    sections.append('<h2>7. 方法来源与课程对应</h2><p>课程要求强调控制实验、局限和失败案例，并认可有证据的负面结果。本交付支持这些要求，但不是完整最终课程报告：LoRA、真实 CKA、报告整合与最终提交仍在后续范围。</p><ul><li><a href="https://proceedings.mlr.press/v97/kornblith19a.html">Kornblith et al. (2019), Similarity of Neural Network Representations Revisited</a></li><li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html">SciPy：配对置换检验</a></li><li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html">SciPy：配对 t 检验与区间</a></li><li><a href="https://huggingface.co/docs/transformers/v4.51.3/model_doc/roberta">Hugging Face：RoBERTa 隐藏状态接口</a></li></ul><p>AI 使用：Codex 协助实现、测试、统计核验与解释草稿；团队负责复核、最终科研判断和真实服务器执行。未发送组员消息，未修改 A/B 原始实验。</p>')
    css='''body{max-width:1080px;margin:40px auto;padding:0 24px;color:#111;background:white;font:16px/1.75 -apple-system,BlinkMacSystemFont,"Noto Sans CJK SC",Arial,sans-serif}h1{font-size:30px}h2{margin-top:42px;border-bottom:2px solid #111;padding-bottom:8px}h3{margin-top:28px}.lead{font-size:19px;font-weight:600}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6;margin:18px 0;display:block;overflow-x:auto}th,td{border-bottom:1px solid #ccc;text-align:left;padding:8px;vertical-align:top}th{background:#eee;color:black}figure{margin:25px 0;break-inside:avoid}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#444}code{overflow-wrap:anywhere}a{color:#111;text-decoration:underline}@media print{body{margin:0;font-size:10pt}h2{break-after:avoid}table{display:table;font-size:8pt}a{color:#111}}'''
    (out/'REPORT_ZH.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SC4001 C 前三项交付</title><style>'+css+'</style><body>'+''.join(sections)+'</body></html>')
    print('Wrote standalone REPORT_ZH.html')


if __name__=='__main__': main()
