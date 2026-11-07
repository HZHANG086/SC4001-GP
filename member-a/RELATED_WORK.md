# Related work draft

以下英文段落可供 A 的报告部分使用。文献已核对原始论文页面；项目主实验尚未完成，因此使用研究动机和拟验证问题的表述。正文须根据最终实际实现和结果调整。

## English draft

**Pretrained language representations.** RoBERTa revisits BERT pretraining and shows that training data, optimization choices and other design decisions substantially affect downstream performance. This motivates using a common pretrained backbone and a controlled adaptation protocol in our study. We use the original RoBERTa-base checkpoint, rather than a sentiment classifier already trained on IMDb, so that comparisons concern adaptation from the same starting representation. IMDb provides our labeled source domain, while evaluation on other review domains tests transfer beyond the source dataset. [Liu et al., 2019](https://arxiv.org/abs/1907.11692); [Maas et al., 2011](https://aclanthology.org/P11-1015/).

**Adaptation and distribution shift.** Kumar et al. study the trade-off between linear probing and full fine-tuning under distribution shift. Their empirical benchmarks are vision datasets; their analysis motivates a two-stage linear-probing-then-fine-tuning strategy. This supplies a hypothesis for our text experiments: adaptation flexibility may interact with retention of pretrained features. It does not establish that full fine-tuning must hurt sentiment transfer. We therefore compare absolute source and external-domain accuracies, and treat any accuracy–representation association as supporting evidence rather than a causal result. LoRA supplies a complementary parameter-efficient comparison by freezing pretrained weights and learning low-rank updates. Its reported language-model results motivate including it in the broader project, without implying an automatic advantage in our low-label setting. [Kumar et al., 2022](https://arxiv.org/abs/2202.10054); [Hu et al., 2021](https://arxiv.org/abs/2106.09685).

**Regularized transfer learning.** Li, Grandvalet and Davoine investigate penalties that bias fine-tuned parameters toward their pretrained initialization, motivating an L2-SP control. DELTA instead constrains selected feature-map outputs in convolutional networks. These works distinguish parameter preservation from representation preservation and motivate the project's planned comparison between the two. Our intended contribution is a controlled adaptation and evaluation of this idea for low-label cross-domain sentiment classification. The final report must specify the actual representation, normalization and loss used, and report unsuccessful settings as well as any benefits. The vision results in these papers do not establish effectiveness on RoBERTa or on unseen review domains. [Li et al., 2018](https://arxiv.org/abs/1802.01483); [Li et al., 2019](https://arxiv.org/abs/1901.09229).

## 使用边界

- 不把冻结/部分/全量微调比较本身称为新方法。
- 不把 LP-FT、DELTA 或 L2-SP 的视觉结论直接写成文本领域事实。
- 不把只有单种子的 pilot 称为稳定优势；跨域结论必须来自冻结协议后的正式测试。
- B/C 的最终实现如果改变表示约束或分类头，应同步修改本段及方法部分。
- 公开课程仓库可在额外背景阅读中引用；没有可核验成绩记录时不能称为满分项目。
