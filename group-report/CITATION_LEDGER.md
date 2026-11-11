# Original-source citation checks

Verified during report integration on 11 November 2026. Metadata and relevant abstract/landing-page scope were checked against original publication hosts; this does not claim that a human read every full paper. No prior student report is presented as scientific evidence.

| Key | Primary source | Supported role and limit |
|---|---|---|
| roberta | [Liu et al., 2019](https://arxiv.org/abs/1907.11692) | RoBERTa pretraining design; not evidence for our own transfer result. |
| imdb | [Maas et al., ACL 2011](https://aclanthology.org/P11-1015/) | IMDb sentiment benchmark; our cleaning and sampling decisions are documented separately. No DOI invented. |
| reviews | [Zhang, Zhao and LeCun, 2015](https://arxiv.org/abs/1509.01626) | Review polarity datasets. Distributed data cross-checked against [Yelp](https://huggingface.co/datasets/fancyzhx/yelp_polarity) and [Amazon](https://huggingface.co/datasets/fancyzhx/amazon_polarity) cards. |
| lpft | [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054) | Feature distortion and LP-FT motivation. Vision results do not establish a sentiment-domain guarantee. |
| lora | [Hu et al., 2021 preprint](https://arxiv.org/abs/2106.09685) | Low-rank adaptation as established prior work. The report cites the verified preprint year. |
| l2sp | [Li, Grandvalet and Davoine, ICML 2018](https://arxiv.org/abs/1802.01483) | Anchoring fine-tuned parameters to their pretrained values. Our encoder-only implementation and coefficient convention are stated explicitly. |
| delta | [Li et al., ICLR 2019](https://arxiv.org/abs/1901.09229) | Attention-weighted convolutional feature preservation as related work. M6 is not claimed to reproduce DELTA. |
| cka | [Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html) | Representation similarity via CKA; it is not a semantic correctness or causal explanation metric. |

All eleven bibliography entries are cited in the main body. Report results come from this repository's actual experiments, not from these papers. Pretrained model identity and exact revision are recorded in all three frozen protocols and Appendix A; the [model card](https://huggingface.co/FacebookAI/roberta-base) supplies distribution context.

## Grading-focused revision

Three closely related NLP sources were added after checking the official abstract and metadata on 11 November 2026. Their roles are context, not unperformed baselines or evidence of our method outperforming them.

| Key | Original source | Scope used in the report |
|---|---|---|
| recadam | [Chen et al., EMNLP 2020, pp. 7870–7881](https://aclanthology.org/2020.emnlp-main.634/) | Pretraining simulation and gradual objective shifting; distinguishes our fixed cosine objective. |
| smart | [Jiang et al., ACL 2020, pp. 2177–2190](https://aclanthology.org/2020.acl-main.197/) | Smoothness regularization and proximal optimization; not implemented or compared here. |
| r3f | [Aghajanyan et al., 2020 preprint](https://arxiv.org/abs/2008.03156) | Noise-based regularization and representation-collapse motivation; no claim that geometry is a semantic score. |

The revised related-work discussion uses less than 50 words of paraphrased source-specific content per added paper and no direct quotations. The public-coursework comparison is in `RUBRIC_REVIEW_ZH.md`; coursework is not added as scientific evidence in the bibliography.
