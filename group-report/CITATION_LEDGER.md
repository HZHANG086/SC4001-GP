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

All eight bibliography entries are cited in the main body. Report results come from this repository's actual experiments, not from these papers. Pretrained model identity and exact revision are recorded in all three frozen protocols and Appendix A; the [model card](https://huggingface.co/FacebookAI/roberta-base) supplies distribution context.
