# Low-Resource Sentiment Classification and Cross-Domain Generalisation

This project investigates how adaptation strategies for pretrained language models affect sentiment classification when labelled source-domain data are limited. Evaluation measures both in-domain performance and transfer to unseen review domains.

## Experimental components

| Component | Scope | Documentation |
|---|---|---|
| Member A | Data preparation, fixed evaluation protocol, frozen-encoder and partial fine-tuning baselines | [Implementation and results](member-a/README.md) |

The completed Member A experiments use RoBERTa-base, three label budgets, and five paired seeds. Training and model selection use IMDb; evaluation covers IMDb, Yelp Polarity, and Amazon Polarity. The committed records include 90 candidate training runs, 30 selected configurations, per-example predictions, and verified aggregate results.

## Getting started

Read the [Member A README](member-a/README.md) for the research objective, methods, experimental protocol, result tables, and reproduction commands. The [English results summary](member-a/formal-results/results/A-v1/summary.md) provides accuracy statistics and paired comparisons; [per-seed records](member-a/formal-results/results/A-v1/per_seed.csv) expose the underlying measurements.

Model weights and downloaded datasets are excluded from the repository. Environment and data reconstruction instructions are included with the implementation.
