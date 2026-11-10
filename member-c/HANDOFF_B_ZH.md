# 给 B：只运行推理，不重训、不搬 29GB 权重

## 目标

比较 M3 与 M6 的表示：三个预算 × 五个种子共 30 个已经选定的模型。每个域固定 256 条测试文本，全部模型使用相同 ID 和顺序。每个种子一个任务，加载 6 个模型。测试文本只用于解释模型，不选择参数。

只依赖你现有环境的 torch 2.5.1、transformers 4.51.3、datasets 3.5.0 和 numpy 1.26.4。服务器端**不需要 scipy、matplotlib、scikit-learn 或 peft**。

## 放置文件

把这个 `member-c/` 文件夹放到你现有项目根目录，与 `member-a/`、`member-b/` 并列。你原来的数据和 `runs/B-v1/*/best/` 保持原位置。不要修改 A/B 冻结源码。

脚本需要这些原有文件：

- `member-b/formal-results/selected_models.tsv`：60 个选中模型清单；自动筛选 M3/M6。
- `member-b/runs/B-v1/*/best/`：真实模型权重、配置和分词器。
- 每个真实运行目录的 `run_config.json`。
- `member-b/formal-results/runs/B-v1/selection-*.json` 与归档预测。
- `member-b/artifacts/data-v1/`：现成数据，含 READY 和 sources.json。
- `member-a/remote-records/data-v1/`：参考哈希。
- 本地缓存的固定版本 `FacebookAI/roberta-base`。正式任务不联网下载，不使用不明新版本。

## 第一次先测两个模型

在 TC1 的 **member-b 目录**提交：

```bash
mkdir -p logs
sbatch ../member-c/scripts/cka.sbatch 13 smoke
```

这会在 GPU 上运行 seed 13、budget 400 的 M3/M6，输出 `analysis/cka-C-v1/smoke-s13/`。请先确认有 `COMPLETE`，并检查日志无 traceback。该任务包含全部三个域、固定 256 条输入，检验真实权重路径和与历史预测的一致性。

`run.json` 会记录实际耗时。尚未测量 V100 正式运行时间，不能保证每个完整任务的耗时；若 smoke 耗时异常，先反馈日志再安排正式任务。

## 再跑五个正式种子

同样在 member-b 目录提交，**同一账号最多同时提交两个任务**；等名额空出再提交下一批：

```bash
sbatch ../member-c/scripts/cka.sbatch 13 full
sbatch ../member-c/scripts/cka.sbatch 21 full
# 等上面任务结束或有名额后，再提交 42、87、100。
```

任务调用的是已有模型推理。每个任务请求 1 张 GPU、24GB 主机内存、5小时50分钟上限。若任务中断，不会损坏权重；保留日志。重新运行相同命令会检查输入一致性，重算该种子的未完成任务；已完整完成的任务跳过。不混用不同代码版本的中间结果。

## 自动检查内容

1. 数据元信息与 A 固定哈希一致；固定模型版本和长度一致。
2. 模型清单和已冻结的 winner 一致，记录实际 safetensors SHA256。
3. 每个模型在固定样本上的输出与归档概率相比，最大差异不得超过 1e-4；超出则停止，不输出成功结论。
4. 同时计算普通线性 CKA、去偏 CKA 敏感性、固定乱序控制和余弦距离。
5. 第 0 层 CLS 对所有文本可能恒定，CKA 记为 NA；不强行写成 0 或 1。
6. 所有推理 eval、FP32、关闭 TF32；最终相似度在 CPU 上用 float64 计算。

## 返回什么

只返回这些小文件，无需权重或中间特征：

```text
analysis/cka-C-v1/seed-13/{cka.csv,prediction_replay.csv,manifest.json,run.json,COMPLETE}
analysis/cka-C-v1/seed-21/...
analysis/cka-C-v1/seed-42/...
analysis/cka-C-v1/seed-87/...
analysis/cka-C-v1/seed-100/...
logs/cka_c_<jobid>.out
logs/cka_c_<jobid>.err
```

`manifest.json` 已包含样本 ID、文本哈希、模型哈希和代码哈希；`run.json` 已包含 GPU、软件版本、耗时和 job ID。可提交到你原来的 GitHub 分支。C 再合并并核验。

保留 smoke 输出也有助于查错，但它不应重复计入五个正式种子的统计。当前 C 的分析包已通过离线小模型测试，**尚未代替这次真实模型运行**。
