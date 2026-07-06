# Results

Per-item predictions and aggregate metrics for the base (zero-shot) vs. QLoRA fine-tuned
Qwen2.5-3B model, across ToMBench (in-domain) and four external benchmarks.

## Per-item files

`baseline_<benchmark>.csv` / `finetuned_<benchmark>.csv` — one row per evaluated item.

| column | meaning |
|---|---|
| `source` | benchmark name |
| `task` | task type |
| `category` | ATOMS ability category |
| `ability` | fine-grained ATOMS ability |
| `answer` | gold answer key (option letter) |
| `pred` | model's predicted option |
| `raw` | model's raw generation (e.g. `[[D]]`) |
| `correct` | 1 if `pred == answer`, else 0 |

> **Note:** the original `user_prompt` column (the full benchmark story/question/options) has
> been **removed** from these files, because the benchmark data are under their original
> licenses and are not redistributed here. Rows are in the same order as the corresponding
> split-index files in `../splits/`, so full inputs can be reconstructed after obtaining the
> original benchmark data. `baseline_*` and `finetuned_*` rows align 1:1 by row order.

## Aggregate / analysis files

| file | contents |
|---|---|
| `SUMMARY_main_table.csv` | main results table (accuracy / macro-F1 per benchmark) |
| `SUMMARY_all_metrics.csv` | all computed metrics |
| `mcnemar_results.csv` | McNemar test per benchmark (b, c, statistic, p-value) |
| `mcnemar_ability.csv` | McNemar broken down by ATOMS ability |
| `training_log.csv` | training loss / step log |
| `stats/paired_<benchmark>.csv` | paired correctness (`idx, ability, base_correct, tuned_correct`) — input to McNemar/bootstrap |
| `stats/mcnemar_pvalue_table.csv` | McNemar p-value table (continuity-corrected χ²) |
| `stats/bootstrap_delta_CI.csv` | bootstrap 95% CIs for Δ accuracy (10,000 resamples, seed=42) |
