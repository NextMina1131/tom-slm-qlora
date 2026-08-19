# Results

Per-item predictions and derived statistics underlying every number reported in the
manuscript. All files come from a single evaluation pipeline (`notebooks/tombench_full_rerun_v2.ipynb`)
that assigns every item a content-based, cross-run-stable identifier before any comparison is
computed; the analysis in `code/analyze_tier4_v2.py` verifies this identity before running any
paired statistical test.

None of these files contain the underlying benchmark story/question text. Each row is keyed by
`item_id` (a hash of the normalized source benchmark, task, story, and question) and
`prompt_hash` (a hash of the exact rendered prompt), plus the model's own prediction and
generated text (`pred`, `raw`). To reconstruct the original questions, obtain the benchmark data
from its official source (see the top-level README) and rebuild the manifests by running the
notebook, which regenerates them deterministically from the same hashing scheme.

## File groups

- `baseline_<dataset>.csv` — zero-shot base model, one file per dataset (ToMBench, OpenToM,
  ToMi, SocialIQa, HiToM).
- `finetuned_seed<N>_<dataset>.csv` — fine-tuned model, one file per training seed
  (42, 43, 44, 45, 46) per dataset. Seed 42 is the pre-registered primary comparison; the
  remaining seeds are the multi-seed replication.
- `groupsplit_base_ToMBench.csv`, `groupsplit_finetuned_ToMBench.csv`,
  `groupsplit_summary_v2.json` — the story-identity-only group split control (Section 4.6).
- `<condition>_ex<K>_<dataset>.csv` — training-free few-shot baselines (Section 4.7):
  `answer_only_3shot`, `synthetic_format_only_3shot`, `rationale_cot_3shot`, each with three
  independent exemplar draws (`ex0`-`ex2`).
- `fewshot_conditions_summary_v2.csv`, `multi_seed_summary_v2.csv`,
  `multi_seed_results_v2.json` — aggregated accuracy summaries used to build the manuscript
  tables.
- `inference_perf_v2.json` — measured latency/throughput/peak-VRAM figures (Section 5.4).
- `ANALYSIS_primary_results_v2.csv`, `ANALYSIS_hitom_*_v2.csv` — output of
  `code/analyze_tier4_v2.py`: McNemar tests, story-cluster bootstrap CIs, Holm-corrected
  p-values, and the Hi-ToM gold-answer-position confusion matrix and permutation test.

## Column reference (per-item CSVs)

| Column | Meaning |
|---|---|
| `item_id` | Canonical, content-based item identifier (stable across runs/sessions) |
| `prompt_hash` | Hash of the exact rendered prompt (verifies option order/content match before any paired comparison) |
| `story_id` | Hash of the source story only, used for story-cluster bootstrap resampling |
| `source`, `task`, `category`, `ability` | Benchmark provenance / ATOMS taxonomy fields |
| `answer` | Gold answer letter |
| `pred` | Extracted predicted answer letter |
| `raw` | Full raw model generation for this item |
| `correct` | 1 if `pred == answer`, else 0 |
| `n_options` | Number of answer options for this item (Hi-ToM only; NaN elsewhere) |
| `marker_found` | Whether the exact `[[X]]` format was found in `raw` (few-shot CSVs only) |
| `condition`, `seed`, `exemplar_seed` | Which evaluation condition produced this row |

## Reproducing the statistics

```
python code/analyze_tier4_v2.py --dir results
```
