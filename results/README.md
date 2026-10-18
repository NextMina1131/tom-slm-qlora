# Results

Per-item predictions and derived statistics underlying every number reported in the
manuscript. All files come from a single evaluation pipeline (`notebooks/tombench_slm_qlora_complete_pipeline.ipynb`)
that assigns every item a content-based, cross-run-stable identifier before any comparison is
computed; the analysis in `code/analyze_tier4_v2.py` verifies this identity before running any
paired statistical test.

None of these files contain the underlying benchmark story/question text. Each row is keyed by
`item_id` (a hash of the normalized source benchmark, task, story, and question) and
`prompt_hash` (a hash of the exact rendered prompt), plus the model's prediction and scoring
fields. The raw generations were removed because chain-of-thought outputs quote benchmark text
(see `CHANGELOG.md`); they are available from the corresponding author on request. To reconstruct the original questions, obtain the benchmark data
from its official source (see the top-level README) and rebuild the manifests by running the
notebook, which regenerates them deterministically from the same hashing scheme.

## File groups

- `baseline_<dataset>.csv` — zero-shot base model, one file per dataset (ToMBench, OpenToM,
  ToMi, SocialIQa, HiToM).
- `finetuned_seed<N>_<dataset>.csv` — fine-tuned model, one file per training seed
  (42, 43, 44, 45, 46) per dataset. Seed 42 is the pre-specified primary comparison; the
  remaining seeds are the multi-seed replication.
- `groupsplit_base_ToMBench.csv` and `groupsplit_finetuned_<dataset>.csv` (seed 42),
  `groupsplit_finetuned_seed<N>_<dataset>.csv` (seeds 43–46) — the story-disjoint split
  (Section 4.5), five seeds, on the 837-item story-disjoint ToMBench test split and the four
  external benchmarks. `storydisjoint_5seed_by_seed.csv`, `storydisjoint_5seed_summary.csv` —
  per-seed and five-seed summaries.
- `groupsplit_finetuned_run20260818_ToMBench.csv`, `groupsplit_summary_v2.json` — an earlier
  seed-42 story-disjoint run (2026-08-18) reported as a single run in the original submission
  (+10.39 points). Its adapter was not retained, and it differs from the seed-42 run above (whose
  adapter was retained and which also produced the external-benchmark files) on 21 of the 837
  test items; the paper reports both (Section 4.5, footnote).
- `cyclic_itemlevel/`, `cyclic_storydisjoint/` — cyclic-permutation re-evaluation (Section 4.7)
  of the base model and of the five item-level and five story-disjoint models: one row per
  item and rotation (`shift_k`), with `gold_orig` (gold letter in the original order),
  `new_gold` (gold letter after rotation), `pred` (letter chosen), `pred_orig_letter` (the
  chosen option's letter in the original order, used for the consistency measure) and
  `letter_margin`. The item-level model files hold shifts 1..K−1; their unrotated predictions,
  from the same adapters loaded the same way, are `reproducibility/repro_unslothload_*`.
  Item-level seeds 43–46 are same-configuration replication adapters.
- `fewshot_cyclic/` — the three 3-shot conditions under the same permutation (shifts 1..K−1;
  shift 0 is the original-order few-shot file). `gate_k0_check.csv` re-ran shift 0 on a
  sample and matched the original-order files.
- `reproducibility/` — Section 4.9: repeated evaluation of the seed-42 adapter (`repro_run1`,
  `repro_run2`, `repro_reload`), the Unsloth loading path (`repro_unslothload_seed<N>`),
  re-trained seeds 43–46 against the released predictions (`unsloth_reeval_*`,
  `socialiqa_*`), fp16/bf16 loading of the non-quantized control (`dtype_check_*`),
  answer-letter logits under the two loading paths (`answer_letter_logit_magnitude.csv`,
  `hf_and_logit_check/logit_diff_two_paths.csv`), and the evaluation of the adapter that was
  previously on the Hugging Face Hub (`hf_and_logit_check/hf_adapter_*`).
- `storydisjoint_trainlogs/` — per-epoch training/validation logs of story-disjoint seeds 43–46.
- `seed_epoch_recovery/` — per-epoch validation loss of four item-level replicate runs
  (seeds 43–46). **The seed-42 row of `seed_epoch_recovery/seed_epoch_table.csv`
  and `training_log.csv` come from an earlier development run (May 2026), not from the run that
  produced the released predictions;** the per-epoch log of that run was not preserved
  (Section 4.9 of the paper).
- `<condition>_ex<K>_<dataset>.csv` — training-free few-shot baselines (Section 4.6):
  `answer_only_3shot`, `synthetic_format_only_3shot`, `rationale_cot_3shot`, each with three
  independent exemplar draws (`ex0`-`ex2`).
- `fewshot_conditions_summary_v2.csv`, `multi_seed_summary_v2.csv`,
  `multi_seed_results_v2.json` — aggregated accuracy summaries used to build the manuscript
  tables.
- `inference_perf_v2.json` — measured latency/throughput/peak-VRAM figures (Section 5.4).
- `ANALYSIS_primary_results_v2.csv`, `ANALYSIS_hitom_*_v2.csv` — output of
  `code/analyze_tier4_v2.py`: McNemar tests, story-cluster bootstrap CIs, Holm-corrected
  p-values, and the Hi-ToM gold-answer-position confusion matrix and permutation test.
- `ANALYSIS_cyclic_*_v2.csv`, `ANALYSIS_storydisjoint_original_order*_v2.csv` — output of
  `code/analyze_cyclic.py` (Tables "Story-disjoint split", "Cyclic-permutation
  re-evaluation" and "3-shot conditions under cyclic permutation").
- `ANALYSIS_ability_level_v2.csv`, `ANALYSIS_task_split_counts_v2.csv` — output of
  `code/analyze_ability_and_split.py`: every ToMBench ability (seed 42, paired item-level
  bootstrap CI) and the train/validation/test count of every ToMBench task.
- `VERIFICATION_LOG_v2.csv`, `VERIFICATION_LOG_cyclic_v2.csv` — every item-identity check run
  by `analyze_tier4_v2.py` and `analyze_cyclic.py`.

## Column reference (per-item CSVs)

| Column | Meaning |
|---|---|
| `item_id` | Canonical, content-based item identifier (stable across runs/sessions) |
| `prompt_hash` | Hash of the exact rendered prompt (verifies option order/content match before any paired comparison) |
| `story_id` | Hash of the source story only, used for story-cluster bootstrap resampling |
| `source`, `task`, `category`, `ability` | Benchmark provenance / ATOMS taxonomy fields |
| `answer` | Gold answer letter |
| `pred` | Extracted predicted answer letter |
| `correct` | 1 if `pred == answer`, else 0 |
| `n_options` | Number of answer options for this item (Hi-ToM only; NaN elsewhere) |
| `marker_found` | Whether a `[[X]]` marker (case-insensitive, spaces allowed) was found in the generation |
| `strict_format` | Whether the generation contained an exact `[[A-E]]` marker (strict-format analysis, Section 5.2) |
| `extraction_rule` | How the letter was obtained: `marker`, `fallback` (first standalone A–E letter), or `none` (default `A`) |
| `raw_len` | Length of the generation in characters |
| `condition`, `seed`, `exemplar_seed` | Which evaluation condition produced this row |

## Reproducing the statistics

```
cd code
python analyze_tier4_v2.py --dir ../results
python analyze_cyclic.py --dir ../results
python analyze_ability_and_split.py --dir ../results --splits ../splits
```
