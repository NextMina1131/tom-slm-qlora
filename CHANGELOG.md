# Changelog

## [Unreleased] — second revision for applsci-4589116 (October 2026)

### Added
- `code/analyze_tomi_distractors.py`, `results/tomi_item_story_mention.csv`,
  `results/ANALYSIS_tomi_story_mention.csv`. ToMi distractors are sampled from the pool of all
  ToMi answers, so in 720 of the 1,000 evaluated items the gold answer is the only option that
  occurs in the story. The script flags these items and splits the existing cyclic-permutation
  results of the fine-tuned models and the 3-shot conditions by the flag. No model was re-run,
  and no existing file changed.

### Changed
- `code/gen_figure1.py`: Figure 1 is drawn at its printed width (7.0 in) with Panels A and B
  side by side above Panel C, and saved at 1000 dpi. The plotted values are unchanged.
- `notebooks/tombench_slm_qlora_complete_pipeline.ipynb`: table numbers in five `print`
  messages now match the revised paper (Tables 6, 7 and 13). No code or output changed.

## Revision for applsci-4589116 (October 2026)

### Removed
- **`raw` column (model generations) from all per-item result CSVs in `results/`.** Chain-of-thought
  generations quoted benchmark source text verbatim (in the rationale-CoT 3-shot files, for example,
  25–255 of 842–1,000 items per file contained a passage of 40 or more characters identical to the
  source story or question). The paper's data availability statement says that released files
  contain identifiers and hashes rather than source text, so the column was removed from the public
  files. The removal applies to all 91 files that had the column, so that every per-item file has
  the same schema. Raw generations are available from the corresponding author on request.
  The files were published in commit `e3f155c` and later; this change is made by a new commit,
  and the repository history was not rewritten.

### Added
- `marker_found`, `strict_format`, `extraction_rule`, `raw_len` columns in place of `raw`
  (`code/strip_raw_generations.py`). Before removal, the letter re-extracted from each generation
  with the evaluation pipeline's rule matched the stored `pred` on all 87,988 rows.
- `code/analyze_tier4_v2.py`: the strict-format-subset analysis reads `strict_format` when the raw
  text is absent; its output is identical before and after the change.
- Per-item results of the fine-tuned seed-42 model under the three 3-shot conditions
  (`results/<condition>_tuned_ex<K>_<dataset>.csv`, 45 files, and
  `results/fewshot_tuned_conditions_summary_v2.csv`). The paper has reported these results since
  the original submission, but the files had not been added to the repository. They carry the
  same scoring fields as the other files and no raw generations; before the `raw` column was
  removed, the re-extracted letter matched the stored `pred` on all 43,578 rows.
  `code/analyze_tier4_v2.py` now computes their table (`results/ANALYSIS_fewshot_tuned_variance_v2.csv`)
  and checks their item identity against the fine-tuned zero-shot files, which adds 45 rows to
  `results/VERIFICATION_LOG_v2.csv` (132 checks, all passed). No other output changed.
- Results of the revision's new experiments, with the same scoring fields and without raw
  generations: cyclic-permutation re-evaluation of the item-level and story-disjoint models
  (`results/cyclic_itemlevel/`, `results/cyclic_storydisjoint/`) and of the 3-shot conditions
  (`results/fewshot_cyclic/`); story-disjoint seeds 43–46 (`results/groupsplit_finetuned_seed<N>_*.csv`,
  `results/storydisjoint_trainlogs/`); reproducibility measurements (`results/reproducibility/`).
- `code/analyze_cyclic.py` (cyclic-permutation tables and the story-disjoint original-order table,
  with item-identity checks logged to `results/VERIFICATION_LOG_cyclic_v2.csv`) and
  `code/analyze_ability_and_split.py` (full ability-level table and task-level split counts,
  which the revised paper no longer prints).

- `notebooks/revision/`: the eight Colab notebooks that produced the revision's new results
  (replication adapters for item-level seeds 43-46, reproducibility of the evaluation, story-disjoint
  seeds 43-46, cyclic permutation of the fine-tuned models and of the 3-shot conditions), with a
  README that maps each notebook to its files in `results/` and to the section of the paper.

### Changed
- `results/training_log.csv` renamed to `results/training_log_run20260525.csv`, and a `source_run`
  column added to `results/seed_epoch_recovery/seed_epoch_table.csv` (`2026-05-25` for seed 42,
  `2026-09-replicate` for seeds 43–46), so that each file and row states which run it comes from
  without relying on the README. No values changed.
- Code comments and docstrings shortened; references to the review process and a local output path
  in `code/gen_figure1.py` removed (it now writes `Figure1.png` to the current directory). Markdown
  cells of the main notebook updated in the same way. No computation changed.
- `results/groupsplit_finetuned_ToMBench.csv` now holds the seed-42 story-disjoint run whose
  adapter was retained (2026-09-03), the same run as the four external-benchmark
  `groupsplit_finetuned_*.csv` files and the one used in the paper's five-seed story-disjoint
  results. The earlier run (2026-08-18) that the file held before is kept as
  `results/groupsplit_finetuned_run20260818_ToMBench.csv`.
- `code/gen_figure1.py`: Panel C marked "Involuntary lies" as having a bootstrap interval that
  excludes zero; its interval is [0.0, +41.7] (`results/ANALYSIS_ability_level_v2.csv`), so the
  marker was removed. No other value in the figure changed.
- README: the introduction names both data splits.
- README: key results restated with the position-controlled (cyclic-permutation) figures of the
  revised paper; adapter layout on the Hugging Face Hub; batch split corrected to the L4 runs
  (per-device 4 × accumulation 2); `results/README.md` labels `training_log.csv` (now `training_log_run20260525.csv`) and the seed-42
  row of `seed_epoch_recovery/seed_epoch_table.csv` as coming from an earlier development run.
