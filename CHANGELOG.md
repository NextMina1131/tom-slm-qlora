# Changelog

## [Unreleased] — revision for applsci-4589116 (October 2026)

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
- Results of the revision's new experiments, with the same scoring fields and without raw
  generations: cyclic-permutation re-evaluation of the item-level and story-disjoint models
  (`results/cyclic_itemlevel/`, `results/cyclic_storydisjoint/`) and of the 3-shot conditions
  (`results/fewshot_cyclic/`); story-disjoint seeds 43–46 (`results/groupsplit_finetuned_seed<N>_*.csv`,
  `results/storydisjoint_trainlogs/`); reproducibility measurements (`results/reproducibility/`).
- `code/analyze_cyclic.py` (cyclic-permutation tables and the story-disjoint original-order table,
  with item-identity checks logged to `results/VERIFICATION_LOG_cyclic_v2.csv`) and
  `code/analyze_ability_and_split.py` (full ability-level table and task-level split counts,
  which the revised paper no longer prints).

### Changed
- `results/groupsplit_finetuned_ToMBench.csv` now holds the seed-42 story-disjoint run whose
  adapter was retained (2026-09-03), the same run as the four external-benchmark
  `groupsplit_finetuned_*.csv` files and the one used in the paper's five-seed story-disjoint
  results. The earlier run (2026-08-18) that the file held before is kept as
  `results/groupsplit_finetuned_run20260818_ToMBench.csv`.
- README: key results restated with the position-controlled (cyclic-permutation) figures of the
  revised paper; adapter layout on the Hugging Face Hub; batch split corrected to the L4 runs
  (per-device 4 × accumulation 2); `results/README.md` labels `training_log.csv` and the seed-42
  row of `seed_epoch_recovery/seed_epoch_table.csv` as coming from an earlier development run.
