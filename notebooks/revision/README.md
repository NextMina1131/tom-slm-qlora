# Revision notebooks

Colab notebooks for the experiments added in the revision. They use the outputs of
`../tombench_slm_qlora_complete_pipeline.ipynb` (the `tier4_v2_fixed/` folder on Drive: split and
manifest files, per-item CSVs and the seed-42 adapters) and the same pinned package versions
(`requirements.lock.txt`). Each notebook starts with a description of what it runs and in which
order.

| Notebook | What it runs | Files in `results/` | Paper |
|---|---|---|---|
| `01_replication_adapters_and_peft_reload` | retrains item-level seeds 43-46 and saves the adapters; re-evaluates seeds 42-46 in the original order after loading with `PeftModel.from_pretrained` | (adapters: `replication/` on the Hub; `plain_*` files are inputs to 02 and 03) | 4.9 |
| `02_evaluation_reproducibility` | repeated evaluation in one session, reload, and loading with unsloth, seed 42 | `reproducibility/repro_run1_*`, `repro_run2_*`, `repro_reload_*`, `repro_unslothload_seed42_{HiToM,SocialIQa}.csv`, `repro_mismatch_*`, `repro_margin_run1_vs_published.csv` | 4.9 |
| `03_unsloth_reload_reevaluation` | all five item-level adapters loaded with unsloth; retraining noise | `reproducibility/repro_unslothload_*`, `unsloth_reeval_*`, `socialiqa_*`, `adapter_hashes_1014.json` | 4.9 |
| `04_hf_adapter_and_logit_check` | the earlier Hub root adapter; logits under the two loading paths | `reproducibility/hf_and_logit_check/` | 4.9 |
| `05_storydisjoint_seeds_43_46` | story-disjoint seeds 43-46: training and evaluation | `groupsplit_finetuned_seed<N>_*.csv`, `storydisjoint_5seed_*.csv`, `storydisjoint_trainlogs/` | 4.5 |
| `06_cyclic_permutation_itemlevel` | cyclic permutation, base model and item-level seeds 42-46 | `cyclic_itemlevel/` | 4.7 |
| `07_fewshot_cyclic_permutation` | cyclic permutation of the three 3-shot conditions; answer-letter logit magnitudes; dtype of the 16-bit control | `fewshot_cyclic/`, `reproducibility/answer_letter_logit_magnitude.csv`, `reproducibility/dtype_check_*` | 4.6, 4.7, 4.9 |
| `08_cyclic_permutation_storydisjoint` | cyclic permutation, story-disjoint seeds 42-46 | `cyclic_storydisjoint/` | 4.7 |

Run 01 before 02-04, 03 before 06, 05 before 08, and 06 before 07 and 08.

Notes

- The notebooks write the `raw` generation column; it was removed from the released files with
  `code/strip_raw_generations.py` (see `CHANGELOG.md`).
- The Drive folder names are `ToMBench_Results/` (as in the main notebook), with `cyclic_shift_check/`
  (01), `repro_check/` (02, 03), `hf_logit_check/` (04), `storydisjoint_5seed/` (05),
  `cyclic_itemlevel/` (06), `fewshot_cyclic/` (07) and `cyclic_storydisjoint/` (08).
- The labels `gate1011` (02) and `mismatch_peft_1011` (03) refer to the `PeftModel` re-evaluation of
  notebook 01; they are kept because they appear in the released files.
- The local tables of the paper are recomputed from `results/` without a GPU by the scripts in
  `code/` (see the top-level README).
