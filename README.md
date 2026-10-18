# QLoRA Fine-Tuning for Theory-of-Mind Reasoning in a 3B-Parameter Small Language Model

A ToMBench-Based Cross-Benchmark Generalization Study.

This repository contains the code, data-split indices, per-item prediction results, and
statistical analysis outputs for the paper *"QLoRA Fine-Tuning for Theory-of-Mind Reasoning
in a 3B-Parameter Small Language Model: A ToMBench-Based Cross-Benchmark Generalization Study."*

We fine-tune the 4-bit pre-quantized **Qwen2.5-3B-Instruct** model (Unsloth) with a **QLoRA**
adapter on an ability-stratified ToMBench train/validation/test split, and analyze how the
in-domain gains transfer to four external benchmarks (ToMi, OpenToM, SocialIQa, Hi-ToM).

**Authors:** Ji-Hyeong Hong¹, Sang-Hyun Yoo²,\*
¹ Department of Cyber Hacking and Security, Seoul HOSEO Technical College, Seoul, Republic of Korea
² School of Computer Science and Engineering, Soongsil University, Seoul, Republic of Korea
\* Corresponding author: simonyoo@ssu.ac.kr

## Key results

Change from the zero-shot base model in percentage points, five training seeds, mean ± SD with
the between-seed 95% t-interval. "Lenient" is accuracy averaged over cyclic permutations of the
answer options, so that each item's gold answer appears once in every position (Section 4.7 of
the paper). The paper states a direction only when the interval excludes zero for lenient,
strict and consistency measures with the same sign.

| Benchmark | Story-disjoint split, lenient | Item-level split, lenient |
|---|---|---|
| ToMBench (in-domain, held-out) | +8.15 ± 0.56 [+7.46, +8.85] | +14.01 ± 0.71 [+13.12, +14.89] |
| ToMi | +14.19 ± 1.87 [+11.87, +16.50] | +10.38 ± 1.73 [+8.24, +12.52] |
| OpenToM | +0.67 ± 0.68 [−0.17, +1.52] | +0.03 ± 0.90 [−1.09, +1.14] |
| SocialIQa | −0.77 ± 1.44 [−2.56, +1.03] | −4.14 ± 0.30 [−4.51, −3.77] |
| Hi-ToM | +1.26 ± 1.12 [−0.13, +2.65] | +2.16 ± 1.12 [+0.77, +3.54] |

Training-free 3-shot prompting under the same permutation raises ToMi lenient accuracy by
+7.23 points with content-free format exemplars and by +12.96 with rationale exemplars. See the
paper for the strict and consistency measures and for the interpretation.
`python code/analyze_cyclic.py --dir results` reproduces these tables from the released files.

## Repository layout

```
code/           fine-tuning / inference / evaluation / analysis scripts, including
                gen_figure1.py (regenerates the manuscript's Figure 1 from results/ tables)
code/legacy/    original Chinese-language ToMBench runner scripts (run_api.py, run_huggingface.py,
                prompts.py, etc.); not used in the current pipeline — kept for reference only
notebooks/      tombench_slm_qlora_complete_pipeline.ipynb — the end-to-end notebook covering the
                full pipeline, including the story-level group-split control, fine-tuned few-shot
                evaluation, and fp16 control experiment; run this in Google Colab.
notebooks/revision/
                the revision's experiments (replication adapters, reproducibility of the
                evaluation, story-disjoint seeds 43-46, cyclic permutation); see its README.md
splits/         ability-stratified ToMBench train/val/test index files (item_id only, seed=42)
results/        per-item predictions and analysis outputs — see results/README.md:
                original-order predictions (item-level seeds 42-46, story-disjoint seeds 42-46,
                three training-free few-shot conditions, the 16-bit control),
                cyclic_itemlevel/, cyclic_storydisjoint/, fewshot_cyclic/ (cyclic-permutation
                re-evaluation, Section 4.7), reproducibility/ (Section 4.9),
                storydisjoint_trainlogs/ and seed_epoch_recovery/ (training logs)
docs/           ATOMS ability mapping
```

## Item identity and reproducibility

Every evaluation record carries a canonical `item_id` (a SHA-256 hash of the normalized source
benchmark, task, story, and question) and a `prompt_hash` (a hash of the exact rendered
prompt). `code/analyze_tier4_v2.py` verifies that two result files being compared cover
identical items with identical prompts before computing any paired statistic, and refuses to
proceed otherwise. `notebooks/tombench_slm_qlora_complete_pipeline.ipynb` builds each
external-benchmark evaluation set once and reuses it for every downstream condition (all five
training seeds, the group-split control, and all three few-shot conditions), so every
comparison in `results/` is guaranteed to be over the same items.

## Model

- Base model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Adapter: QLoRA, r=16, α=16, dropout=0.0
- Trained adapters: **https://huggingface.co/nextmina/qwen2.5-3b-tombench-qlora** — the
  item-level seed-42 adapter that produced the released predictions (root), item-level seeds
  43–46 as same-configuration replications (`replication/`; the original seed 43–46 adapters
  were not saved), and the five story-disjoint adapters that produced the story-disjoint results
  (`story-disjoint/`). The model card maps each adapter to the results it produced. Load with
  Unsloth `FastLanguageModel.from_pretrained(<adapter folder>)` in the pinned environment
  (`requirements.lock.txt`) to reproduce the released predictions exactly.

### Training hyperparameters

| Item | Setting |
|---|---|
| Learning rate | 2e-4 |
| Effective batch size | 8 (per-device 4 × grad. accum. 2 on the NVIDIA L4 used for the released runs; the notebook picks the split from the GPU) |
| Max sequence length | 2048 |
| Optimizer | AdamW (8-bit) |
| LR scheduler | Cosine (warmup ratio 0.03) |
| Weight decay | 0.01 |
| Epochs | 4 (best validation-loss checkpoint selected) |
| Training seeds | 42, 43, 44, 45, 46 |

## Reproducing the experiments

1. Install dependencies: `pip install -r requirements.txt`
2. Open `notebooks/tombench_slm_qlora_complete_pipeline.ipynb` in Google Colab (GPU runtime
   required) and run it top to bottom. It clones the ToMBench/OpenToM/ToMi/HiToM source
   repositories and downloads SocialIQa itself; no manual data download is required. Each
   expensive step (baseline, each seed, the group split, each few-shot condition, and the
   fp16 control experiment) checks for its own already-saved output first, so the
   notebook can be safely re-run or resumed across sessions.
3. To recompute the statistics from the released `results/` folder without a GPU:
   ```
   python code/analyze_tier4_v2.py --dir results          # original-order tests; VERIFICATION_LOG_v2.csv
   python code/analyze_cyclic.py --dir results            # cyclic permutation, story-disjoint split; VERIFICATION_LOG_cyclic_v2.csv
   python code/analyze_position_bias_multiseed.py --dir results
   python code/analyze_position_bias_by_noptions.py --dir results
   python code/analyze_ability_and_split.py --dir results --splits splits
   ```
   The scripts in `code/` must be run from that folder or with it on `PYTHONPATH`, since
   they import `analyze_tier4_v2`.

## Data (original sources — not redistributed here)

| Benchmark | Use in this study | License (repository / dataset card, checked 2026-10-18) |
|---|---|---|
| ToMBench | fine-tuning and in-domain evaluation | MIT; the authors ask that it be used for evaluation only (our departure from this is discussed in the paper, Section 5.3) |
| ToMi | external evaluation only | CC BY-NC 4.0 |
| OpenToM | external evaluation only | CC BY-NC 4.0; the authors ask that it not be exposed where it may be used for LLM training |
| SocialIQa | external evaluation only | CC BY 4.0 |
| Hi-ToM | external evaluation only | Apache-2.0 (repository license) |

Obtain each benchmark from its original source cited in the paper.

The `splits/` index files and the `results/` CSVs are keyed by `item_id`/`prompt_hash` and contain
model predictions and scoring fields only; they do not contain benchmark story, question, or
option text. Results can be reconstructed once the original benchmark data is obtained and the
notebook is re-run (manifests are rebuilt deterministically from the same hashing scheme).

**Raw generations are available from the corresponding author on request.** They were removed
from the public files because chain-of-thought outputs quote benchmark source text verbatim,
which we do not redistribute, in keeping with the benchmarks' licenses and usage requests. In
their place each per-item file carries `marker_found`, `strict_format`, `extraction_rule`
(`marker` / `fallback` / `none`) and `raw_len`, which are sufficient to re-check answer extraction,
the strict-format analysis, and truncation (`code/strip_raw_generations.py`; see `CHANGELOG.md`).

## License

Code in this repository is released under the MIT License. The released split-index files and
result CSVs are our own derived outputs (item identifiers, hashes, model predictions, and scoring fields only).
Benchmark data remain under their original licenses and are not included here.

## Citation

```
@article{hong2026tomqlora,
  title   = {QLoRA Fine-Tuning for Theory-of-Mind Reasoning in a 3B-Parameter Small Language Model:
             A ToMBench-Based Cross-Benchmark Generalization Study},
  author  = {Hong, Ji-Hyeong and Yoo, Sang-Hyun},
  year    = {2026}
}
```
