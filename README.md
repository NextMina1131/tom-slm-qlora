# QLoRA Fine-Tuning for Theory-of-Mind Reasoning in Small Language Models

A ToMBench-Based Cross-Benchmark Generalization Study.

This repository contains the code, data-split indices, per-item prediction results, and
statistical analysis outputs for the paper *"QLoRA Fine-Tuning for Theory-of-Mind Reasoning
in Small Language Models: A ToMBench-Based Cross-Benchmark Generalization Study."*

We fine-tune the 4-bit pre-quantized **Qwen2.5-3B-Instruct** model (Unsloth) with a **QLoRA**
adapter on an ability-stratified ToMBench train/validation/test split, and analyze how the
in-domain gains transfer to four external benchmarks (ToMi, OpenToM, SocialIQa, Hi-ToM).

**Authors:** Ji-Hyeong Hong¹, Sang-Hyun Yoo²,\*
¹ Department of Cyber Hacking and Security, Seoul HOSEO Technical College, Seoul, Republic of Korea
² School of Computer Science and Engineering, Soongsil University, Seoul, Republic of Korea
\* Corresponding author: simonyoo@ssu.ac.kr

## Key result (primary run, seed 42)

| Benchmark | Base Acc. | Fine-tuned Acc. | Δ | Holm-corrected McNemar |
|---|---|---|---|---|
| ToMBench (in-domain, held-out) | 61.52 | 75.18 | **+13.66** | significant |
| ToMi | 76.30 | 83.00 | **+6.70** | significant |
| OpenToM | 63.40 | 59.50 | **−3.90** | significant |
| SocialIQa | 67.70 | 61.70 | **−6.00** | significant |
| Hi-ToM | 62.40 | 62.30 | −0.10 | not significant |

All deltas above are confirmed by an exact paired McNemar test (Holm-Bonferroni-corrected
across the five datasets) and a story-cluster bootstrap 95% CI, and replicate in sign across
five independent training seeds (42–46) for ToMBench, ToMi, and SocialIQa. Hi-ToM shows no
reliable effect in either direction across all five seeds (mean +0.94 ± 0.84 pp); a
gold-answer-position analysis (see `results/ANALYSIS_hitom_*_v2.csv`) shows this flat aggregate
conceals a large, statistically significant redistribution tied to the position of the correct
answer rather than to belief order, i.e. a response-position artifact rather than a genuine
higher-order-reasoning effect. This overall pattern supports the paper's central claim: ToM
ability in a small language model is not a single capability that improves uniformly under
fine-tuning, but a set of partially separable abilities that **transfer selectively**.

## Repository layout

```
code/           fine-tuning / inference / evaluation / analysis scripts, including
                gen_figure1.py (regenerates the manuscript's Figure 1 from results/ tables)
code/legacy/    original Chinese-language ToMBench runner scripts (run_api.py, run_huggingface.py,
                prompts.py, etc.); not used in the current pipeline — kept for reference only
notebooks/      tombench_slm_qlora_complete_pipeline.ipynb — the single, end-to-end notebook
                covering the full pipeline, including the story-level group-split control,
                fine-tuned few-shot evaluation, and fp16 control experiment; run this in Google Colab.
splits/         ability-stratified ToMBench train/val/test index files (item_id only, seed=42)
results/        per-item base vs. fine-tuned predictions (5 seeds), the story-level group-split
                control, three training-free few-shot conditions, the gold-answer-position and
                position-balanced-accuracy analysis outputs, and the statistical analysis outputs
                derived from them — see results/README.md.
                results/seed_epoch_recovery/ — per-epoch validation loss for the second replicate
                set of seeds 43-46 (Section 4.6 of the manuscript)
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
- Trained adapter weights (seed 42): **https://huggingface.co/nextmina/qwen2.5-3b-tombench-qlora**

### Training hyperparameters

| Item | Setting |
|---|---|
| Learning rate | 2e-4 |
| Effective batch size | 8 (per-device 2 × grad. accum. 4) |
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
3. To recompute the statistics from an existing `results/`-style folder without a GPU:
   `python code/analyze_tier4_v2.py --dir results`

## Data (original sources — not redistributed here)

- **ToMBench** — used for fine-tuning and in-domain evaluation, under its original license,
  for academic research purposes. Original repository: see paper reference.
- **ToMi**, **OpenToM**, **SocialIQa**, **Hi-ToM** — external transfer benchmarks; obtain
  from their respective original sources cited in the paper.

The `splits/` index files and the `results/` CSVs are keyed by `item_id`/`prompt_hash` only and
do not contain the underlying benchmark story or question text, so results can be reconstructed
once the original benchmark data is obtained and the notebook is re-run (manifests are rebuilt
deterministically from the same hashing scheme).

## License

Code in this repository is released under the MIT License. The released split-index files and
result CSVs are our own derived outputs (item identifiers, hashes, and model predictions only).
Benchmark data remain under their original licenses and are not included here.

## Citation

```
@article{hong2026tomqlora,
  title   = {QLoRA Fine-Tuning for Theory-of-Mind Reasoning in Small Language Models:
             A ToMBench-Based Cross-Benchmark Generalization Study},
  author  = {Hong, Ji-Hyeong and Yoo, Sang-Hyun},
  year    = {2026}
}
```
