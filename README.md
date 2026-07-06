# QLoRA Fine-tuning for Theory of Mind Reasoning in Small Language Models

A Cross-Benchmark Generalization Study on ToMBench.

This repository contains the code, data-split indices, per-item prediction results, and
statistical analysis outputs for the paper *"QLoRA Fine-tuning for Theory of Mind Reasoning
in Small Language Models: A Cross-Benchmark Generalization Study on ToMBench."*

We fine-tune the 4-bit pre-quantized **Qwen2.5-3B-Instruct** model (Unsloth) with a **QLoRA**
adapter on a stratified ToMBench train/validation/test split, and analyze how the in-domain
gains transfer to four external benchmarks (ToMi, OpenToM, SocialIQa, Hi-ToM).

## Key result

| Benchmark | Metric | Base | Fine-tuned | Δ | McNemar |
|---|---|---|---|---|---|
| ToMBench (in-domain, held-out) | Acc. | 61.52 | 76.13 | **+14.61** | sig. |
| ToMi | Acc. | — | — | **+4.90** | sig. |
| OpenToM | macro-F1 | — | — | +2.67 | n.s. (acc) |
| SocialIQa | Acc. | — | — | **−4.50** | sig. |
| Hi-ToM | Acc. | — | — | **−5.00** | sig. |

The gains and drops are both statistically significant (paired McNemar + bootstrap 95% CI),
suggesting ToM ability is not a single learnable capability but a set of distinct abilities
that **transfer selectively**.

## Repository layout

```
code/       fine-tuning / inference / evaluation scripts
notebooks/  full training pipeline (Colab)
splits/     stratified ToMBench train/val/test index files (seed=42)
results/    per-item base vs fine-tuned predictions + aggregate metrics
results/stats/  paired tables, McNemar p-values, bootstrap CIs
docs/       ATOMS ability mapping
```

## Model

- Base model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Adapter: QLoRA, r=16, α=16
- Trained adapter weights: **https://huggingface.co/NextMina1131/qwen2.5-3b-tombench-qlora**

## Reproducing the experiments

1. Install dependencies: `pip install -r requirements.txt`
2. Obtain the benchmark data from the original sources (see **Data** below). We do **not**
   redistribute raw benchmark data.
3. Use the index files in `splits/` to reconstruct the exact train/val/test partition.
4. Run the training pipeline in `notebooks/tombench_full_pipeline.ipynb`.
5. Evaluate with `code/run_huggingface.py` (local model) or `code/run_api.py` (API), then
   score with `code/get_results.py`.

## Data (original sources — not redistributed here)

- **ToMBench** — used for fine-tuning and in-domain evaluation, under its original license,
  for academic research purposes. Original repository: see paper reference.
- **ToMi**, **OpenToM**, **SocialIQa**, **Hi-ToM** — external transfer benchmarks; obtain
  from their respective original sources cited in the paper.

The `splits/` index files record the exact items used, so results can be reproduced once the
original data is obtained. The result CSVs in `results/` have had the raw benchmark question
text (`user_prompt`) removed for the same licensing reason — see `results/README.md`.

## License

Code in this repository is released under the MIT License. The released split-index files and
result CSVs are our own outputs. Benchmark data remain under their original licenses and are
not included here.

## Citation

```
@article{hong2026tomqlora,
  title   = {QLoRA Fine-tuning for Theory of Mind Reasoning in Small Language Models:
             A Cross-Benchmark Generalization Study on ToMBench},
  author  = {Hong, Ji-Hyeong},
  year    = {2026}
}
```
