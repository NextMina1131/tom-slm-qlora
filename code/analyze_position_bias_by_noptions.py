"""
Gold-answer-position permutation test within option count (Reviewer 4, Comment 4).

ToMBench mixes two-option and four-option items, and OpenToM mixes two-option
location items with three-option attitude items. A permutation test that
shuffles gold letters across all items lets a letter that only exists in the
four-option items (C, D) be assigned to two-option items, so the null
distribution mixes item types. This script repeats the test of
analyze_tier4_v2.analyze_gold_position_bias separately within each option
count, for every seed, using only the released per-item CSVs and
results/item_n_options_v2.csv (no GPU, no new inference).

Usage:
    python analyze_position_bias_by_noptions.py --dir results
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_tier4_v2 import RNG_SEED, load_csv, matched_pair

N_PERM = 5000


def gap_test(gold, dl, rng):
    """Max-min per-gold-letter mean change and continuity-corrected permutation p."""
    df = pd.DataFrame({"g": gold, "dl": dl})
    by = df.groupby("g")["dl"].mean()
    observed = by.max() - by.min()
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        s = pd.Series(dl).groupby(rng.permutation(gold)).mean()
        perm[i] = s.max() - s.min()
    p = (np.sum(perm >= observed) + 1) / (N_PERM + 1)
    return observed, p, by


def run(dir_, datasets, seeds):
    nopt = pd.read_csv(os.path.join(dir_, "item_n_options_v2.csv"))
    rows = []
    for ds in datasets:
        k_map = nopt[nopt["benchmark"] == ds].set_index("item_id")["n_options"]
        base = load_csv(dir_, "baseline", ds)
        for seed in seeds:
            tuned = load_csv(dir_, f"finetuned_seed{seed}", ds)
            if base is None or tuned is None:
                print(f"[skip] {ds} seed{seed}")
                continue
            m = matched_pair(base, tuned, f"{ds} baseline vs seed{seed}")
            m["K"] = m["item_id"].map(k_map)
            if m["K"].isna().any():
                raise ValueError(f"{ds}: {int(m['K'].isna().sum())} items without n_options")
            for K, sub in m.groupby("K"):
                rng = np.random.default_rng(RNG_SEED)
                dl = ((sub["correct_R"] - sub["correct_L"]) * 100).values
                gap, p, by = gap_test(sub["answer_L"].values, dl, rng)
                rows.append({
                    "dataset": ds, "seed": seed, "n_options": int(K), "n_items": len(sub),
                    "delta_pp": dl.mean(), "position_gap_pp": gap, "permutation_p": p,
                    "significant_at_05": p < 0.05,
                    "per_letter_delta_pp": "; ".join(f"{k}:{v:+.1f}" for k, v in by.items()),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--datasets", nargs="+", default=["ToMBench", "OpenToM"])
    args = ap.parse_args()
    df = run(args.dir, args.datasets, args.seeds)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["per_letter_delta_pp"]).round(4).to_string(index=False))
    out = os.path.join(args.dir, "ANALYSIS_position_bias_by_noptions_v2.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\n[seeds significant within each option count]")
    print(df.groupby(["dataset", "n_options"]).agg(
        n_items=("n_items", "first"), seeds_sig=("significant_at_05", "sum"),
        gap_min=("position_gap_pp", "min"), gap_max=("position_gap_pp", "max")).round(1).to_string())


if __name__ == "__main__":
    main()
