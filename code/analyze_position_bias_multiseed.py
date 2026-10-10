"""
Multi-seed extension of the gold-answer-position bias analysis.

analyze_tier4_v2.py's analyze_gold_position_bias and
analyze_residual_after_extreme_stratum only ran on the primary seed (42).
Reviewers 3 and 4 asked whether the observed position-gap and residual
effects are specific to that one seed or hold across all five training
seeds. This script re-runs both checks for every (dataset, seed) pair using
the already-released per-item prediction CSVs for seeds 43-46, and adds a
per-dataset seed-consistency summary.

Requires no GPU and no new fine-tuning or inference: it only re-slices
per-item predictions that are already in `results/`.

Usage:
    python analyze_position_bias_multiseed.py --dir results
"""
import argparse
import os

import pandas as pd

from analyze_tier4_v2 import (
    ManifestIntegrityError,
    analyze_gold_position_bias,
    analyze_residual_after_extreme_stratum,
)


def run_multiseed_position_bias(dir_, datasets, seeds):
    """Table 18, extended from seed 42 only to all five seeds."""
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                result = analyze_gold_position_bias(dir_, ds, seed)
            except ManifestIntegrityError as e:
                print(f"[{ds} seed{seed}] integrity error: {e}")
                continue
            if result is None:
                print(f"[skip] {ds} seed{seed}: baseline or finetuned CSV not found")
                continue
            rows.append({
                "dataset": ds,
                "seed": seed,
                "n_items": result["n_items"],
                "overall_delta_pp": result["overall_delta_pp"],
                "position_gap_pp": result["observed_position_gap_pp"],
                "permutation_p": result["permutation_p_value"],
                "significant_at_05": result["permutation_p_value"] < 0.05,
            })
    return pd.DataFrame(rows)


def run_multiseed_residual(dir_, datasets, seeds):
    """Table 19, extended from seed 42 only to all five seeds."""
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = analyze_residual_after_extreme_stratum(dir_, ds, seed)
            except ManifestIntegrityError as e:
                print(f"[{ds} seed{seed}] integrity error: {e}")
                continue
            if r is None:
                print(f"[skip] {ds} seed{seed}: baseline or finetuned CSV not found")
                continue
            r["seed"] = seed
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_by_dataset(position_df, residual_df, n_seeds_expected):
    """One row per dataset: how many of the n_seeds_expected seeds agree,
    and how much the position gap / residual effect varies across seeds."""
    pos_rows = []
    for ds, sub in position_df.groupby("dataset"):
        pos_rows.append({
            "dataset": ds,
            "n_seeds_run": len(sub),
            "n_seeds_significant": int(sub["significant_at_05"].sum()),
            "position_gap_mean_pp": sub["position_gap_pp"].mean(),
            "position_gap_sd_pp": sub["position_gap_pp"].std(ddof=1) if len(sub) > 1 else 0.0,
            "position_gap_min_pp": sub["position_gap_pp"].min(),
            "position_gap_max_pp": sub["position_gap_pp"].max(),
        })
    pos_summary = pd.DataFrame(pos_rows)

    res_rows = []
    for ds, sub in residual_df.groupby("dataset"):
        survives = sub["still_significant_at_05"] & sub["same_direction_as_full"]
        res_rows.append({
            "dataset": ds,
            "n_seeds_run": len(sub),
            "n_seeds_residual_survives": int(survives.sum()),
            "residual_delta_mean_pp": sub["residual_delta_pp"].mean(),
            "residual_delta_sd_pp": sub["residual_delta_pp"].std(ddof=1) if len(sub) > 1 else 0.0,
            "excluded_letter_consistent": sub["excluded_letter"].nunique() == 1,
        })
    res_summary = pd.DataFrame(res_rows)

    for df_ in (pos_summary, res_summary):
        if len(df_) and df_["n_seeds_run"].max() < n_seeds_expected:
            print(f"\n[warning] expected {n_seeds_expected} seeds but some datasets have fewer "
                  f"-- check for missing finetuned_seed<N>_<dataset>.csv files.")
            break

    return pos_summary, res_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Path to the results folder")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--datasets", nargs="+", default=["ToMBench", "ToMi", "OpenToM", "SocialIQa", "HiToM"])
    args = ap.parse_args()

    print("=" * 70)
    print(f"Gold-answer-position bias, {len(args.datasets)} benchmarks x {len(args.seeds)} seeds")
    print("=" * 70)
    position_df = run_multiseed_position_bias(args.dir, args.datasets, args.seeds)
    if len(position_df) == 0:
        print("No data produced. Check --dir and that finetuned_seed<N>_<dataset>.csv files exist.")
        return
    print(position_df.to_string(index=False))
    out1 = os.path.join(args.dir, "ANALYSIS_position_bias_multiseed_v2.csv")
    position_df.to_csv(out1, index=False)
    print(f"\nWrote {out1}")

    print("\n" + "=" * 70)
    print("Residual effect after excluding the extreme stratum, all seeds")
    print("=" * 70)
    residual_df = run_multiseed_residual(args.dir, args.datasets, args.seeds)
    residual_cols = ["dataset", "seed", "full_delta_pp", "excluded_letter", "excluded_letter_delta_pp",
                      "residual_delta_pp", "residual_ci_lo", "residual_ci_hi",
                      "residual_mcnemar_p", "same_direction_as_full", "still_significant_at_05"]
    print(residual_df[residual_cols].to_string(index=False))
    out2 = os.path.join(args.dir, "ANALYSIS_residual_multiseed_v2.csv")
    residual_df.to_csv(out2, index=False)
    print(f"\nWrote {out2}")

    print("\n" + "=" * 70)
    print("Per-dataset consistency summary across seeds")
    print("=" * 70)
    pos_summary, res_summary = summarize_by_dataset(position_df, residual_df, len(args.seeds))
    print("\n[Position-gap consistency]")
    print(pos_summary.to_string(index=False))
    print("\n[Residual-after-extreme-stratum consistency]")
    print(res_summary.to_string(index=False))
    out3 = os.path.join(args.dir, "ANALYSIS_position_bias_seed_consistency_v2.csv")
    pos_summary.merge(res_summary, on="dataset", suffixes=("_pos", "_res")).to_csv(out3, index=False)
    print(f"\nWrote {out3}")

    print("\n" + "=" * 70)
    print("Note")
    print("=" * 70)
    print("These numbers come from re-slicing the already-released per-item prediction")
    print("CSVs (baseline_<ds>.csv, finetuned_seed<N>_<ds>.csv) -- no GPU time or new")
    print("inference was used. Report the per-dataset consistency summary above when")
    print("writing up whether the seed-42 position-gap and residual findings replicate.")


if __name__ == "__main__":
    main()
