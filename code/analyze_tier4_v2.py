"""
Offline statistical analysis for tombench_full_rerun_v2.ipynb outputs.

Does not require a GPU. Given a results directory containing per-item
prediction CSVs (baseline, fine-tuned per seed, few-shot conditions), this
script:

  1. Joins base/tuned predictions on a canonical item identifier and verifies
     prompt equality before computing any paired statistic, so that a paired
     test is never run on mismatched items.
  2. Computes exact McNemar tests and story-cluster bootstrap confidence
     intervals for the base-vs-fine-tuned accuracy change on each primary
     dataset, with Holm-Bonferroni correction applied within the pre-defined
     confirmatory comparison family (primary seed x 5 datasets). Multi-seed
     replication runs are reported as an exploratory family and are not
     further corrected.
  3. Builds a gold-answer-letter confusion matrix for Hi-ToM and runs a
     label-permutation test to check whether an accuracy change is
     concentrated on items sharing a particular gold-answer position, which
     would indicate a response-position bias rather than a genuine change in
     reasoning ability.

Usage:
    python analyze_tier4_v2.py --dir /path/to/tier4_v2_fixed

`--dir` should point at the results folder produced by
tombench_full_rerun_v2.ipynb (CSV files with item_id/prompt_hash/story_id
columns), downloaded as-is from the Drive output folder.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

RNG_SEED = 42
N_BOOTSTRAP = 10000


class ManifestIntegrityError(RuntimeError):
    pass


def load_csv(dir_, tag, ds):
    path = os.path.join(dir_, f"{tag}_{ds}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def matched_pair(left, right, where):
    """Inner-join two result sets on item_id (1:1) and verify prompt_hash
    equality. This is the minimum precondition for any paired test: without
    it, row-position alignment between two independently generated CSVs is
    not guaranteed, and a paired test would silently compare different
    items."""
    l_ids, r_ids = set(left["item_id"]), set(right["item_id"])
    only_l, only_r = l_ids - r_ids, r_ids - l_ids
    if only_l or only_r:
        raise ManifestIntegrityError(
            f"[{where}] item_id mismatch: {len(only_l)} item(s) only in left, "
            f"{len(only_r)} only in right -- these are not the same item set, "
            f"refusing to compute a paired test."
        )
    merged = left.merge(right, on="item_id", how="inner", validate="one_to_one", suffixes=("_L", "_R"))
    if len(merged) != len(left) or len(left) != len(right):
        raise ManifestIntegrityError(f"[{where}] row count changed after merge")
    mism = merged[merged["prompt_hash_L"] != merged["prompt_hash_R"]]
    if len(mism) > 0:
        raise ManifestIntegrityError(
            f"[{where}] {len(mism)} row(s) share an item_id but have different "
            f"prompt_hash values -- same item, different prompt content."
        )
    return merged


def mcnemar_exact(b, c):
    """Exact McNemar test via the binomial test on discordant pairs.
    b = items correct under the left condition only, c = items correct under
    the right condition only. Implemented directly on scipy.binomtest to
    avoid a statsmodels dependency."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def cluster_bootstrap_delta_ci(merged, correct_l_col, correct_r_col, cluster_col, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Resample whole clusters (source-story groups), not individual items.
    ToMBench items are not independent: several items are derived from the
    same source story, so a standard item-level bootstrap understates the
    true sampling variance. Resampling at the cluster level accounts for
    this within-story correlation."""
    rng = np.random.default_rng(seed)
    clusters = merged[cluster_col].values
    uniq_clusters = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq_clusters}
    correct_l = merged[correct_l_col].values
    correct_r = merged[correct_r_col].values

    deltas = np.empty(n_boot)
    n_clusters = len(uniq_clusters)
    for b in range(n_boot):
        sampled = rng.choice(uniq_clusters, size=n_clusters, replace=True)
        rows = np.concatenate([idx_by_cluster[c] for c in sampled])
        deltas[b] = correct_r[rows].mean() * 100 - correct_l[rows].mean() * 100

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_point": correct_r.mean() * 100 - correct_l.mean() * 100,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "n_items": len(merged),
        "n_clusters": n_clusters,
    }


def holm_correction(pvalues):
    """Holm-Bonferroni step-down procedure, implemented directly (no
    statsmodels dependency)."""
    idx = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        val = min((m - rank) * pvalues[i], 1.0)
        running_max = max(running_max, val)
        adj[i] = running_max
    return adj


def analyze_primary_datasets(dir_, datasets, seeds):
    """Base vs. each fine-tuned seed, per dataset: item-identity-checked
    join, exact McNemar test, and cluster bootstrap CI. The first seed in
    `seeds` is treated as the pre-registered primary comparison; the
    remaining seeds are reported as an exploratory multi-seed replication."""
    rows = []
    raw_pvalues = []
    keys = []
    for ds in datasets:
        base = load_csv(dir_, "baseline", ds)
        if base is None:
            print(f"[skip] baseline_{ds}.csv not found")
            continue
        for seed in seeds:
            tuned = load_csv(dir_, f"finetuned_seed{seed}", ds)
            if tuned is None:
                continue
            merged = matched_pair(base, tuned, f"{ds} baseline vs seed{seed}")
            b = ((merged["correct_L"] == 1) & (merged["correct_R"] == 0)).sum()
            c = ((merged["correct_L"] == 0) & (merged["correct_R"] == 1)).sum()
            p = mcnemar_exact(b, c)
            ci = cluster_bootstrap_delta_ci(merged, "correct_L", "correct_R", "story_id_L")
            rows.append({
                "dataset": ds, "seed": seed, "n_items": len(merged),
                "base_acc": merged["correct_L"].mean() * 100,
                "tuned_acc": merged["correct_R"].mean() * 100,
                "delta_pp": ci["delta_point"], "delta_ci_lo": ci["delta_ci_lo"], "delta_ci_hi": ci["delta_ci_hi"],
                "n_clusters": ci["n_clusters"], "mcnemar_b": int(b), "mcnemar_c": int(c), "mcnemar_p_raw": p,
            })
            raw_pvalues.append(p)
            keys.append((ds, seed))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Confirmatory family = all primary datasets at the first (pre-registered) seed.
    # Multi-seed replication rows are exploratory and are not independently corrected.
    primary_mask = df["seed"] == seeds[0]
    primary_p = df.loc[primary_mask, "mcnemar_p_raw"].values
    df.loc[primary_mask, "mcnemar_p_holm"] = holm_correction(primary_p)
    df["family"] = np.where(primary_mask, "confirmatory (primary seed x 5 datasets)", "exploratory (multi-seed replication)")
    return df


def analyze_hitom_confusion(dir_, seed):
    """Hi-ToM gold-answer-letter x predicted-letter confusion matrix and a
    label-permutation test. This checks whether an observed accuracy change
    is concentrated on items whose gold answer occupies a particular option
    position, which would indicate a response-position bias rather than a
    genuine change in higher-order belief reasoning."""
    base = load_csv(dir_, "baseline", "HiToM")
    tuned = load_csv(dir_, f"finetuned_seed{seed}", "HiToM")
    if base is None or tuned is None:
        print("[skip] HiToM baseline/finetuned CSV not found")
        return None
    merged = matched_pair(base, tuned, f"HiToM baseline vs seed{seed}")

    letters = sorted(set(merged["answer_L"].unique()) | set(merged["pred_L"].unique()) | set(merged["pred_R"].unique()))
    conf_base = pd.crosstab(merged["answer_L"], merged["pred_L"]).reindex(index=letters, columns=letters, fill_value=0)
    conf_tuned = pd.crosstab(merged["answer_R"], merged["pred_R"]).reindex(index=letters, columns=letters, fill_value=0)

    by_gold = merged.groupby("answer_L").agg(
        n=("item_id", "count"),
        base_acc=("correct_L", "mean"),
        tuned_acc=("correct_R", "mean"),
    )
    by_gold["delta_pp"] = (by_gold["tuned_acc"] - by_gold["base_acc"]) * 100
    by_gold["share_of_total_decline"] = np.nan
    total_decline_items = ((merged["correct_L"] == 1) & (merged["correct_R"] == 0)).sum()
    if total_decline_items > 0:
        for letter in by_gold.index:
            sub = merged[merged["answer_L"] == letter]
            declined = ((sub["correct_L"] == 1) & (sub["correct_R"] == 0)).sum()
            by_gold.loc[letter, "share_of_total_decline"] = declined / total_decline_items

    # Null model: if gold-answer position has no real effect, shuffling the
    # gold-letter labels should reproduce a max-min accuracy-change gap as
    # large as the observed one about as often as chance allows.
    rng = np.random.default_rng(RNG_SEED)
    observed_gap = by_gold["delta_pp"].max() - by_gold["delta_pp"].min()
    n_perm = 5000
    perm_gaps = np.empty(n_perm)
    gold_vals = merged["answer_L"].values.copy()
    correct_l = merged["correct_L"].values
    correct_r = merged["correct_R"].values
    for i in range(n_perm):
        shuffled = rng.permutation(gold_vals)
        tmp = pd.DataFrame({"g": shuffled, "dl": (correct_r - correct_l) * 100})
        gap = tmp.groupby("g")["dl"].mean()
        perm_gaps[i] = gap.max() - gap.min()
    perm_p = (perm_gaps >= observed_gap).mean()

    return {
        "confusion_base": conf_base,
        "confusion_tuned": conf_tuned,
        "by_gold_letter": by_gold,
        "observed_position_gap_pp": observed_gap,
        "permutation_p_value": perm_p,
        "n_permutations": n_perm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Path to the tier4_v2_fixed results folder")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--datasets", nargs="+", default=["ToMBench", "ToMi", "OpenToM", "SocialIQa", "HiToM"])
    args = ap.parse_args()

    print("=" * 70)
    print("Primary datasets: item-identity-checked join + McNemar + cluster bootstrap + Holm correction")
    print("=" * 70)
    try:
        df = analyze_primary_datasets(args.dir, args.datasets, args.seeds)
    except ManifestIntegrityError as e:
        print(f"\nAborted due to an item-identity integrity error: {e}")
        sys.exit(1)
    if len(df) == 0:
        print("No data to analyze. Check the --dir path.")
        sys.exit(1)
    print(df.to_string(index=False))
    out_csv = os.path.join(args.dir, "ANALYSIS_primary_results_v2.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    print("\n" + "=" * 70)
    print("Hi-ToM gold-letter confusion matrix + permutation test")
    print("=" * 70)
    try:
        hitom = analyze_hitom_confusion(args.dir, args.seeds[0])
    except ManifestIntegrityError as e:
        print(f"\nAborted due to an item-identity integrity error: {e}")
        hitom = None
    if hitom is not None:
        print("\n[Accuracy change by gold-answer letter]")
        print(hitom["by_gold_letter"].to_string())
        print(f"\nObserved gold-position gap (max - min delta, pp): {hitom['observed_position_gap_pp']:.2f}")
        print(f"Permutation p-value (n={hitom['n_permutations']}): {hitom['permutation_p_value']:.4f}")
        if hitom["permutation_p_value"] < 0.05:
            print("-> The accuracy change is statistically uneven across gold-answer letters, "
                  "consistent with a response-position bias rather than a uniform reasoning change.")
        else:
            print("-> The accuracy change is not statistically distinguishable from an even "
                  "effect across gold-answer letters.")
        hitom["by_gold_letter"].to_csv(os.path.join(args.dir, "ANALYSIS_hitom_by_gold_letter_v2.csv"))
        hitom["confusion_base"].to_csv(os.path.join(args.dir, "ANALYSIS_hitom_confusion_base_v2.csv"))
        hitom["confusion_tuned"].to_csv(os.path.join(args.dir, "ANALYSIS_hitom_confusion_tuned_v2.csv"))
        print("Wrote ANALYSIS_hitom_*_v2.csv")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    primary = df[df["family"].str.startswith("confirmatory")]
    for _, r in primary.iterrows():
        sig = "significant" if r["mcnemar_p_holm"] < 0.05 else "not significant"
        print(f"- {r['dataset']}: delta={r['delta_pp']:+.2f}pp [95% CI {r['delta_ci_lo']:+.2f}, {r['delta_ci_hi']:+.2f}] "
              f"(story-cluster bootstrap, n_clusters={r['n_clusters']}), McNemar p_holm={r['mcnemar_p_holm']:.4f} ({sig}, Holm-corrected)")
    print("\nNote: these figures are only valid for the tier4_v2_fixed/ output of an actual")
    print("tombench_full_rerun_v2.ipynb run. This script itself can also be run against")
    print("synthetic data to check the statistical logic, but any number quoted in the")
    print("manuscript must come from a real experimental run.")


if __name__ == "__main__":
    main()
