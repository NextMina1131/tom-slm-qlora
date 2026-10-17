"""
Full ToMBench ability-level table and task-level split counts.

The manuscript reports the largest ability-level improvements (Table
"Top ToMBench ability-level improvements") and states that the full table,
including the abilities that did not improve, and the task-level split
counts are released with the code. This script produces both from the
released files:

  ANALYSIS_ability_level_v2.csv   every ATOMS ability label in the test split,
                                  seed-42 base vs fine-tuned accuracy, change
                                  and paired item-level bootstrap 95% CI
                                  (10,000 resamples, numpy seed 42)
  ANALYSIS_task_split_counts_v2.csv
                                  number of train/validation/test items per
                                  ToMBench task (splits/*.csv)

Usage:
    python analyze_ability_and_split.py --dir results --splits splits
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_tier4_v2 import N_BOOTSTRAP, RNG_SEED, load_csv, matched_pair


def item_bootstrap_ci(correct_l, correct_r, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(correct_l)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        deltas[b] = (correct_r[i].mean() - correct_l[i].mean()) * 100
    return np.percentile(deltas, [2.5, 97.5])


def ability_table(dir_, seed=42):
    base = load_csv(dir_, "baseline", "ToMBench")
    tuned = load_csv(dir_, f"finetuned_seed{seed}", "ToMBench")
    m = matched_pair(base, tuned, f"ToMBench baseline vs seed{seed} (ability table)")
    rows = []
    for ability, sub in m.groupby("ability_L"):
        l, r = sub["correct_L"].values, sub["correct_R"].values
        lo, hi = item_bootstrap_ci(l, r)
        rows.append({"ability": ability, "n": len(sub), "base_acc": l.mean() * 100, "tuned_acc": r.mean() * 100,
                     "delta_pp": (r.mean() - l.mean()) * 100, "ci_lo": lo, "ci_hi": hi,
                     "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows).sort_values("delta_pp", ascending=False)


def task_split_counts(splits_dir):
    parts = [pd.read_csv(os.path.join(splits_dir, f"split_{s}_index_v2.csv")) for s in ("train", "val", "test")]
    d = pd.concat(parts)
    c = d.groupby(["task", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    c["total"] = c.sum(axis=1)
    return c.sort_values("total", ascending=False).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    ab = ability_table(args.dir)
    print(ab.round(2).to_string(index=False))
    ab.to_csv(os.path.join(args.dir, "ANALYSIS_ability_level_v2.csv"), index=False)
    ts = task_split_counts(args.splits)
    print("\n" + ts.to_string(index=False))
    print(f"\ntotal: {ts[['train', 'val', 'test', 'total']].sum().to_dict()}")
    ts.to_csv(os.path.join(args.dir, "ANALYSIS_task_split_counts_v2.csv"), index=False)
    print("\nWrote ANALYSIS_ability_level_v2.csv, ANALYSIS_task_split_counts_v2.csv")


if __name__ == "__main__":
    main()
