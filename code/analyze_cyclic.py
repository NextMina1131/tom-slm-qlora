"""
Cyclic-permutation re-evaluation (Section 4.7, Tables "Cyclic-permutation
re-evaluation" and "3-shot conditions under cyclic permutation").

Each item with K options was presented K times, with the options rotated so
that the gold answer occupied every position once. Given the released
rotation-level CSVs, this script

  1. verifies, for every model/condition file, that it covers exactly the
     same items as the base model with the same gold answer and option count,
     that every item has all K rotations, and that the item set equals the
     item set of the corresponding original-order per-item file; every check
     is logged to VERIFICATION_LOG_cyclic_v2.csv;
  2. computes per-item lenient (mean over rotations), strict (correct in all
     rotations) and consistent (same option content chosen in all rotations)
     scores, and the five-seed change from the base model with a between-seed
     95% t-interval, by benchmark, by option count and by OpenToM subtask;
  3. computes the change in lenient accuracy of the training-free 3-shot
     conditions relative to the base model under the same permutation;
  4. compares the unrotated (shift 0) predictions with the original-order
     per-item files and reports the number of differing items.

No GPU and no benchmark text are needed.

Usage:
    python analyze_cyclic.py --dir results
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

BENCHMARKS = ["ToMBench", "ToMi", "OpenToM", "SocialIQa", "HiToM"]
SEEDS = [42, 43, 44, 45, 46]
MEASURES = ["acc_orig", "lenient", "strict", "consistent"]

_LOG = []


class IntegrityError(RuntimeError):
    pass


def log(where, n_items, mismatches):
    _LOG.append({"where": where, "n_items": n_items, "mismatches": mismatches, "ok": mismatches == 0})
    if mismatches:
        raise IntegrityError(f"[{where}] {mismatches} mismatching item(s)")


def check_rotations(df, where):
    """Every item must have exactly K rows, with shifts 0..K-1."""
    g = df.groupby("item_id")
    k = g["n_options"].first()
    bad = (g.size() != k) | (g["shift_k"].nunique() != k) | (g["shift_k"].max() != k - 1)
    log(f"{where}: complete rotations", len(k), int(bad.sum()))


def check_same_items(base, other, where):
    """Same item set, same gold answer and option count as the base model."""
    b = base.groupby("item_id")[["gold_orig", "n_options"]].first()
    o = other.groupby("item_id")[["gold_orig", "n_options"]].first()
    only = len(set(b.index) ^ set(o.index))
    if only:
        log(f"{where}: same items as base", len(b), only)
    j = b.join(o, rsuffix="_o", how="inner")
    diff = int(((j["gold_orig"] != j["gold_orig_o"]) | (j["n_options"] != j["n_options_o"])).sum())
    log(f"{where}: same items as base", len(j), diff)


def check_against_original(rot, orig, where):
    """Item set of the rotation file equals that of the original-order file."""
    a, b = set(rot["item_id"]), set(orig["item_id"])
    log(f"{where}: item set equals original-order file", len(b), len(a ^ b))


def per_item(df):
    df = df.copy()
    df["c0"] = np.where(df["shift_k"] == 0, df["correct"], np.nan)
    keys = ["item_id", "task", "n_options"]
    df["task"] = df["task"].fillna("-")
    out = df.groupby(keys).agg(acc_orig=("c0", "max"), lenient=("correct", "mean"),
                               strict=("correct", "min"), n_choice=("pred_orig_letter", "nunique")).reset_index()
    out["consistent"] = (out["n_choice"] == 1).astype(float)
    return out.drop(columns="n_choice")


def t_interval(x):
    x = np.asarray(x, dtype=float)
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, len(x) - 1) * s / np.sqrt(len(x))
    return m, s, m - h, m + h


def five_seed_delta(base_items, seed_items, group):
    """Mean over seeds of (seed accuracy - base accuracy), in percentage points."""
    base = base_items.groupby(group)[MEASURES].mean() * 100
    rows = []
    for key in base.index:
        row = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        sel = base_items.set_index(group).index == key
        row["n_items"] = int(sel.sum())
        for m in MEASURES:
            d = [(si.groupby(group)[m].mean() * 100).loc[key] - base.loc[key, m] for si in seed_items]
            mean, sd, lo, hi = t_interval(d)
            row.update({f"{m}_mean": mean, f"{m}_sd": sd, f"{m}_ci_lo": lo, f"{m}_ci_hi": hi,
                        f"{m}_excl0": bool(lo > 0 or hi < 0)})
        if all(row[f"{m}_excl0"] for m in ["lenient", "strict", "consistent"]) and \
                len({np.sign(row[f"{m}_mean"]) for m in ["lenient", "strict", "consistent"]}) == 1:
            row["direction"] = "+" if row["lenient_mean"] > 0 else "-"
        else:
            row["direction"] = "none"
        rows.append(row)
    return pd.DataFrame(rows)


def read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_shift0(df, base, path):
    """The item-level rotation files hold shifts 1..K-1 only; the unrotated
    predictions of the same adapters, loaded the same way, are the
    re-evaluation files in reproducibility/ (as in the original analysis)."""
    if (df["shift_k"] == 0).any():
        return df
    d = pd.read_csv(path)[["item_id", "answer", "pred", "correct"]]
    info = base.drop_duplicates("item_id")[["item_id", "task", "n_options", "gold_orig"]]
    d = d.merge(info, on="item_id", how="left")
    if d["n_options"].isna().any():
        raise IntegrityError(f"{path}: item_id not in base rotation file")
    k0 = pd.DataFrame({"model": df["model"].iloc[0], "benchmark": df["benchmark"].iloc[0], "item_id": d["item_id"],
                       "task": d["task"], "n_options": d["n_options"].astype(int), "shift_k": 0,
                       "gold_orig": d["answer"], "new_gold": d["answer"], "pred": d["pred"],
                       "pred_orig_letter": d["pred"], "correct": d["correct"]})
    return pd.concat([df, k0], ignore_index=True)


def analyze_split(dir_, label, sub, model_fmt, base_tombench, orig_base, orig_model_fmt, shift0_fmt=None):
    """One split: base + five seeds, all five benchmarks."""
    seed_items = {s: [] for s in SEEDS}
    base_items, letters, k0 = [], [], []
    for bm in BENCHMARKS:
        base = read(os.path.join(dir_, base_tombench if bm == "ToMBench" else f"cyclic_itemlevel/cyclic_base_{bm}.csv"))
        check_rotations(base, f"{label} base {bm}")
        check_against_original(base, read(os.path.join(dir_, orig_base[bm])), f"{label} base {bm}")
        bi = per_item(base)
        bi["benchmark"] = bm
        base_items.append(bi)
        for s in SEEDS:
            df = read(os.path.join(dir_, sub, model_fmt.format(seed=s, bm=bm)))
            if shift0_fmt:
                df = add_shift0(df, base, os.path.join(dir_, shift0_fmt.format(seed=s, bm=bm)))
            check_rotations(df, f"{label} seed{s} {bm}")
            check_same_items(base, df, f"{label} seed{s} {bm}")
            si = per_item(df)
            si["benchmark"] = bm
            seed_items[s].append(si)
            if bm == "HiToM":
                letters.append({"split": label, "seed": s, "E_choices": int((df["pred"] == "E").sum()),
                                "rotations": len(df), "strict_items": int(si["strict"].sum())})
            orig_path = os.path.join(dir_, orig_model_fmt(s, bm))
            if os.path.exists(orig_path):
                o = read(orig_path).set_index("item_id")["pred"]
                z = df[df["shift_k"] == 0].set_index("item_id")["pred_orig_letter"]
                k0.append({"split": label, "seed": s, "benchmark": bm, "n_items": len(z),
                           "differ_from_original_order_file": int((z != o.reindex(z.index)).sum())})
    base_items = pd.concat(base_items)
    seed_items = [pd.concat(seed_items[s]) for s in SEEDS]
    out = {
        "by_benchmark": five_seed_delta(base_items, seed_items, ["benchmark"]),
        "by_noptions": five_seed_delta(base_items, seed_items, ["benchmark", "n_options"]),
        "opentom_subtask": five_seed_delta(base_items[base_items.benchmark == "OpenToM"],
                                           [x[x.benchmark == "OpenToM"] for x in seed_items], ["task"]),
    }
    return out, pd.DataFrame(letters), pd.DataFrame(k0), base_items


def analyze_fewshot(dir_, base_items):
    rows = []
    base = base_items.groupby("benchmark")["lenient"].mean() * 100
    for path in sorted(glob.glob(os.path.join(dir_, "fewshot_cyclic", "fscyc_*.csv"))):
        cond, ex, bm = re.match(r"fscyc_(.+)_ex(\d)_(\w+)\.csv", os.path.basename(path)).groups()
        base_df = read(os.path.join(dir_, "cyclic_itemlevel", f"cyclic_base_{bm}.csv"))
        # shift 0 of each condition is its released original-order file
        df = add_shift0(pd.read_csv(path).assign(model=cond, benchmark=bm), base_df,
                        os.path.join(dir_, f"{cond}_ex{ex}_{bm}.csv"))
        check_rotations(df, f"fewshot {cond} ex{ex} {bm}")
        check_same_items(base_df, df, f"fewshot {cond} ex{ex} {bm}")
        rows.append({"condition": cond, "exemplar_draw": int(ex), "benchmark": bm,
                     "lenient_delta_pp": per_item(df)["lenient"].mean() * 100 - base.loc[bm]})
    df = pd.DataFrame(rows)
    return df.groupby(["condition", "benchmark"])["lenient_delta_pp"].agg(["mean", "std", "count"]).reset_index()


def storydisjoint_original_order(dir_):
    """Table "Story-disjoint split": original-order change for the five
    story-disjoint seeds, with exact McNemar tests Holm-corrected within each
    seed's five datasets. Pairs are joined with analyze_tier4_v2.matched_pair,
    which checks item identifiers and prompt hashes."""
    from analyze_tier4_v2 import _INTEGRITY_CHECK_LOG, holm_correction, matched_pair, mcnemar_exact
    rows = []
    for s in SEEDS:
        ps = []
        for bm in BENCHMARKS:
            base = read(os.path.join(dir_, "groupsplit_base_ToMBench.csv" if bm == "ToMBench" else f"baseline_{bm}.csv"))
            name = f"groupsplit_finetuned_{bm}.csv" if s == 42 else f"groupsplit_finetuned_seed{s}_{bm}.csv"
            m = matched_pair(base, read(os.path.join(dir_, name)), f"story-disjoint base vs seed{s} {bm}")
            b = int(((m["correct_L"] == 1) & (m["correct_R"] == 0)).sum())
            c = int(((m["correct_L"] == 0) & (m["correct_R"] == 1)).sum())
            ps.append(mcnemar_exact(b, c))
            rows.append({"seed": s, "benchmark": bm, "n_items": len(m),
                         "delta_pp": (m["correct_R"].mean() - m["correct_L"].mean()) * 100})
        for r, ph in zip(rows[-len(BENCHMARKS):], holm_correction(ps)):
            r["p_holm"] = ph
    df = pd.DataFrame(rows)
    out = []
    for bm, g in df.groupby("benchmark", sort=False):
        mean, sd, lo, hi = t_interval(g["delta_pp"])
        out.append({"benchmark": bm, "mean_pp": mean, "sd_pp": sd, "ci_lo": lo, "ci_hi": hi,
                    "seeds_significant": int((g["p_holm"] < 0.05).sum())})
    for e in _INTEGRITY_CHECK_LOG:
        _LOG.append(e)
        if not e["ok"]:
            raise IntegrityError(e["where"])
    return df, pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir
    pd.set_option("display.width", 250)

    item_orig = {bm: f"baseline_{bm}.csv" for bm in BENCHMARKS}
    sd_orig = dict(item_orig, ToMBench="groupsplit_base_ToMBench.csv")
    results = {}
    try:
        results["itemlevel"] = analyze_split(
            d, "item-level", "cyclic_itemlevel", "cyclic_seed{seed}_{bm}.csv",
            "cyclic_itemlevel/cyclic_base_ToMBench.csv", item_orig, lambda s, bm: f"finetuned_seed{s}_{bm}.csv",
            shift0_fmt="reproducibility/repro_unslothload_seed{seed}_{bm}.csv")
        results["storydisjoint"] = analyze_split(
            d, "story-disjoint", "cyclic_storydisjoint", "cyclic_sd_sdseed{seed}_{bm}.csv",
            "cyclic_storydisjoint/cyclic_sd_base_ToMBench.csv", sd_orig,
            # the seed-42 story-disjoint files predate the per-seed naming
            lambda s, bm: f"groupsplit_finetuned_{bm}.csv" if s == 42 else f"groupsplit_finetuned_seed{s}_{bm}.csv")
        fewshot = analyze_fewshot(d, results["itemlevel"][3])
        sd_by_seed, sd_orig = storydisjoint_original_order(d)
    except IntegrityError as e:
        print(f"[integrity check failed] {e}")
        pd.DataFrame(_LOG).to_csv(os.path.join(d, "VERIFICATION_LOG_cyclic_v2.csv"), index=False)
        sys.exit(1)

    for name, (tabs, letters, k0, _) in results.items():
        print("\n" + "=" * 70 + f"\nCyclic permutation, {name} split: five-seed change from base (pp)\n" + "=" * 70)
        for tname, t in tabs.items():
            cols = [c for c in t.columns if c.endswith("_mean") or c in ("benchmark", "n_options", "task", "n_items", "direction")]
            print(f"\n[{tname}]\n" + t[cols].round(2).to_string(index=False))
            t.to_csv(os.path.join(d, f"ANALYSIS_cyclic_{name}_{tname}_v2.csv"), index=False)
        letters["E_share_pct"] = letters["E_choices"] / letters["rotations"] * 100
        letters["strict_over_E_bound"] = letters["strict_items"] / letters["E_choices"]
        print("\n[Hi-ToM option E]\n" + letters.round(3).to_string(index=False))
        letters.to_csv(os.path.join(d, f"ANALYSIS_cyclic_{name}_hitom_E_v2.csv"), index=False)
        if len(k0):
            print("\n[shift 0 vs original-order per-item file: differing items]")
            print(k0.pivot(index="seed", columns="benchmark", values="differ_from_original_order_file").to_string())
            k0.to_csv(os.path.join(d, f"ANALYSIS_cyclic_{name}_shift0_vs_original_v2.csv"), index=False)

    print("\n" + "=" * 70 + "\n3-shot conditions under cyclic permutation: lenient change from base (pp)\n" + "=" * 70)
    print(fewshot.round(2).to_string(index=False))
    fewshot.to_csv(os.path.join(d, "ANALYSIS_cyclic_fewshot_v2.csv"), index=False)

    print("\n" + "=" * 70 + "\nStory-disjoint split, original order: five-seed change from base (pp)\n" + "=" * 70)
    print(sd_orig.round(2).to_string(index=False))
    sd_orig.to_csv(os.path.join(d, "ANALYSIS_storydisjoint_original_order_v2.csv"), index=False)
    sd_by_seed.to_csv(os.path.join(d, "ANALYSIS_storydisjoint_original_order_by_seed_v2.csv"), index=False)

    log_df = pd.DataFrame(_LOG)
    log_df.to_csv(os.path.join(d, "VERIFICATION_LOG_cyclic_v2.csv"), index=False)
    print(f"\n{len(log_df)} integrity checks, {int((~log_df['ok']).sum())} failed. "
          f"Wrote VERIFICATION_LOG_cyclic_v2.csv")


if __name__ == "__main__":
    main()
