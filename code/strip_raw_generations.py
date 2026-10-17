"""
Remove raw model generations from the released per-item CSVs.

Chain-of-thought generations quote benchmark source text verbatim, which the
benchmark licenses do not permit us to redistribute. This script replaces the
`raw` column with derived columns that keep every check in the paper
reproducible without the generated text:

    pred            extracted answer letter (unchanged)
    correct         scoring result (unchanged)
    marker_found    a [[X]] marker (case-insensitive, spaces allowed) was found
    strict_format   the generation contains an exact [[A-E]] marker
                    (used by the strict-format-subset analysis, Section 5.2)
    extraction_rule how the letter was obtained: marker / fallback / none
    raw_len         length of the generation in characters (truncation check)

Before dropping `raw`, the script re-extracts the letter from each generation
with the evaluation pipeline's rule and counts rows whose stored `pred` differs,
so the derived columns can be audited against the stored predictions.

Usage:
    python strip_raw_generations.py --dir results            # dry run, report only
    python strip_raw_generations.py --dir results --write    # rewrite files in place
"""
import argparse
import glob
import os
import re

import pandas as pd

ANS_RE = re.compile(r"\[\[\s*([A-E])\s*\]\]", re.IGNORECASE)   # evaluation pipeline
FB_RE = re.compile(r"\b([A-E])\b")                              # evaluation pipeline fallback
STRICT_FORMAT_RE = re.compile(r"\[\[[A-E]\]\]")                 # analyze_tier4_v2 strict-format check


def derive(raw):
    raw = "" if pd.isna(raw) else str(raw)
    m = ANS_RE.search(raw)
    if m:
        rule, letter = "marker", m.group(1).upper()
    else:
        f = FB_RE.search(raw)
        rule, letter = ("fallback", f.group(1).upper()) if f else ("none", "A")
    return {
        "marker_found": bool(m),
        "strict_format": bool(STRICT_FORMAT_RE.search(raw.strip())),
        "extraction_rule": rule,
        "raw_len": len(raw),
        "_letter": letter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        df = pd.read_csv(path)
        if "raw" not in df.columns:
            continue
        d = pd.DataFrame([derive(r) for r in df["raw"]], index=df.index)
        mismatch = int((d["_letter"] != df["pred"].astype(str)).sum()) if "pred" in df.columns else None
        for col in ["marker_found", "strict_format", "extraction_rule", "raw_len"]:
            df[col] = d[col]
        df = df.drop(columns=["raw"])
        rows.append({"file": os.path.basename(path), "rows": len(df), "pred_mismatch": mismatch,
                     "fallback": int((d["extraction_rule"] == "fallback").sum()),
                     "none": int((d["extraction_rule"] == "none").sum()),
                     "max_len": int(d["raw_len"].max())})
        if args.write:
            df.to_csv(path, index=False)
    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))
    print(f"\nfiles: {len(rep)}, rows: {rep['rows'].sum()}, pred mismatches: {rep['pred_mismatch'].sum()}")
    print("written" if args.write else "dry run (use --write to rewrite files)")


if __name__ == "__main__":
    main()
