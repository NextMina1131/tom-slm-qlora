"""ToMi: share of items whose only story-mentioned option is the gold answer,
and cyclic-permutation accuracy changes split by that flag (Section 4.6).

ToMi distractors are sampled from the pool of all ToMi answers (Table 4), so a
distractor often does not occur in the item's story. An option counts as
occurring in the story when its text, with underscores read as spaces, matches
a whole-word string in the story.

The ToMi evaluation manifest contains the original story text and is not
redistributed (see the Data Availability statement), so this script needs a
local copy of it. Its outputs contain only identifiers, flags and accuracies.

Usage:
    python code/analyze_tomi_distractors.py <manifest_ToMi.json> results results
Output:
    results/tomi_item_story_mention.csv      per-item flags
    results/ANALYSIS_tomi_story_mention.csv  base accuracy and changes by subset
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

manifest, res, out = sys.argv[1:4]


def norm(s):
    return s.lower().replace("_", " ")


rows = []
for m in json.load(open(manifest)):
    opts = dict(re.findall(r"^([A-E])\. (.+)$", m["user_prompt"].split("[Candidate Answers]")[1], re.M))
    story = norm(m["story"])
    inn = {k: bool(re.search(r"\b" + re.escape(norm(v)) + r"\b", story)) for k, v in opts.items()}
    g = m["answer"]
    rows.append({"item_id": m["item_id"], "n_options": len(opts), "gold_in_story": inn[g],
                 "n_distractors_in_story": sum(inn[k] for k in opts if k != g)})
items = pd.DataFrame(rows).set_index("item_id")
items["only_gold_in_story"] = items.gold_in_story & (items.n_distractors_in_story == 0)
items.to_csv(f"{out}/tomi_item_story_mention.csv")


def per_item(df):
    g = df.groupby("item_id").agg(lenient=("correct", "mean"), strict=("correct", "min"),
                                  n_choice=("pred_orig_letter", "nunique"))
    g["consistent"] = (g.n_choice == 1).astype(float)
    return g.join(items.only_gold_in_story)


def by_flag(g, cols):
    return g.groupby("only_gold_in_story")[cols].mean() * 100


M = ["lenient", "strict", "consistent"]
base = per_item(pd.read_csv(f"{res}/cyclic_itemlevel/cyclic_base_ToMi.csv"))
B = by_flag(base, M)
summary = []
for flag, r in B.iterrows():
    summary.append({"model": "base", "only_gold_in_story": flag, "n_items": int((items.only_gold_in_story == flag).sum()),
                    **{f"{m}_acc": r[m] for m in M}})

for split, pat in [("item-level", "cyclic_itemlevel/cyclic_seed4?_ToMi.csv"),
                   ("story-disjoint", "cyclic_storydisjoint/cyclic_sd_sdseed4?_ToMi.csv")]:
    d = np.stack([(by_flag(per_item(pd.read_csv(p)), M) - B).values for p in sorted(glob.glob(f"{res}/{pat}"))])
    n = len(d)
    mean, sd = d.mean(0), d.std(0, ddof=1)
    half = 2.776 * sd / np.sqrt(n)
    for i, flag in enumerate(B.index):
        summary.append({"model": f"fine-tuned {split} (5 seeds)", "only_gold_in_story": flag,
                        **{f"{m}_delta": mean[i, j] for j, m in enumerate(M)},
                        **{f"{m}_ci_lo": mean[i, j] - half[i, j] for j, m in enumerate(M)},
                        **{f"{m}_ci_hi": mean[i, j] + half[i, j] for j, m in enumerate(M)}})

for cond in ["synthetic_format_only_3shot", "answer_only_3shot", "rationale_cot_3shot"]:
    ds = []
    for e in range(3):
        orig = pd.read_csv(f"{res}/{cond}_ex{e}_ToMi.csv").assign(pred_orig_letter=lambda x: x.pred)
        a = pd.concat([pd.read_csv(f"{res}/fewshot_cyclic/fscyc_{cond}_ex{e}_ToMi.csv"), orig])
        assert (a.groupby("item_id").size() == 4).all()
        ds.append((by_flag(per_item(a), M) - B).values)
    d = np.stack(ds)
    mean, sd = d.mean(0), d.std(0, ddof=1)
    for i, flag in enumerate(B.index):
        summary.append({"model": f"base + {cond} (3 exemplar sets)", "only_gold_in_story": flag,
                        **{f"{m}_delta": mean[i, j] for j, m in enumerate(M)},
                        **{f"{m}_delta_sd": sd[i, j] for j, m in enumerate(M)}})

s = pd.DataFrame(summary)
s.to_csv(f"{out}/ANALYSIS_tomi_story_mention.csv", index=False)
print(items.only_gold_in_story.value_counts())
print(s.round(2).to_string())
