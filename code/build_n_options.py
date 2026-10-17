"""Build an item-level answer-option-count table for all five evaluation sets.

The per-item result CSVs carry an ``n_options`` column that is only populated
for Hi-ToM. This script recovers the option count of every evaluated item from
the fixed evaluation manifests (the same prompts the model saw) and writes a
mapping keyed by ``item_id`` so that any result CSV can be joined against it.

The manifests contain the original story/question text and are not
redistributed (see the Data Availability statement), so this script needs a
local copy of them. Its output contains only identifiers and counts.

Usage:
    python code/build_n_options.py --manifest-dir <dir with split_test_full_v2.json
                                                   and manifest_<Benchmark>.json>
Output:
    results/item_n_options_v2.csv
"""
import argparse
import json
import os
import re

import pandas as pd

BENCHMARK_FILES = {
    'ToMBench': 'split_test_full_v2.json',
    'ToMi': 'manifest_ToMi.json',
    'OpenToM': 'manifest_OpenToM.json',
    'SocialIQa': 'manifest_SocialIQa.json',
    'HiToM': 'manifest_HiToM.json',
}

# Same option parser as the evaluation pipeline
OPT_BLOCK_RE = re.compile(r'\[Candidate Answers\]\n(.+)', re.S)
OPT_LINE_RE = re.compile(r'^([A-E])\.\s*(.+?)$', re.M)


def parse_letters(user_prompt):
    m = OPT_BLOCK_RE.search(user_prompt)
    block = m.group(1) if m else user_prompt
    return [l for l, _ in OPT_LINE_RE.findall(block)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-dir', required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'results', 'item_n_options_v2.csv'))
    args = ap.parse_args()

    rows = []
    for bm, fname in BENCHMARK_FILES.items():
        with open(os.path.join(args.manifest_dir, fname), encoding='utf-8') as f:
            items = json.load(f)
        for it in items:
            letters = parse_letters(it['user_prompt'])
            n = len(letters)
            # Options must be contiguous A, B, ... and contain the gold letter
            assert letters == [chr(65 + i) for i in range(n)], (bm, it['item_id'], letters)
            assert it['answer'] in letters, (bm, it['item_id'], it['answer'], letters)
            if it.get('n_options') is not None:
                assert int(it['n_options']) == n, (bm, it['item_id'], it['n_options'], n)
            rows.append({'benchmark': bm, 'item_id': it['item_id'], 'prompt_hash': it.get('prompt_hash'),
                         'task': it.get('task'), 'n_options': n, 'gold': it['answer']})

    df = pd.DataFrame(rows)
    assert not df.duplicated(['benchmark', 'item_id']).any()
    df.to_csv(args.out, index=False)

    print(f'wrote {len(df)} rows -> {args.out}')
    print(df.groupby(['benchmark', 'n_options']).size().to_string())


if __name__ == '__main__':
    main()
