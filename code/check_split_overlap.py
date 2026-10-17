"""Train/val/test overlap check for the released item-level split.

Usage: python check_split_overlap.py <run_dir> [<published_results_dir>]
  run_dir: folder with split_{train,val,test}_full_v2.json and manifest_*.json
  published_results_dir: folder with finetuned_seed42_ToMBench.csv etc. (default: run_dir)
"""
import json, sys, os, itertools
import pandas as pd


def norm(s):
    return ' '.join(str(s).split())


def load(path):
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))


run_dir = sys.argv[1]
pub_dir = sys.argv[2] if len(sys.argv) > 2 else run_dir
sp = {k: load(f'{run_dir}/split_{k}_full_v2.json') for k in ['train', 'val', 'test']}
for d in sp.values():
    d['sq'] = d['story'].map(norm) + '||' + d['question'].map(norm)
    d['story_n'] = d['story'].map(norm)

print('[sizes]', {k: len(d) for k, d in sp.items()})
for k, d in sp.items():
    print(f'  {k}: duplicate item_id within split = {d.item_id.duplicated().sum()}')

print('\n[ToMBench pairwise overlap]')
print('pair         item_id  prompt_hash  story+question  story')
for a, b in itertools.combinations(['train', 'val', 'test'], 2):
    A, B = sp[a], sp[b]
    r = [len(set(A[c]) & set(B[c])) for c in ['item_id', 'prompt_hash', 'sq', 'story_n']]
    print(f'{a:>5}-{b:<5}  {r[0]:>7}  {r[1]:>11}  {r[2]:>14}  {r[3]:>5}')

trainval = pd.concat([sp['train'], sp['val']])
shared_train = sp['test']['story_n'].isin(set(sp['train']['story_n']))
print(f'test items whose story also appears in train: {shared_train.mean() * 100:.1f}% ({shared_train.sum()})')
test_story_shared = sp['test']['story_n'].isin(set(trainval['story_n'])).mean() * 100
print(f'test items whose story also appears in train/val: {test_story_shared:.1f}% (item-level split, expected > 0)')

# Released predictions must cover exactly the test split
for tag in ['baseline', 'finetuned_seed42']:
    p = f'{pub_dir}/{tag}_ToMBench.csv'
    if os.path.exists(p):
        ids = set(pd.read_csv(p)['item_id'])
        print(f'{tag}_ToMBench.csv item_id == test split: {ids == set(sp["test"].item_id)} ({len(ids)})')

print('\n[External benchmarks vs ToMBench train+val]')
print('benchmark   n     item_id  prompt_hash  story  story+question')
tv_cols = {c: set(trainval[c]) for c in ['item_id', 'prompt_hash', 'story_n', 'sq']}
for name in ['ToMi', 'OpenToM', 'SocialIQa', 'HiToM']:
    p = f'{run_dir}/manifest_{name}.json'
    if not os.path.exists(p):
        continue
    m = load(p)
    m['story_n'] = m['story'].map(norm)
    m['sq'] = m['story_n'] + '||' + m['question'].map(norm)
    r = [int(m[c].isin(tv_cols[c]).sum()) for c in ['item_id', 'prompt_hash', 'story_n', 'sq']]
    print(f'{name:<10} {len(m):>5}  {r[0]:>7}  {r[1]:>11}  {r[2]:>5}  {r[3]:>14}')
