"""exp83: 候选库规模扫描分析（|U| = 3,4,5 → 见过的老类 4,3,2）"""
import os, json, numpy as np
from scipy.stats import ttest_rel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
FILES = {3: 'exp82_dis.json', 4: 'exp82_nhu4.json', 5: 'exp82_nhu5.json'}
PRI = ['clm', 'ecfp2048', 'maccs', 'physchem', 'charbag', 'rdkit', 'onehot', 'rand384']
NAME = {'clm': 'CLM(本文)', 'ecfp2048': 'ECFP-2048', 'maccs': 'MACCS',
        'physchem': 'RDKit理化', 'charbag': 'SMILES字符袋', 'rdkit': 'RDKit描述符',
        'onehot': 'one-hot', 'rand384': '随机码本'}


def load(nhu):
    d = json.load(open(os.path.join(OUT, FILES[nhu]), encoding='utf-8'))
    lam0 = d['META']['lams'][0]
    return d, lam0


def cellmean(d, arm, kind='dis', field='auc_dis'):
    v = d['A'].get(arm, {}).get(kind, {})
    out = {}
    for key, lst in v.items():
        out[key] = float(np.mean([q[field] for q in lst]))
    return out


def get(d, lam0, p, kind='dis', field='auc_dis'):
    return cellmean(d, f'A:{p}|lam{lam0:g}', kind, field)


L = []
w = L.append
w('# exp83 · 候选库规模扫描：先验的耐久性\n')
w('> 零样本筛查的真实场景是候选库很大、见过的参考分子很少。')
w('> 本表把 `|U|`（未见候选数）从 3 推到 5，同时见过老类从 4 降到 2。\n')
w('| 先验 | ' + ' | '.join(f'\\|U\\|={n} 老类={7-n}' for n in (3, 4, 5)) + ' | 衰减 |')
w('|---|' + '---:|' * 4)

curves = {}
for p in PRI:
    row = []
    vals = []
    for n in (3, 4, 5):
        d, lam0 = load(n)
        cm = get(d, lam0, p)
        v = float(np.mean(list(cm.values()))) if cm else np.nan
        vals.append(v)
        row.append(f'{v:.3f}')
    curves[p] = vals
    w(f'| `{p}` | ' + ' | '.join(row) + f' | {vals[-1]-vals[0]:+.3f} |')
w('')

w('## top-1（在 |U| 个候选里选对）\n')
w('| 先验 | |U|=3 | |U|=4 | |U|=5 | 衰减 |')
w('|---|---:|---:|---:|---:|')
t1 = {}
for p in PRI:
    row = []
    vals = []
    for n in (3, 4, 5):
        d, lam0 = load(n)
        cm = get(d, lam0, p, 'top1', 'top1')
        v = float(np.mean(list(cm.values()))) if cm else np.nan
        vals.append(v)
        row.append(f'{v:.3f}')
    t1[p] = vals
    w(f'| `{p}` | ' + ' | '.join(row) + f' | {vals[-1]-vals[0]:+.3f} |')
w('')

w('## CLM vs 每个普通先验（逐规模配对检验，单元=(tag,候选对)）\n')
w('| 对手 | |U|=3 Δ / p | |U|=4 Δ / p | |U|=5 Δ / p |')
w('|---|---|---|---|')
for q in PRI[1:]:
    cells = []
    for n in (3, 4, 5):
        d, lam0 = load(n)
        A, B = get(d, lam0, 'clm'), get(d, lam0, q)
        ks = sorted(set(A) & set(B))
        if len(ks) < 3:
            cells.append('—')
            continue
        x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
        cells.append(f'{np.mean(x-y):+.3f} / p={ttest_rel(x, y).pvalue:.3g}')
    w(f'| `{q}` | ' + ' | '.join(cells) + ' |')
w('')

md = '\n'.join(L)
open(os.path.join(OUT, 'exp83_nscan.md'), 'w', encoding='utf-8').write(md)
print(md)
