"""exp81d: 合并 7 个实体 json，逐实体重算三轴结果 + 判废线。
口径：group 协议 / CLEAN 池 5 实体 × 3 seeds / 逐实体配对 t。
"""
import json, os, glob, numpy as np
from scipy.stats import ttest_rel

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
EIDS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07']
CLEAN = ['A01', 'A02', 'A03', 'A04', 'A07']
TASKS = ['same-matrix', 'old-heldout']
SHOTS = ['0', '5']

# ---- 合并 ----
M = {}
for p in sorted(glob.glob(os.path.join(OUT, 'exp81_zoo_A*.json'))):
    d = json.load(open(p, encoding='utf-8'))
    for bk in ['B', 'P', 'C', 'O']:
        for k in SHOTS:
            for t in TASKS:
                for a, by in d[bk][k][t].items():
                    for e, recs in by.items():
                        M.setdefault(bk, {}).setdefault(k, {}).setdefault(t, {}) \
                         .setdefault(a, {}).setdefault(e, []).extend(recs)

json.dump(M, open(os.path.join(OUT, 'exp81_merged.json'), 'w', encoding='utf-8'))


def ev(bk, k, t, a, e, key='auc'):
    v = M.get(bk, {}).get(k, {}).get(t, {}).get(a, {}).get(e)
    return float(np.mean([q[key] for q in v])) if v else None


def pm(bk, k, t, a, pool=CLEAN, key='auc'):
    v = [ev(bk, k, t, a, e, key) for e in pool]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def paired(bk1, a1, bk2, a2, k, t, pool=CLEAN, key='auc'):
    x = np.array([ev(bk1, k, t, a1, e, key) for e in pool], float)
    y = np.array([ev(bk2, k, t, a2, e, key) for e in pool], float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None
    return dict(a=float(x.mean()), b=float(y.mean()), d=float((x - y).mean()),
                p=float(ttest_rel(x, y).pvalue), n=len(x))


def line(s=''):
    print(s)


line('=' * 100)
line('T1 · 算法动物园（原始光谱 / 单类族 / 本文 stage1 表示），AUC，CLEAN 池 5 实体×3 seeds')
line('=' * 100)
for t in TASKS:
    for k in SHOTS:
        arms = sorted(a for a in M['B'][k][t] if a.startswith('B:'))
        vals = sorted(((pm('B', k, t, a), a) for a in arms if pm('B', k, t, a) is not None),
                      reverse=True)
        ours = pm('O', k, t, 'ours-det' if t == 'same-matrix' else 'ours-id-m')
        line(f'\n[{t}  k={k}]  本文 O = {ours:.3f}')
        line('  ' + ' | '.join(f'{a[2:]}={v:.3f}' for v, a in vals))
        above = sum(1 for v, a in vals if v > ours)
        line(f'  → 本文低于 {above}/{len(vals)} 个普通算法；最强基线 {vals[0][1][2:]}={vals[0][0]:.3f} '
             f'(Δ={ours - vals[0][0]:+.3f})')

line()
line('=' * 100)
line('T2 · 先验动物园（同一个岭回归头，只换码本），识别 old-heldout')
line('=' * 100)
for k in SHOTS:
    for suf in ['', '-m']:
        priors = ['clm', 'ecfp384', 'ecfp2048', 'maccs', 'avalon', 'rdkit',
                  'physchem', 'charbag', 'onehot', 'rand384']
        vals = sorted(((pm('P', k, 'old-heldout', f'P:{p}{suf}'), p)
                       for p in priors if pm('P', k, 'old-heldout', f'P:{p}{suf}') is not None),
                      reverse=True)
        line(f'\n[k={k}  suffix={suf or "raw"}]  ' +
             ' | '.join(f'{p}={v:.3f}' for v, p in vals))
        c = paired('P', f'P:clm{suf}', 'P', f'P:rand384{suf}', k, 'old-heldout')
        line(f'  Δ(clm−rand) = {c["d"]:+.3f}  p={c["p"]:.3g}  '
             f'CLM 排名 {[p for v, p in vals].index("clm") + 1}/{len(vals)}')
        best_other = next((p for v, p in vals if p not in ('clm', 'rand384')), None)
        if best_other:
            c2 = paired('P', 'P:clm' + suf, 'P', f'P:{best_other}{suf}', k, 'old-heldout')
            line(f'  Δ(clm−最强普通先验 {best_other}) = {c2["d"]:+.3f}  p={c2["p"]:.3g}')

line()
line('=' * 100)
line('T3 · ★ 判废线：先验 × 普通算法（条件化打分器 g([z,S_T])），同架构只换码本')
line('   Δ_prior = clm − rand384；正 = 化学先验带来净增益')
line('=' * 100)
ALGOS = ['logreg', 'rf', 'histgb', 'svm_rbf', 'mlp', 'knn']
for t in TASKS:
    for k in SHOTS:
        line(f'\n[{t}  k={k}]')
        line('  算法       clm(raw)  rand(raw)   Δ_prior    p      |  clm-m   rand-m   Δ_m     p_m')
        pos = tot = sig = 0
        for a in ALGOS:
            cr = f'C:{a}|clm'; rr = f'C:{a}|rand384'
            c = paired('C', cr, 'C', rr, k, t)
            cm = f'C:{a}|clm-m'; rm = f'C:{a}|rand384-m'
            cm_p = paired('C', cm, 'C', rm, k, t) if cm in M['C'][k][t] else None
            if c is None:
                continue
            tot += 1
            pos += (c['d'] > 0); sig += (c['d'] > 0 and c['p'] < 0.05)
            if cm_p:
                line(f'  {a:10s} {c["a"]:8.3f} {c["b"]:9.3f} {c["d"]:+9.3f} {c["p"]:8.3g} | '
                     f'{cm_p["a"]:7.3f} {cm_p["b"]:7.3f} {cm_p["d"]:+7.3f} {cm_p["p"]:8.3g}')
            else:
                line(f'  {a:10s} {c["a"]:8.3f} {c["b"]:9.3f} {c["d"]:+9.3f} {c["p"]:8.3g} | (无 margin)')
        line(f'  → 正增益 {pos}/{tot} 格，其中显著 {sig} 格')

line()
line('=' * 100)
line('T3b · 加先验 vs 不加先验：C:algo|clm(z+码本) 减 B:algo(原始光谱)')
line('   ⚠ 输入不同（含 stage1 表示），是混合对比，不能当纯先验增量')
line('=' * 100)
for t in TASKS:
    for k in SHOTS:
        line(f'\n[{t}  k={k}]')
        for a in ALGOS:
            b = f'B:{a}'; c = f'C:{a}|clm'
            if b not in M['B'][k][t] or c not in M['C'][k][t]:
                continue
            q = paired('C', c, 'B', b, k, t)
            line(f'  {a:10s} C|clm={q["a"]:.3f}  B={q["b"]:.3f}  Δ={q["d"]:+.3f}  p={q["p"]:.3g}')

line()
line('=' * 100)
line('T4 · 平滑头动物园（ridge/pls/krr/gp × clm/rand），同 P bucket')
line('=' * 100)
for t in TASKS:
    for k in SHOTS:
        line(f'\n[{t}  k={k}]')
        for hn in ['ridge', 'pls', 'krr', 'gp']:
            zt = 'n' if t == 'same-matrix' else 'c'
            a = f'H:{hn}|clm|{zt}'; b = f'H:{hn}|rand384|{zt}'
            if a not in M['P'][k][t]:
                continue
            q = paired('P', a, 'P', b, k, t)
            line(f'  {hn:8s} clm={q["a"]:.3f} rand={q["b"]:.3f} Δ={q["d"]:+.3f} p={q["p"]:.3g}')

line()
line('=' * 100)
line('T5 · 检出：本文最强臂 vs 最强普通算法（same-matrix，全 k）')
line('=' * 100)
for k in SHOTS:
    arms = sorted(a for a in M['B'][k]['same-matrix'] if a.startswith('B:'))
    best = max(arms, key=lambda a: pm('B', k, 'same-matrix', a) or -1)
    ours = pm('O', k, 'same-matrix', 'ours-det')
    line(f'  k={k}  本文 ours-det={ours:.3f}  最强 {best[2:]}={pm("B", k, "same-matrix", best):.3f} '
         f' Δ={ours - pm("B", k, "same-matrix", best):+.3f}')

line('\n[exp81d] 完成；合并结果 → out/exp81_merged.json')
