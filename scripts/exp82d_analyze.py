"""exp82d: 汇总定向归因三轴结果 → md。
配对单元 = (seed/组合 tag, 候选对)。随机码本先按 tag 平均再配对。
"""
import json, os, numpy as np
from scipy.stats import ttest_rel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
d = json.load(open(os.path.join(OUT, 'exp82_dis.json'), encoding='utf-8'))
PRIORS = d['META']['priors']
LAM0 = d['META']['lams'][0]
LAMS = d['META']['lams']
CHEM = ["ecfp384", "ecfp2048", "maccs", "avalon", "rdkit", "physchem", "charbag"]
NULL = ["onehot", "rand384"]


def per_tag(bucket, arm, kind='dis', field='auc_dis'):
    """{(tag, key): 该 tag 内该格的平均}"""
    out = {}
    v = d.get(bucket, {}).get(arm, {}).get(kind)
    if not v:
        return out
    for key, lst in v.items():
        tmp = {}
        for q in lst:
            tmp.setdefault(q['tag'], []).append(q[field])
        for t, vs in tmp.items():
            out[(t, key)] = float(np.mean(vs))
    return out


def pt(bucket, a1, a2, kind='dis', field='auc_dis'):
    A, B = per_tag(bucket, a1, kind, field), per_tag(bucket, a2, kind, field)
    ks = sorted(set(A) & set(B))
    if len(ks) < 2:
        return None
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    return dict(a=x.mean(), b=y.mean(), d=(x - y).mean(),
                p=ttest_rel(x, y).pvalue, n=len(ks))


def flat(bucket, arm, kind='dis', field='auc_dis'):
    v = d.get(bucket, {}).get(arm, {}).get(kind, {})
    xs = [q[field] for lst in v.values() for q in lst]
    return float(np.mean(xs)) if xs else None


L = []
w = L.append
w('# exp82 · 定向归因口径三轴基准（T、D 均未见）\n')
w('> `auc_dis > 0.5` = 指名正确；`top-1` = 在 |U|=3 候选中选对。')
w('> 配对单元 = (seed/组合, 候选对)，n 见各行。轴 B 为「伪定向」（打分不依赖候选集），')
w('> 只报可分性 `sep=|auc−0.5|×2`，top-1 结构性 **undefined**。\n')

# ---------------- 轴 A ----------------
w('## 轴 A · 先验动物园（同一岭回归头，只换码本）\n')
w(f'主口径 λ={LAM0:g}（exp53/58d 最优）；另给每个先验在 λ 网格上的最优值（上界诊断）。\n')
rows = []
for p in PRIORS:
    arm = f'A:{p}|lam{LAM0:g}' if p != 'rand384' else 'A:rand384'
    if p == 'rand384':
        arm = f'A:rand384|lam{LAM0:g}'
        if arm not in d['A']:
            arm = 'A:rand384'
    a = flat('A', arm)
    t = flat('A', arm, 'top1', 'top1')
    # λ 最优
    best = None
    for lam in LAMS:
        aa = flat('A', f'A:{p}|lam{lam:g}')
        if aa is not None and (best is None or aa > best[1]):
            best = (lam, aa)
    if p == 'rand384':
        best = (LAM0, a)
    rows.append((p, a, t, best[1] if best else None, best[0] if best else None))
rows.sort(key=lambda r: -(r[1] or 0))
w('| 排名 | 先验 | auc_dis (λ主口径) | top-1 | λ最优 auc | λ* |')
w('|---:|---|---:|---:|---:|---:|')
for i, (p, a, t, ba, bl) in enumerate(rows, 1):
    tag = ' **←本文**' if p == 'clm' else ''
    w(f'| {i} | `{p}`{tag} | {a:.3f} | {t:.3f} | {ba:.3f} | {bl:g} |')

c = pt('A', f'A:clm|lam{LAM0:g}', 'A:rand384')
w(f'\n- Δ(**clm − rand**)：{c["d"]:+.3f}（p={c["p"]:.3g}, n={c["n"]}）')
for p in CHEM:
    q = pt('A', f'A:clm|lam{LAM0:g}', f'A:{p}|lam{LAM0:g}')
    if q:
        w(f'- Δ(clm − {p})：{q["d"]:+.3f}（p={q["p"]:.3g}）')
q = pt('A', f'A:clm|lam{LAM0:g}', f'A:onehot|lam{LAM0:g}')
if q:
    w(f'- Δ(clm − onehot)：{q["d"]:+.3f}（p={q["p"]:.3g}）')
rank_clm = [r[0] for r in rows].index('clm') + 1
w(f'\n**CLM 排名 {rank_clm}/{len(rows)}**（λ 主口径）。\n')

w('### top-1（三选一，真实决策指标）配对检验\n')
tops = sorted([(flat('A', f'A:{p}|lam{LAM0:g}' if p != 'rand384' else 'A:rand384',
                     'top1', 'top1'), p) for p in PRIORS], reverse=True)
w('| 排名 | 先验 | top-1 |')
w('|---:|---|---:|')
for i, (v, p) in enumerate(tops, 1):
    w(f'| {i} | `{p}`{" **←本文**" if p=="clm" else ""} | {v:.3f} |')
for p in ['charbag', 'ecfp2048', 'ecfp384', 'physchem', 'maccs', 'onehot', 'rand384']:
    a2 = f'A:{p}|lam{LAM0:g}' if p != 'rand384' else 'A:rand384'
    q = pt('A', f'A:clm|lam{LAM0:g}', a2, 'top1', 'top1')
    if q:
        w(f'- Δ(clm − {p}) top-1：{q["d"]:+.3f}（p={q["p"]:.3g}）')
w('')

# ---------------- 轴 C ----------------
w('## 轴 C · 把先验装到普通算法上（算法学 S→Z，再外推候选）\n')
H = d['META'].get('cond_algos')  # 兼容
w('| 算法 | clm | rand | Δ(clm−rand) | p | 最强普通先验 | Δ(clm−best) | p |')
w('|---|---:|---:|---:|---:|---|---:|---:|')
for hn in ["ridge", "pls", "krr", "knn1", "mlp"]:
    base = f'C:{hn}|clm'
    if base not in d['C']:
        continue
    c = pt('C', f'C:{hn}|clm', f'C:{hn}|rand384')
    # 最强普通先验（排除 onehot）
    cand = []
    for p in CHEM:
        v = flat('C', f'C:{hn}|{p}')
        if v is not None:
            cand.append((v, p))
    if not cand:
        continue
    bv, bp = max(cand)
    q = pt('C', f'C:{hn}|clm', f'C:{hn}|{bp}')
    w(f'| {hn} | {c["a"]:.3f} | {c["b"]:.3f} | {c["d"]:+.3f} | {c["p"]:.3g} | '
      f'`{bp}` {bv:.3f} | {q["d"]:+.3f} | {q["p"]:.3g} |')
pos = neg = 0
for hn in ["ridge", "pls", "krr", "knn1", "mlp"]:
    q = pt('C', f'C:{hn}|clm', f'C:{hn}|rand384')
    if q:
        pos += q['d'] > 0; neg += q['d'] <= 0
w(f'\n- 相对随机码本：**{pos} 正 / {neg} 负**（共 {pos+neg} 个算法）\n')

# ---------------- 轴 B ----------------
w('## 轴 B · 普通 ML/DL 算法（原始光谱，k=0 无 U 标样）\n')
w('⚠ 这些算法在 k=0 的打分 `f(x)=1−maxprob(x)`（或异常分）**不依赖候选分子集**，')
w('因此只能报可分性 `sep`，无法执行「在候选集中指名」（top-1 = undefined）。\n')
rb = []
for arm, v in d['B'].items():
    xs = [q['auc_dis'] for lst in v.get('dis', {}).values() for q in lst]
    if xs:
        sep = float(np.mean([abs(x - .5) * 2 for x in xs]))
        rb.append((arm.replace('B:', ''), sep, len(xs)))
rb.sort(key=lambda r: -r[1])
w('| 算法 | sep (可分性) | n |')
w('|---|---:|---:|')
for nm, s, n in rb:
    w(f'| {nm} | {s:.3f} | {n} |')
w('')

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, 'exp82_p6_dis_report.md')
open(p, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print(f'\n[exp82d] → {p}')
