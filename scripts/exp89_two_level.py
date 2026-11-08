# -*- coding: utf-8 -*-
"""
exp89 —— 「两级结构」验证：先验的作用是「从无到有」，不是「从好到更好」

不跑新实验，只从 exp82_dis / exp82_nhu4 / exp82_nhu5 逐格重算。

三种统计量：
  1. 第一级：结构类均值 vs 非结构对照（rand384 / onehot）—— 期望极大效应
  2. 第二级：结构类内部两两配对 —— 期望 0 显著（平台）
  3. 边界：physchem 何时脱出平台（|U|=5）

两种汇总口径并列（不挑有利的）：
  - pair 等权（主）：先按候选对平均跨 seed，再对候选对平均  → n = 候选对数
  - record 等权（稳健）：每个 (tag, 候选对) 一条              → n = 36/120/200

输出：out/exp89_two_level.json、out/exp89_two_level.md
"""
import json
import os
import itertools
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")

FILES = [("exp82_dis.json", 3), ("exp82_nhu4.json", 4), ("exp82_nhu5.json", 5)]
LAM = 1e-4

# 分组：结构/序列类 (S) / 组成描述符 (P) / 非结构对照 (N)
GROUP = {
    "clm": "S", "ecfp384": "S", "ecfp2048": "S", "maccs": "S",
    "avalon": "S", "charbag": "S",
    "physchem": "P", "rdkit": "P",
    "onehot": "N", "rand384": "N",
}
LABEL = {
    "clm": "ChemBERTa (384d)", "ecfp384": "ECFP-384", "ecfp2048": "ECFP-2048",
    "maccs": "MACCS-167", "avalon": "Avalon-1024", "charbag": "SMILES char-bag (17d)",
    "physchem": "RDKit physchem (25d)", "rdkit": "RDKit all desc (200+d)",
    "onehot": "one-hot identity", "rand384": "random (384d)",
}


def load(fn):
    with open(os.path.join(OUT, fn), encoding="utf-8") as f:
        return json.load(f)


def get_arm(d, prior, field):
    arm = d["A"].get(f"A:{prior}|lam{LAM:g}")
    if arm is None:
        return None
    return arm[field]


KEYMAP = {"dis": "auc_dis", "top1": "top1"}


def pairwise_means(d, prior, field):
    """候选对等权：每个候选对（top-1 时为每个候选）先跨 seed/tag 平均。"""
    a = get_arm(d, prior, field)
    if a is None:
        return None
    rk = KEYMAP[field]
    out = {}
    for key, recs in a.items():
        vals = [r.get(rk) for r in recs if r.get(rk) is not None]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def record_values(d, prior, field):
    """记录等权：每个 (tag, key) 一条。"""
    a = get_arm(d, prior, field)
    if a is None:
        return None
    rk = KEYMAP[field]
    out = {}
    for key, recs in a.items():
        for r in recs:
            v = r.get(rk)
            if v is None:
                continue
            out[(key, r["tag"])] = float(v)
    return out


def align(prior_dicts, priors):
    sets = [set(prior_dicts[p]) for p in priors]
    keys = sorted(set.intersection(*sets)) if len(sets) > 1 else sorted(sets[0])
    return keys, {p: np.array([prior_dicts[p][k] for k in keys], float) for p in priors}


def analyse(d, nhu, field):
    priors = [p for p in d["META"]["priors"] if get_arm(d, p, field) is not None]
    res = {}

    # ---- pair 等权（主） ----
    pm = {p: pairwise_means(d, p, field) for p in priors}
    pm = {p: v for p, v in pm.items() if v}
    keys, V = align(pm, list(pm))
    res["pair"] = _stats(V, keys, field)

    # ---- record 等权（稳健） ----
    rv = {p: record_values(d, p, field) for p in priors}
    rv = {p: v for p, v in rv.items() if v}
    keys2, V2 = align(rv, list(rv))
    res["record"] = _stats(V2, keys2, field)

    res["priors"] = list(V)
    res["nhu"] = nhu
    return res, V, keys


def _stats(V, keys, field):
    priors = list(V)
    S = [p for p in priors if GROUP.get(p) == "S"]
    sm = np.mean([V[p] for p in S], axis=0)
    out = {"n_unit": len(keys), "priors": priors,
           "mean": {p: float(V[p].mean()) for p in priors},
           "sd": {p: float(V[p].std(ddof=1)) for p in priors},
           "n_struct": len(S),
           "span_struct": float(max(V[p].mean() for p in S) - min(V[p].mean() for p in S))}

    # 第二级：结构类内部
    pairs = []
    for a, b in itertools.combinations(S, 2):
        t, p_ = stats.ttest_rel(V[a], V[b])
        pairs.append({"a": a, "b": b, "d": float((V[a] - V[b]).mean()), "p": float(p_)})
    out["lvl2"] = {"n_pairs": len(pairs), "n_sig": int(sum(1 for r in pairs if r["p"] < 0.05)),
                   "min_p": float(min(r["p"] for r in pairs)),
                   "max_abs_d": float(max(abs(r["d"]) for r in pairs)),
                   "pairs": sorted(pairs, key=lambda r: r["p"])}

    # 第一级：结构类 vs 每个对照
    out["lvl1"] = {}
    for q in ["onehot", "rand384", "physchem", "rdkit"]:
        if q not in V:
            continue
        dd = sm - V[q]
        t, p_ = stats.ttest_rel(sm, V[q])
        out["lvl1"][q] = {"d": float(dd.mean()), "p": float(p_),
                          "dz": float(dd.mean() / (dd.std(ddof=1) + 1e-12))}
    return out


def fmt(d, nhu, field, tag):
    res = d["pair"]
    pri = res["priors"]
    lines = []
    lines.append(f"### |U|={nhu} · {tag} · pair 等权（n={res['n_unit']}）\n")
    lines.append("| 组 | 先验 | 均值 | sd | 结构类−此先验 Δ (p) | dz |")
    lines.append("|---|---|---:|---:|---|---:|")
    order = [p for p in pri if GROUP.get(p) == "S"] + \
            [p for p in pri if GROUP.get(p) == "P"] + \
            [p for p in pri if GROUP.get(p) == "N"]
    for p in order:
        g = GROUP.get(p, "?")
        if p in res["lvl1"]:
            l = res["lvl1"][p]
            delta = f"{l['d']:+.4f} ({l['p']:.2e})"
            dz = f"{l['dz']:+.2f}"
        else:
            delta, dz = "—", "—"
        lines.append(f"| {g} | {LABEL.get(p,p)} | {res['mean'][p]:.4f} | {res['sd'][p]:.4f} | {delta} | {dz} |")
    l2 = res["lvl2"]
    lines.append("")
    lines.append(f"**第二级**：结构类 {res['n_struct']} 个，{l2['n_pairs']} 对两两比较，"
                 f"**显著 {l2['n_sig']} 对**，最小 p={l2['min_p']:.3f}，最大 |Δ|={l2['max_abs_d']:.4f}，"
                 f"跨度 {res['span_struct']:.4f}")
    # record 口径对照
    r = d["record"]
    lines.append(f"**稳健（record 等权 n={r['n_unit']}）**：结构类跨度 {r['span_struct']:.4f}，"
                 f"显著 {r['lvl2']['n_sig']}/{r['lvl2']['n_pairs']}，"
                 f"vs rand Δ={r['lvl1']['rand384']['d']:+.4f} (dz={r['lvl1']['rand384']['dz']:+.2f})，"
                 f"vs physchem Δ={r['lvl1']['physchem']['d']:+.4f} (p={r['lvl1']['physchem']['p']:.3f})")
    lines.append("")
    return "\n".join(lines)


def main():
    out = {}
    md = ["# exp89 · 两级结构验证（不跑新实验，仅重算 exp82）", "",
          "> 主张：零样本归因的性能由**码本是否携带化学结构**决定（第一级，断崖），",
          "> 而在结构类内部**编码家族不可辨**（第二级，平台）。",
          "> 所有数字从 exp82 系列 json 逐格重算；pair 等权为主口径，record 等权并列。", ""]

    for fn, nhu in FILES:
        d = load(fn)
        entry = {}
        for field, tag in [("dis", "auc_dis"), ("top1", "top-1")]:
            try:
                res, V, keys = analyse(d, nhu, field)
            except Exception as e:
                md.append(f"> |U|={nhu} {tag} 跳过：{e}\n")
                continue
            entry[field] = res
            md.append(fmt(res, nhu, field, tag))
        out[nhu] = entry

    with open(os.path.join(OUT, "exp89_two_level.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "exp89_two_level.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
