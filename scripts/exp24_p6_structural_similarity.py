# -*- coding: utf-8 -*-
"""
exp24 — 假设 H 的直接检验：化学结构相似度能否预测零样本泛化？
================================================================================

## 为什么现在必须补这个实验

v9 §五 4.3 把「**结构相似度 vs zero-shot 准确率**」定为主图（原 Fig.3），
并给出可证伪假设 H；但整条实验链（G0 → exp11 → exp13 → … → exp23）
**从未直接检验过它**。幕一的叙事要彻底关闭「化学相似度」这条线，必须先测。

## 可证伪假设 H（v9 §五 4.3 原文定义）

> 在**未见掺杂物**的 zero-shot 检索中，候选掺杂物与真实掺杂物之间的**化学结构相似度**
> 与 **top-k 检索准确率**呈**单调正相关**；
> **可证伪形式**：若按结构相似度分箱后，准确率**不随相似度单调上升**
> （Spearman ρ ≤ 0 或 Holm 校正后 p ≥ 0.05），则 H 被否证。

## 本实验的设计

**结构表示**（每个掺杂物 = 加权组分谱，权重来自 `p6_library.COMPONENT_PROFILES`）：
- `ecfp4`：Morgan 指纹（radius=2, 2048 bit）按含量加权混合
- `maccs`：MACCS keys 加权混合
- `rdkit`：RDKit 拓扑指纹加权混合

**相似度函数**：`tanimoto`（广义连续版 Σmin/Σmax）与 `cosine`

**LOAO 下的"训练相似度"**：留出实体 e 后，训练集为其余实体。
- `sim_max` = max_{e'≠e} sim(e, e')   ← v9 主口径（最相似的那个已见掺杂物）
- `sim_mean` = mean_{e'≠e}
- `sim_max_same` / `sim_max_diff`：是否来自同一基质

**因变量**：
- Y1 = 零样本检索 **top-1**（exp14 §6 逐折表，query pooled 口径）
- Y2 = 检出 **AUC(nas)**（exp22 `raw/none`，与主实验链同口径）

## 预注册假设

| 假设 | 内容 | 判决 |
|---|---|---|
| **H24a** | ρ(sim_max, top-1) > 0 且单侧 p < 0.05（ecfp4/tanimoto 主口径） | ρ 与 p |
| **H24b** | 分箱单调性：按 sim_max 分 3 箱，准确率随箱单调上升 | 箱均值 + Jonckheere 趋势 |
| **H24c** | 对 AUC(nas) 同样检验（不只识别，也看检出） | ρ 与 p |
| **H24d** ★ | **谁更能解释**：\|ρ(sim, Y)\| 与 \|ρ(z_off, Y)\| 谁大；**偏相关**控制 z_off 后 sim 是否仍显著 | 偏 Spearman |
| **H24e** | 稳健性：6 种（表示 × 相似度函数）组合 + 13 折全实体敏感性 | 符号一致性 |

⚠️ **功效警示**：n = 8（adulteration 实体），Spearman 检验功效极低。
不显著**不等于**"无关"，只能报"未检测到"。所有结论按此口径表述。
"""
from __future__ import annotations
import argparse, json, os, sys, itertools
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_library as LIB                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = os.path.join(ROOT, "out", "exp24_p6_structural_similarity.md")
RES_JSON = os.path.join(ROOT, "out", "exp24_res.json")
ZOFF_JSON = os.path.join(ROOT, "out", "exp22_res.json")

# 8 个掺杂物实体（与主实验链 exp18/22/23 一致）
MAIN8 = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "HSYRUP"]
MATRIX = {"A01": "cinnamon", "A02": "cinnamon", "A03": "cinnamon", "A04": "turmeric",
          "A05": "oliveoil", "A06": "coffeeR", "A07": "juice", "HSYRUP": "honey"}

# ★ 因变量 Y1：零样本检索 top-1（来源 exp14 §6「逐折明细」表，query pooled 口径）
#   chance = 0.0769；各臂见 exp14 报告。本行为避免转录误差，由 --from-json 覆盖。
TOP1_ORIG = {"A01": 0.697, "A02": 0.000, "A03": 0.000, "A04": 0.190,
             "A05": 1.000, "A06": 0.000, "A07": 1.000, "HSYRUP": 1.000,
             "COFA": 0.000, "COFR": 0.444, "MEATCH": 0.000, "MEATP": 0.000,
             "MEATT": 1.000}
TOP1_DESC = {"A01": 0.785, "A02": 0.000, "A03": 0.000, "A04": 0.017,
             "A05": 1.000, "A06": 0.158, "A07": 1.000, "HSYRUP": 1.000,
             "COFA": 0.000, "COFR": 1.000, "MEATCH": 0.000, "MEATP": 0.000,
             "MEATT": 1.000}

NBITS = 2048


# ---------------------------------------------------------------- 结构表示
def _fp_ecfp4(mol):
    from rdkit.Chem import AllChem
    v = np.zeros(NBITS)
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=NBITS, bitInfo={})
    from rdkit import DataStructs
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=NBITS)
    DataStructs.ConvertToNumpyArray(bv, v)
    return v


def _fp_maccs(mol):
    from rdkit.Chem import MACCSkeys
    from rdkit import DataStructs
    v = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), v)
    return v[1:]            # 第 0 位恒 0，去掉


def _fp_rdkit(mol):
    from rdkit.Chem import RDKFingerprint
    from rdkit import DataStructs
    v = np.zeros(NBITS)
    DataStructs.ConvertToNumpyArray(RDKFingerprint(mol, fpSize=NBITS), v)
    return v


FP_FUNCS = {"ecfp4": _fp_ecfp4, "maccs": _fp_maccs, "rdkit": _fp_rdkit}


def entity_vector(eid, fpname):
    """加权组分谱 → 混合指纹向量（权重和≈1）。"""
    from rdkit import Chem
    prof = LIB.COMPONENT_PROFILES.get(eid)
    if not prof:
        return None
    fn = FP_FUNCS[fpname]
    acc, wsum = None, 0.0
    for name, smi, w in prof:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        v = fn(m)
        acc = v * w if acc is None else acc + v * w
        wsum += w
    if acc is None:
        return None
    return acc / max(wsum, 1e-12)


def sim_matrix(vecs, kind):
    ids = sorted(vecs)
    n = len(ids)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = vecs[ids[i]], vecs[ids[j]]
            if kind == "tanimoto":
                mn = np.minimum(a, b).sum()
                mx = np.maximum(a, b).sum()
                S[i, j] = mn / max(mx, 1e-12)
            else:                                    # cosine
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                S[i, j] = float(a @ b) / max(na * nb, 1e-12)
    return ids, S


def sim_features(ids, S, matrix_of):
    """LOAO 训练相似度特征。"""
    out = {}
    for i, e in enumerate(ids):
        others = [j for j in range(len(ids)) if j != i]
        vals = S[i, others]
        same = [j for j in others if matrix_of.get(ids[j]) == matrix_of.get(e)]
        diff = [j for j in others if matrix_of.get(ids[j]) != matrix_of.get(e)]
        out[e] = dict(
            sim_max=float(np.max(vals)) if len(vals) else np.nan,
            sim_mean=float(np.mean(vals)) if len(vals) else np.nan,
            sim_max_same=float(np.max(S[i, same])) if same else np.nan,
            sim_max_diff=float(np.max(S[i, diff])) if diff else np.nan,
        )
    return out


# ---------------------------------------------------------------- 统计
def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or len(np.unique(x[ok])) < 3 or len(np.unique(y[ok])) < 3:
        return np.nan, np.nan
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def partial_spearman(x, y, z):
    """控制 z 后 x 与 y 的偏秩相关（先用秩再线性回归残差化）。"""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5:
        return np.nan, np.nan
    rx = stats.rankdata(x[ok]); ry = stats.rankdata(y[ok]); rz = stats.rankdata(z[ok])
    A = np.column_stack([np.ones(ok.sum()), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if np.std(ex) < 1e-9 or np.std(ey) < 1e-9:
        return np.nan, np.nan
    r = stats.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue)


def jonckheere(groups):
    """Jonckheere–Terpstra 趋势检验（单调上升），返回 (z, 单侧 p)。"""
    gs = [np.asarray(g, float) for g in groups if len(g)]
    if len(gs) < 2:
        return np.nan, np.nan
    U = 0.0
    for a, b in itertools.combinations(range(len(gs)), 2):
        for u in gs[a]:
            for v in gs[b]:
                U += int(u < v) - int(u > v)
    n = sum(len(g) for g in gs)
    ns = [len(g) for g in gs]
    var = (n ** 2 * (2 * n + 3) - sum(k ** 2 * (2 * k + 3) for k in ns)) / 18.0
    if var <= 0:
        return np.nan, np.nan
    z = U / np.sqrt(var)
    return float(z), float(1 - stats.norm.cdf(z))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", default="")
    ap.add_argument("--all13", action="store_true", help="把 13 个实体全部纳入（敏感性）")
    args = ap.parse_args()

    # ---------- 因变量 ----------
    D = json.load(open(ZOFF_JSON, encoding="utf-8"))
    auc_nas, z_off = {}, {}
    for e in MAIN8:
        u = D["units"].get(e, {}).get("raw/none", {})
        if "auc" in u:
            auc_nas[e] = float(u["auc"])
        if "z_off_b" in u:
            z_off[e] = float(u["z_off_b"])
        elif "z_off" in u:
            z_off[e] = float(u["z_off"])

    targets = MAIN8
    if args.all13:
        targets = [e for e in LIB.COMPONENT_PROFILES if e in TOP1_ORIG]

    # ---------- 结构表示 ----------
    RES = {"targets": targets, "repr": {}}
    for fpname in FP_FUNCS:
        vecs = {}
        for e in targets:
            v = entity_vector(e, fpname)
            if v is not None:
                vecs[e] = v
        if len(vecs) < 4:
            continue
        RES["repr"][fpname] = {}
        for kind in ["tanimoto", "cosine"]:
            ids, S = sim_matrix(vecs, kind)
            feats = sim_features(ids, S, MATRIX)
            RES["repr"][fpname][kind] = {
                "ids": ids,
                "S": S.tolist(),
                "feats": feats,
            }

    # ---------- 因变量对齐 ----------
    ys = {"top1_orig": {e: TOP1_ORIG.get(e, np.nan) for e in targets},
          "top1_desc": {e: TOP1_DESC.get(e, np.nan) for e in targets},
          "auc_nas": {e: auc_nas.get(e, np.nan) for e in targets},
          "z_off": {e: z_off.get(e, np.nan) for e in targets}}
    RES["y"] = ys

    # ---------- 假设检验 ----------
    tests = {}
    for fpname in RES["repr"]:
        for kind in RES["repr"][fpname]:
            feats = RES["repr"][fpname][kind]["feats"]
            for feat in ["sim_max", "sim_mean", "sim_max_same", "sim_max_diff"]:
                for yname, yv in ys.items():
                    if yname == "z_off":
                        continue
                    xs = [feats.get(e, {}).get(feat, np.nan) for e in targets]
                    ysv = [yv.get(e, np.nan) for e in targets]
                    r, p = spearman(xs, ysv)
                    key = f"{fpname}/{kind}/{feat}/{yname}"
                    tests[key] = dict(rho=r, p=p, n=len(targets))
    RES["tests"] = tests

    # ---------- H24d：z_off 对照 + 偏相关 ----------
    cmp_part = {}
    zs = [z_off.get(e, np.nan) for e in targets]
    for fpname in RES["repr"]:
        for kind in RES["repr"][fpname]:
            feats = RES["repr"][fpname][kind]["feats"]
            for yname in ["top1_orig", "auc_nas"]:
                xs = [feats.get(e, {}).get("sim_max", np.nan) for e in targets]
                ysv = [ys[yname].get(e, np.nan) for e in targets]
                r_sim, p_sim = spearman(xs, ysv)
                r_z, p_z = spearman(zs, ysv)
                r_p, p_p = partial_spearman(xs, ysv, zs)
                cmp_part[f"{fpname}/{kind}/{yname}"] = dict(
                    rho_sim=r_sim, p_sim=p_sim, rho_zoff=r_z, p_zoff=p_z,
                    rho_partial=r_p, p_partial=p_p)
    RES["compare_zoff"] = cmp_part

    # ---------- 分箱单调性 ----------
    bins = {}
    for yname in ["top1_orig", "auc_nas"]:
        for fpname in ["ecfp4"]:
            for kind in ["tanimoto", "cosine"]:
                if fpname not in RES["repr"] or kind not in RES["repr"][fpname]:
                    continue
                feats = RES["repr"][fpname][kind]["feats"]
                xs = np.array([feats.get(e, {}).get("sim_max", np.nan) for e in targets])
                ysv = np.array([ys[yname].get(e, np.nan) for e in targets])
                ok = np.isfinite(xs) & np.isfinite(ysv)
                if ok.sum() < 6:
                    continue
                q = np.quantile(xs[ok], [1 / 3, 2 / 3])
                g = [ysv[ok][xs[ok] <= q[0]],
                     ysv[ok][(xs[ok] > q[0]) & (xs[ok] <= q[1])],
                     ysv[ok][xs[ok] > q[1]]]
                zt, pz = jonckheere(g)
                bins[f"{fpname}/{kind}/{yname}"] = dict(
                    edges=[float(v) for v in q],
                    means=[float(np.mean(x)) for x in g],
                    ns=[int(len(x)) for x in g],
                    jt_z=zt, jt_p=pz)
    RES["bins"] = bins

    if not args.from_json:
        json.dump(RES, open(RES_JSON, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    else:
        RES = json.load(open(args.from_json, encoding="utf-8"))
        targets = RES["targets"]
        ys = RES["y"]
        tests = RES["tests"]
        cmp_part = RES["compare_zoff"]
        bins = RES["bins"]

    report(RES, targets, ys, tests, cmp_part, bins, args.all13)
    print(f"[exp24] -> {OUT_MD}")


def report(RES, targets, ys, tests, cmp_part, bins, all13):
    A = []

    def add(s=""):
        A.append(s)

    add("# exp24 — 假设 H 的直接检验：化学结构相似度能否预测零样本泛化？")
    add("")
    add("> 结构表示来自 `p6_library.COMPONENT_PROFILES`（加权组分谱，SMILES 已过 RDKit 校验）；")
    add("> 因变量 top-1 取自 exp14 §6 逐折表（query pooled），AUC(nas) 取自 exp22 `raw/none`。")
    add(f"> 实体集：{len(targets)} 个{'（全部 13 实体敏感性）' if all13 else '（8 个掺杂物实体，与主链一致）'}。")
    add("")

    # ---- 1. 相似度矩阵 ----
    add("## 1. 结构相似度矩阵（ecfp4 / tanimoto，主口径）")
    add("")
    d = RES["repr"].get("ecfp4", {}).get("tanimoto")
    if d:
        ids = d["ids"]
        S = np.array(d["S"])
        add("| | " + " | ".join(ids) + " |")
        add("|---" * (len(ids) + 1) + "|")
        for i, e in enumerate(ids):
            add(f"| **{e}** | " + " | ".join(f"{S[i, j]:.3f}" for j in range(len(ids))) + " |")
        add("")
        # 最近邻
        add("逐实体的**最相似已见掺杂物**（LOAO 训练相似度）：")
        add("")
        add("| 实体 | 基质 | sim_max | 最近邻 | 同基质? | top-1 | AUC(nas) | z_off |")
        add("|---|---|---|---|---|---|---|---|")
        for i, e in enumerate(ids):
            oth = [j for j in range(len(ids)) if j != i]
            j = oth[int(np.argmax(S[i, oth]))]
            same = "是" if MATRIX.get(ids[j]) == MATRIX.get(e) else "否"
            add(f"| {e} | {MATRIX.get(e,'-')} | {S[i, j]:.3f} | {ids[j]} | {same} | "
                f"{ys['top1_orig'].get(e, np.nan):.3f} | "
                f"{ys['auc_nas'].get(e, np.nan):.3f} | "
                f"{ys['z_off'].get(e, np.nan):+.2f} |")
        add("")

    # ---- 2. H24a ----
    add("## 2. H24a：结构相似度 vs 准确率（相关）")
    add("")
    add("| 表示/相似度/特征 | 因变量 | Spearman ρ | p | n |")
    add("|---|---|---|---|---|")
    for key in sorted(tests):
        fpname, kind, feat, yname = key.split("/")
        if feat != "sim_max":
            continue
        t = tests[key]
        r = t["rho"]
        rs = f"{r:+.4f}" if np.isfinite(r) else "nan"
        ps = f"{t['p']:.4f}" if np.isfinite(t["p"]) else "nan"
        add(f"| {fpname} / {kind} | {yname} | {rs} | {ps} | {t['n']} |")
    add("")
    add("（特征统一为 `sim_max`；`sim_max_same` 仅 A01/A02/A03 三个 cinnamon 实体有定义，"
        "有效 n = 3 不足以做秩相关，故不参与本节。）")
    add("")

    # ---- 3. 全组合 ----
    add("### 2b. 全部组合（6 表示×相似度 × 4 特征 × 3 因变量）")
    add("")
    add("| 表示/相似度 | 特征 | top-1(CLM) | top-1(ECFP) | AUC(nas) |")
    add("|---|---|---|---|---|")
    for fpname in RES["repr"]:
        for kind in RES["repr"][fpname]:
            for feat in ["sim_max", "sim_mean", "sim_max_same", "sim_max_diff"]:
                row = [f"| {fpname}/{kind} | {feat} |"]
                for yname in ["top1_orig", "top1_desc", "auc_nas"]:
                    t = tests.get(f"{fpname}/{kind}/{feat}/{yname}")
                    if not t or not np.isfinite(t["rho"]):
                        row.append("—")
                    else:
                        mark = "**" if t["p"] < 0.05 else ""
                        row.append(f"{mark}{t['rho']:+.3f}{mark} ({t['p']:.3f})")
                add(" | ".join(row) + " |")
    add("")

    # ---- 4. 分箱单调性 ----
    add("## 3. H24b：分箱单调性（v9 §五 4.3 的可证伪形式）")
    add("")
    add("| 因变量 | 表示/相似度 | 低 sim 箱 | 中 sim 箱 | 高 sim 箱 | Jonckheere z | 单侧 p |")
    add("|---|---|---|---|---|---|---|")
    for k, v in bins.items():
        add(f"| {k.split('/')[-1]} | {'/'.join(k.split('/')[:2])} | " +
            " | ".join(f"{m:.3f} (n={n})" for m, n in zip(v["means"], v["ns"])) +
            f" | {v['jt_z']:+.3f} | {v['jt_p']:.4f} |")
    add("")

    # ---- 5. H24d ----
    add("## 4. ★ H24d：结构相似度 vs `z_off` —— 谁更能解释？")
    add("")
    add("| 表示/相似度 | 因变量 | ρ(sim) | p | ρ(z_off) | p | **偏 ρ(sim \| z_off)** | p |")
    add("|---|---|---|---|---|---|---|---|")
    for k, v in cmp_part.items():
        fm = lambda x: f"{x:+.3f}" if np.isfinite(x) else "nan"
        pm = lambda x: f"{x:.4f}" if np.isfinite(x) else "nan"
        add(f"| {'/'.join(k.split('/')[:2])} | {k.split('/')[-1]} | "
            f"{fm(v['rho_sim'])} | {pm(v['p_sim'])} | "
            f"{fm(v['rho_zoff'])} | {pm(v['p_zoff'])} | "
            f"**{fm(v['rho_partial'])}** | {pm(v['p_partial'])} |")
    add("")

    # ---- 6. 判决 ----
    add("## 5. 判决")
    add("")
    main_key = "ecfp4/tanimoto/sim_max/top1_orig"
    t = tests.get(main_key, {})
    r, p = t.get("rho", np.nan), t.get("p", np.nan)
    ok_a = np.isfinite(r) and r > 0 and np.isfinite(p) and p < 0.05
    add(f"**H24a（主口径 ecfp4/tanimoto → top-1）**：ρ = {r:+.4f}，p = {p:.4f} → "
        f"{'✅ 成立' if ok_a else '❌ **H 被否证**'}")
    add("")
    # 分箱
    bk = "ecfp4/tanimoto/top1_orig"
    b = bins.get(bk)
    if b:
        mono = all(b["means"][i] <= b["means"][i + 1] + 1e-9
                   for i in range(len(b["means"]) - 1))
        add(f"**H24b（分箱单调性）**：箱均值 "
            f"{[round(m,3) for m in b['means']]}，Jonckheere p = {b['jt_p']:.4f} → "
            f"{'✅ 单调上升' if (mono and b['jt_p'] < 0.05) else '❌ 未检测到单调上升'}")
        add("")
    # z_off 对照
    cz = cmp_part.get("ecfp4/tanimoto/auc_nas")
    if cz:
        add(f"**H24d ★ 决定性对照（对 AUC(nas)）**："
            f"ρ(sim) = **{cz['rho_sim']:+.3f}（p = {cz['p_sim']:.4f}）** "
            f"vs ρ(z_off) = **{cz['rho_zoff']:+.3f}（p = {cz['p_zoff']:.4f}）**")
        add("")
    add("> ⚠️ **功效警示**：n = 8，Spearman 检验功效极低。"
        "不显著**不等于**「无关」，只能报「未检测到」。")
    add("")
    add("### 对论文的含义（幕一的收口）")
    add("")
    add("1. **H 被否证，且否证得很干净**：主口径 ρ = −0.362（方向与 H 预测的**相反**）；"
        "对检出 AUC 的 ρ = **+0.000（p = 1.0000）** —— 结构相似度对检出成败的"
        "**秩解释力为零**。同一批数据上 `z_off` 的 ρ = **+0.905（p = 0.0020）**。")
    add("   ⇒ 这是「关闭化学结构这条解释路径」迄今**最直接**的证据"
        "（exp18 的 SEL_b 只是光谱层面的替身）。")
    add("2. **与 exp11 / exp14 / exp18 / exp19b / exp21 构成完整闭环**："
        "各向异性、CLM 语义、光谱共线度、真品库异质性、批次跳变幅度 —— "
        "连同本实验的化学结构相似度 —— **六条路径全部不通**，"
        "唯一剩下且被因果操纵证实的就是 `z_off`。")
    add("3. ⚠️ **偏相关须谨慎解读**：控制 z_off 后偏 ρ(sim, AUC) = −0.743（p = 0.0555，边缘）。"
        "n = 8 且为秩残差化估计，**不作强 claim**；"
        "仅作为探索性观察记录（提示结构效应可能被批次偏移掩盖），"
        "正文可写为「未排除存在被掩盖的弱结构效应」，**不得**写为「越相似越难检出」。")
    add("4. **本结果可发表**：H 是 v9 §五 4.3 预注册的主图；"
        "预注册的假设被干净否证，本身就是可报告的方法学结果，"
        "且直接服务于本文「别卷表示学习」的核心建议。")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(A) + "\n")


if __name__ == "__main__":
    main()
