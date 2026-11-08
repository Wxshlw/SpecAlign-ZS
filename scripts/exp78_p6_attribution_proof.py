"""exp78 —— 归因不可行性证明：RF/SVM 在「识别」任务上能否输出？

核心命题（回应审稿人第一质疑：做了半天还不如 RF/SVM）：
    本文在「检出」上与 RF/SVM 统计不可区分（exp77: Δ=−0.026, p=0.19）——这点不辩。
    但「识别/归因」是另一个任务：问的是「掺的是 U 中的哪一种」。

    ★ 结构性论点：RF/SVM 的 k=0 打分 f(x)=1−maxprob(x) 是 **x 的标量函数，
      不依赖候选分子集合**。因此它无法执行「在候选集中排序」这一操作——
      不是做得差，是**无输出**（undefined）。

本实验把这一点量化：
  Stage A  定向判别 auc_dis：T 谱 vs D 谱（T、D 均为未见化合物）
           - 本文 zs：s_T − s_D（化学外推双方向）
           - 本文 zs-m：margin 版
           - zs-rand：随机码本对照（exp73 隐患复检）
           - rf / svm：1−maxprob（预期 ≈0.5，即无法区分两种未知掺假物）
           - rf-oracle：★ 把 U 三类也放进训练集（作弊上界，证明差异在
             「要不要标样」而非「模型能力」）
  Stage B  候选集排序（真实 NTS 操作）：候选 = U（3 选 1）
           - 本文 top-1 / MRR
           - RF：无输出（结构性），退化基线 1/|U| 一并列出并标注
  Stage C  分数可分性 Cohen's d：RF 异常分对 T 谱与 D 谱的重合程度
  Stage D  接入成本：新增一个候选分子所需成本（本文 SMILES 外推 vs RF 重训）
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, macro_recall,
                                         EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached, pick_lam

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7


def mrr_of(rank_pos):
    """rank_pos: 每样本正确候选的排名（1-based），返回 MRR。"""
    r = np.asarray(rank_pos, float)
    return float(np.mean(1.0 / r)) if len(r) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--nhu", type=int, default=3, help="未见化合物集 U 的大小")
    ap.add_argument("--nrep", type=int, default=4, help="每个 seed 采几个不同的 U")
    ap.add_argument("--proto", default="group")
    ap.add_argument("--out", default="exp78_attrib.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    dz = T_all.shape[1]

    res = {"A": {}, "B": {}, "C": {}, "D": {}}
    combos = list(itertools.combinations(eids, args.nhu))
    print(f"[exp78] device={DEV} 实体={eids} |U|={args.nhu} "
          f"三元组={len(combos)} reps={args.nrep} seeds={args.seeds}",
          flush=True)

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed * 7919 + 13)
        cis = rng.permutation(len(combos))[:args.nrep]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            S_all = T_all[[ki[e] for e in others]
                          + [ki[f"M_{m}"] for m in mids]
                          + [ki[e] for e in U]]
            NEW0 = len(others) + len(mids)          # U 在 W 中的起始列
            iu = {u: NEW0 + j for j, u in enumerate(U)}

            # ---- 划分：U 成员零样本（k=0），其余按组划分 ----
            Xte_U, Xtr_U = {}, {}
            for u in U:
                Xe = np.asarray(ent[u], float)
                te_ix, pool_ix = split_by_group(Xe, GR[u]["gid"], rng)
                Xte_U[u] = Xe[te_ix]
                Xtr_U[u] = Xe[pool_ix]        # 仅供 rf-oracle 作弊使用
            parts, ys = [], []
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if args.proto == "group" and e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:
                    pp = rng.permutation(len(Xo))
                    cut = max(2, int(round(OLD_FRAC * len(Xo))))
                    a, b = pp[:cut], pp[cut:]
                parts.append(Xo[a]); ys.append(np.full(len(a), j))
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                cut = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[pp[:cut]])
                ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)

            # ---------- 外部基线（k=0，标签集不含 U）----------
            sc = StandardScaler().fit(Xtr)
            Xtr_s = sc.transform(Xtr)
            t0 = time.time()
            rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                        n_jobs=-1).fit(Xtr_s, ytr)
            t_rf_fit = time.time() - t0
            sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                     random_state=0).fit(Xtr_s, ytr)
            f_rf = {u: 1 - rf.predict_proba(sc.transform(Xte_U[u])).max(1)
                    for u in U}
            f_sv = {u: 1 - sv.predict_proba(sc.transform(Xte_U[u])).max(1)
                    for u in U}

            # ---------- rf-oracle：把 U 也当已知类（作弊上界）----------
            p2, y2 = list(parts), list(ys)
            for j, u in enumerate(U):
                Xu = Xtr_U[u]
                if len(Xu) < 2:
                    continue
                p2.append(Xu)
                y2.append(np.full(len(Xu), len(others) + len(mids) + j))
            Xtr2 = np.concatenate(p2).astype(np.float32)
            ytr2 = np.concatenate(y2).astype(np.int64)
            sc2 = StandardScaler().fit(Xtr2)
            rf2 = RandomForestClassifier(n_estimators=300, random_state=0,
                                         n_jobs=-1).fit(
                sc2.transform(Xtr2), ytr2)
            # oracle 的「U 内三选一」
            cls_u = [len(others) + len(mids) + j for j in range(len(U))]
            P2 = {u: rf2.predict_proba(sc2.transform(Xte_U[u])) for u in U}
            orc_top1 = {}
            for u in U:
                sub = P2[u][:, cls_u]                # [n, |U|]
                pred = np.array(U)[sub.argmax(1)]
                orc_top1[u] = float(np.mean(pred == u))
                res["B"].setdefault("rf-oracle", {}).setdefault(u, []).append(
                    dict(top1=orc_top1[u], n=len(Xte_U[u])))

            # ---------- 本文 ----------
            nrows = len(others) + len(mids) + len(U)
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps,
                            seed * 1000 + ci, S=S_all)
            Zfit = embed(net, Xtr)
            rows = sorted({int(y) for y in ytr})
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            S_o, mu_S = S_all[rows], S_all[rows].mean(0, keepdims=True)
            # λ 扫描：exp53 已证最优 λ 常在网格边缘，固定取值会压死化学外推
            Zte = {u: embed(net, Xte_U[u]) for u in U}
            S_cand = np.stack([S_all[iu[u]] for u in U])
            for lam in lams:
                Wx = (S_all - mu_S) @ head_ridge_R(S_o, Z_old, lam)
                cd = (S_cand - mu_S) @ head_ridge_R(S_o, Z_old, lam)
                cc = np.asarray(cd, float)
                ccn = cc / (np.linalg.norm(cc, axis=1, keepdims=True) + 1e-12)
                cosw = float((ccn @ ccn.T)[np.triu_indices(len(U), 1)].mean())
                for T, D in itertools.combinations(U, 2):
                    iT, iD = iu[T], iu[D]
                    dT = Zte[T] @ Wx[iT] - Zte[T] @ Wx[iD]
                    dD = Zte[D] @ Wx[iT] - Zte[D] @ Wx[iD]
                    a = float(auc_of(dT, dD))
                    res["A"].setdefault(f"zs-lam{lam:g}", {}).setdefault(
                        f"{T}|{D}", []).append(
                        dict(auc_dis=a, sep=abs(a - 0.5) * 2,
                             cos_cand=cosw,
                             dw=float(np.linalg.norm(Wx[iT] - Wx[iD]))))
            RND = 20                       # 随机码本必须多次独立抽样（σ≈0.2）
            for rr in range(RND):
                rng_r = np.random.default_rng(seed * 100003 + rr * 97 + 5)
                Wr = nrm(rng_r.normal(size=(len(S_all), dz)).astype(np.float32))
                cdr = Wr[[iu[u] for u in U]]
                for T, D in itertools.combinations(U, 2):
                    iT, iD = iu[T], iu[D]
                    dT = Zte[T] @ Wr[iT] - Zte[T] @ Wr[iD]
                    dD = Zte[D] @ Wr[iT] - Zte[D] @ Wr[iD]
                    a = float(auc_of(dT, dD))
                    res["A"].setdefault("zs-rand", {}).setdefault(
                        f"{T}|{D}", []).append(
                        dict(auc_dis=a, sep=abs(a - 0.5) * 2))
                for u in U:
                    Lr = Zte[u] @ cdr.T
                    pred = np.array(U)[Lr.argmax(1)]
                    res["B"].setdefault("zs-rand", {}).setdefault(
                        u, []).append(
                        dict(top1=float(np.mean(pred == u)), n=len(Zte[u])))
            rng_r = np.random.default_rng(seed * 977 + 5)
            Wr = nrm(rng_r.normal(size=(len(S_all), dz)).astype(np.float32))

            # 主口径用最小 λ（exp53/58d 已证其最优）
            lam_main = lams[0]
            W = (S_all - mu_S) @ head_ridge_R(S_o, Z_old, lam_main)
            Wcand = (S_cand - mu_S) @ head_ridge_R(S_o, Z_old, lam_main)
            Wcand_r = Wr[[iu[u] for u in U]]
            # 化学码本行间 cos 结构（诊断）
            Sn = S_cand / (np.linalg.norm(S_cand, axis=1, keepdims=True) + 1e-12)
            res["SCOS"] = res.get("SCOS", []) + [
                dict(cos=float((Sn @ Sn.T)[np.triu_indices(len(U), 1)].mean()),
                     U="|".join(U))]

            for T, D in itertools.combinations(U, 2):
                iT, iD = iu[T], iu[D]
                for nm, Wx, cand in (("zs", W, Wcand),):
                    s_T_onT = Zte[T] @ Wx[iT]; s_D_onT = Zte[T] @ Wx[iD]
                    s_T_onD = Zte[D] @ Wx[iT]; s_D_onD = Zte[D] @ Wx[iD]
                    disc_T = s_T_onT - s_D_onT
                    disc_D = s_T_onD - s_D_onD
                    a0 = float(auc_of(disc_T, disc_D))
                    res["A"].setdefault(nm, {}).setdefault(
                        f"{T}|{D}", []).append(
                        dict(auc_dis=a0, sep=abs(a0 - 0.5) * 2))
                    # margin 版（仅 zs）
                    if nm == "zs":
                        def marg(Z, j):
                            L = Z @ cand.T
                            k = U.index(j)
                            return L[:, k] - np.delete(L, k, 1).max(1)
                        mT = marg(Zte[T], T) - marg(Zte[T], D)
                        mD = marg(Zte[D], T) - marg(Zte[D], D)
                        res["A"].setdefault("zs-m", {}).setdefault(
                            f"{T}|{D}", []).append(
                            dict(auc_dis=float(auc_of(mT, mD))))
                # RF / SVM：异常分不依赖候选 ⇒ 只能做「T 谱 vs D 谱」同函数比较
                res["A"].setdefault("rf", {}).setdefault(
                    f"{T}|{D}", []).append(
                    dict(auc_dis=float(auc_of(f_rf[T], f_rf[D])),
                         sep=abs(float(auc_of(f_rf[T], f_rf[D])) - 0.5) * 2))
                res["A"].setdefault("svm", {}).setdefault(
                    f"{T}|{D}", []).append(
                    dict(auc_dis=float(auc_of(f_sv[T], f_sv[D])),
                         sep=abs(float(auc_of(f_sv[T], f_sv[D])) - 0.5) * 2))
                # Cohen's d：RF 分数对两个未见类的重合度
                a, b = f_rf[T], f_rf[D]
                sd = np.sqrt((a.var() + b.var()) / 2) + 1e-12
                res["C"].setdefault("rf", {}).setdefault(
                    f"{T}|{D}", []).append(
                    dict(d=float(abs(a.mean() - b.mean()) / sd),
                         mean_T=float(a.mean()), mean_D=float(b.mean())))
                c, d2 = f_sv[T], f_sv[D]
                sd2 = np.sqrt((c.var() + d2.var()) / 2) + 1e-12
                res["C"].setdefault("svm", {}).setdefault(
                    f"{T}|{D}", []).append(
                    dict(d=float(abs(c.mean() - d2.mean()) / sd2)))

            # ---------- Stage B：候选集排序 ----------
            for u in U:
                L = Zte[u] @ Wcand.T                 # [n, |U|]
                k = U.index(u)
                Lr = Zte[u] @ Wcand_r.T
                for nm, Lx in (("zs", L), ("zs-rand", Lr)):
                    pred = np.array(U)[Lx.argmax(1)]
                    rk = (np.argsort(np.argsort(-Lx, 1), 1)[:, k] + 1)
                    res["B"].setdefault(nm, {}).setdefault(u, []).append(
                        dict(top1=float(np.mean(pred == u)),
                             mrr=float(mrr_of(rk)), n=len(Zte[u])))
                    # hubness：argmax 落在各候选上的频率（是否集中于吸引子）
                    res.setdefault("HUB", {}).setdefault(nm, {}).setdefault(
                        u, []).append({v: float(np.mean(pred == v))
                                       for v in U})
                # margin 版
                Lm = L - (np.sort(L, 1)[:, -2:-1])
                pred = np.array(U)[Lm.argmax(1)]
                rk = (np.argsort(np.argsort(-Lm, 1), 1)[:, k] + 1)
                res["B"].setdefault("zs-m", {}).setdefault(u, []).append(
                    dict(top1=float(np.mean(pred == u)),
                         mrr=float(mrr_of(rk)), n=len(Zte[u])))
                # RF：分数不依赖候选 ⇒ top-1 未定义；退化基线一并记
                res["B"].setdefault("rf-degenerate", {}).setdefault(
                    u, []).append(dict(top1=1.0 / len(U), n=len(Zte[u])))

            # ---------- Stage D：接入成本 ----------
            t0 = time.time()
            A = head_ridge_R(S_o, Z_old, lam_main)
            s_new = (S_all[iu[U[0]]] - mu_S[0]) @ A       # 本文：一次外推
            t_ours = (time.time() - t0) * 1e6              # µs
            t0 = time.time()
            Xn = np.concatenate([Xtr_s, np.zeros((5, Xtr_s.shape[1]))])
            yn = np.concatenate([ytr, np.full(5, ytr.max() + 1)])
            RandomForestClassifier(n_estimators=300, random_state=0,
                                   n_jobs=-1).fit(Xn, yn)
            t_rf_add = (time.time() - t0) * 1e3            # ms
            res["D"].setdefault("cost", []).append(
                dict(ours_us=float(t_ours), rf_add_ms=float(t_rf_add),
                     rf_fit_ms=float(t_rf_fit * 1e3)))
            print(f"[exp78] seed={seed} U={U} ok", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp78] 已写出 out/{args.out}")
    write_report(res, eids, args)


def split_group_w(X, gid, rng, frac=OLD_FRAC):
    gs = rng.permutation(np.unique(gid))
    tr_g = set(gs[:max(1, int(round(frac * len(gs))))].tolist())
    m = np.isin(gid, list(tr_g))
    return np.where(m)[0], np.where(~m)[0], None


def write_report(res, eids, args):
    from scipy.stats import ttest_1samp, ttest_rel
    L = []
    w = L.append
    lam_arms = [a for a in res["A"] if a.startswith("zs-lam")]
    lam_arms.sort(key=lambda s: float(s[6:]))
    ARMS_A = ["zs", "zs-m", "zs-rand", "rf", "svm"] + lam_arms
    ARMS_B = ["zs", "zs-m", "zs-rand", "rf-degenerate", "rf-oracle"]

    def flatA(arm, key="auc_dis"):
        out = {}
        for pair, vs in res["A"].get(arm, {}).items():
            v = [x[key] for x in vs if key in x]
            if v:
                out[pair] = float(np.mean(v))
        return out

    w("# exp78 —— 归因不可行性证明：RF/SVM 能否回答「掺的是哪一种」？\n")
    w("> 回应审稿人第一质疑：*本文做了大量设计，检出却不如几行的 RF/SVM，创新价值在哪？*\n")
    w("本文立场：**「检出」确实没有超过 RF/SVM（exp77 Δ=−0.026, p=0.19，不辩）。**\n")
    w("但「识别/归因」是**另一个任务**——本文主张 RF/SVM 在该任务上**结构性无输出**，\n")
    w("而非「做得差」。以下四组证据。\n")

    w("## 一、定向判别 auc_dis（T 谱 vs D 谱，T、D 均为未见化合物）\n")
    w("判别函数 = `s_T(x) − s_D(x)`。0.5 = 完全无法区分两种未知掺假物。\n")
    w("`sep = 2·|auc−0.5|`：只看**能不能分开**，不计方向；\n")
    w("`auc_dis > 0.5` 才代表**指名正确**。两者之差 = 「分得开却指错名」。\n")
    w("| 臂 | auc_dis（指名） | sep（可分） | vs 0.5 的 p | 对数 |\n|---|---|---|---|---|")
    for a in ARMS_A:
        d = flatA(a)
        if not d:
            continue
        s = flatA(a, "sep")
        v = np.array(list(d.values()))
        p = ttest_1samp(v, 0.5).pvalue
        sv = f"{np.mean(list(s.values())):.3f}" if s else "—"
        w(f"| `{a}` | **{v.mean():.3f}** | {sv} | {p:.2e} | {len(v)} |")

    w("\n**读法**：`rf` / `svm` 若 ≈0.5，说明其异常分**不编码掺杂物身份**——\n")
    w("它能说「这批样品我不认识」，但说不出「是 T 还是 D」。\n")
    if lam_arms:
        w("\n### λ 敏感性（化学外推强度）\n")
        w("| λ | auc_dis | 可分性 sep（不计方向） | 候选 w 行间 cos | ‖w_T−w_D‖ |\n|---|---|---|---|---|")
        for a in lam_arms:
            d = flatA(a)
            s = flatA(a, "sep")
            cw = [x["cos_cand"] for vs in res["A"][a].values() for x in vs
                  if "cos_cand" in x]
            dw = [x["dw"] for vs in res["A"][a].values() for x in vs
                  if "dw" in x]
            if d:
                w(f"| {a[6:]} | {np.mean(list(d.values())):.3f} | "
                  f"{np.mean(list(s.values())):.3f} | "
                  f"{np.mean(cw):.4f} | {np.mean(dw):.3f} |")
        w("\n`sep = 2·|auc−0.5|`：可分但**方向不受控**也会得高分；")
        w("只有 `auc_dis > 0.5` 才算「指名正确」。\n")

    sc = res.get("SCOS", [])
    if sc:
        w(f"\n### 化学码本结构诊断：候选化学向量行间 cos = "
          f"**{np.mean([x['cos'] for x in sc]):.4f}**\n")
        w("> 若 ≈1.0，说明化学表示把所有掺杂物映射到几乎同一点，")
        w("中心化后的方向差异极小 ⇒ 外推头 `w_T−w_D` 模长趋零 ⇒ 归因必然失败。")
        w("这是**表示问题**，不是对齐方法问题。\n")

    w("\n## 二、候选集排序（真实 NTS 操作：已知掺了 U 中一种，问是哪种）\n")
    w("候选集 = 全部未见化合物（3 选 1，随机基线 top-1 = 0.333）。\n")
    w("| 臂 | top-1 | MRR | 实体数 |\n|---|---|---|---|")
    for a in ARMS_B:
        per_t1, per_mrr = {}, {}
        for e, vs in res["B"].get(a, {}).items():
            t1 = [x["top1"] for x in vs]
            mr = [x["mrr"] for x in vs if "mrr" in x]
            if t1:
                per_t1[e] = float(np.mean(t1))
            if mr:
                per_mrr[e] = float(np.mean(mr))
        if not per_t1:
            continue
        mr = f"{np.mean(list(per_mrr.values())):.3f}" if per_mrr else "—"
        note = ""
        if a == "rf-degenerate":
            note = " ⚠️ **未定义**：RF 的分数不依赖候选集，形式上退化为随机"
        if a == "rf-oracle":
            note = " ⚠️ **作弊上界**：已把 U 三类放进训练集（需标样，非零样本）"
        w(f"| `{a}`{note} | {np.mean(list(per_t1.values())):.3f} | {mr} | "
          f"{len(per_t1)} |")

    w("\n## 三、RF/SVM 异常分对两个未见类的重合度（Cohen's d）\n")
    w("d≈0 ⇒ 两类得分分布完全重合 ⇒ 不可能据此归因。\n")
    w("| 臂 | Cohen's d | 对数 |\n|---|---|---|")
    for a in ("rf", "svm"):
        ds = []
        for pair, vs in res["C"].get(a, {}).items():
            ds += [x["d"] for x in vs]
        if ds:
            w(f"| `{a}` | {np.mean(ds):.3f} | {len(ds)} |")

    hub = res.get("HUB", {})
    if hub:
        w("\n## 二·补、hubness：argmax 是否集中于少数吸引子\n")
        w("`top-1` 与 `auc_dis` 不一致时，通常意味着候选里存在吸引子把样本吸走。\n")
        w("| 臂 | 最热候选吸走比例 | 基尼（不均度） |\n|---|---|---|")
        for nm, d in hub.items():
            hot, gini = [], []
            for u, vs in d.items():
                for x in vs:
                    p = np.array(sorted(x.values()))
                    hot.append(p.max())
                    ps = p / p.sum()
                    gini.append(float(
                        1 - np.sum(ps ** 2)))          # 简化的不均度
            if hot:
                w(f"| `{nm}` | {np.mean(hot):.3f} | {np.mean(gini):.3f} |")

    w("\n## 四、接入成本：新增一个候选掺杂物\n")
    c = res["D"].get("cost", [])
    if c:
        w("| 项 | 本文 | RF |\n|---|---|---|")
        w(f"| 新增 1 个分子 | **{np.mean([x['ours_us'] for x in c]):.1f} µs** | "
          f"{np.mean([x['rf_add_ms'] for x in c]):.0f} ms（且需先取得标样） |")
        w(f"| 完整拟合一次 | — | "
          f"{np.mean([x['rf_fit_ms'] for x in c]):.0f} ms |")

    w("\n## 判据\n")
    zs = flatA("zs"); rfA = flatA("rf")
    com = sorted(set(zs) & set(rfA))
    if len(com) >= 4:
        x = np.array([zs[p] for p in com]); y = np.array([rfA[p] for p in com])
        w(f"\n- **H78a**（归因能力）：`auc_dis` 本文 `zs` {x.mean():.3f} vs rf "
          f"{y.mean():.3f}（Δ={np.mean(x - y):+.3f}, "
          f"p={ttest_rel(x, y).pvalue:.3g}, n={len(com)}）⇒ "
          f"{'化学先验可归因' if np.mean(x - y) > 0.1 else '不成立'}")
        pr = ttest_1samp(y, 0.5).pvalue
        w(f"- **H78b**（RF 不可行性）：rf `auc_dis` {y.mean():.3f} 与 0.5 "
          f"{'无显著差异' if pr > 0.05 else '显著差异'}（p={pr:.3g}）⇒ "
          f"{'RF 无法区分两种未知掺假物' if pr > 0.05 else 'RF 仍有弱区分力'}")
    zr = flatA("zs-rand")
    com2 = sorted(set(zs) & set(zr))
    if len(com2) >= 4:
        x = np.array([zs[p] for p in com2]); z = np.array([zr[p] for p in com2])
        w(f"- **H78c**（随机码本复检，针对 exp73 隐患）：`auc_dis` 本文 "
          f"{x.mean():.3f} vs rand {z.mean():.3f}"
          f"（Δ={np.mean(x - z):+.3f}, p={ttest_rel(x, z).pvalue:.3g}）⇒ "
          f"{'化学语义确为归因来源' if np.mean(x - z) > 0.1 else '⚠️ 随机码本同样可归因'}")

    p = os.path.join(OUT, "exp78_p6_attribution_proof.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[exp78] 已写出 {p}")


if __name__ == "__main__":
    main()
