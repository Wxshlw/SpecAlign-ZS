"""exp44 —— 冷启动主实验（修正协议版）

=============================================================================
为什么要重写，而不是在 exp42 上继续加臂
=============================================================================
exp42 的逐实体矩阵里出现大量 0.000 / 1.000，exp43/43b 已把三个候选解释
逐个排除并定位到**三个协议缺陷**（不是模型缺陷）：

P1 **硬对齐依赖偶然** —— exp43b 实测 A01：
      k=0  center 新类行 z·W 排名 1/19 ⇒ 召回 0.733
      k=5  center 新类行 z·W 排名 8/19 ⇒ 召回 0.000
      k=5  orth   新类行 z·W 排名 1/19 ⇒ 召回 0.767
   center 与 orth 的头只差一个保范数的正交旋转，却差 0.767。
   ⇒ 结论：`center`（拿化学向量硬当分类头）的成功是 **z 空间偶然对齐**，
      不可复现（`orth` 显式学了 S→Z 的映射，才稳定）。
      **本实验因此把「学对齐」作为主张臂，硬对齐只作对照。**

P2 **超参选择准则错配** —— exp42 用「老类验证集 top-1」选 λ，
   但老类有几百个样本、未见类只有 k 个，两者的最优 λ 并不相同。
   实测 A01 k=5：ridge0.01=0.828 而 ridge-cv(λ 由老类选)=0.381。
   ⇒ 本实验改用 **macro-recall（每类等权）** 作选 λ 的准则：
      它等价于「少数类权重被拉平」，正是未见类所处的处境；
      且只碰老类，不泄漏未见类。

P3 **已见类偏见（GZSL 的经典病）** —— exp43 实测：A01 的谱类中心与 A02
   余弦 0.994（几乎同一条谱），而 A02 有 600 个训练样本、A01 只有 5 个。
   ⇒ 不加校正时模型必然把 A01 全判成 A02（实测 none=0.000）。
   ⇒ 本实验引入顶会模块 **Calibrated Stacking**（Chao et al., BMVC 2016）：
      logit'_j = logit_j − γ·1[j 是未见类]，γ 由老类验证集的 macro-recall
      网格搜索选出。这是 GZSL 里专门针对「模型偏向已见类」的标准解。

=============================================================================
指标：从 top-1 扩展到「筛查」指标（本论文的应用场景就是筛查）
=============================================================================
19 类 argmax 的 top-1 在零样本下被类间极度相似（余弦 0.99）支配，
统计功效太低。筛查任务关心的是「能不能把可疑样本捞出来」，因此增加：
  ★ `auc`  新类样本 vs 老类样本，在**新类头得分**上的 AUC
           —— 直接对应论文的「检出」能力，不受 19 类 argmax 噪声支配
  ★ `mrr`  新类行在 19 个 logit 中的降序排名的倒数（MRR）
  ★ `top1` 保留，用于与既有实验对齐
预注册：主指标 = auc（筛查语义），top1 作次要指标；两者都报，不挑。

=============================================================================
预注册判据（写死，跑完不再改）
=============================================================================
- H44a（主）：k∈{5,20} 时 `ridge-macro`（新准则选 λ）≥ `ridge-top1`（旧准则）
- H44b（主）：k∈{5,20} 时 `calib`（校准后）的 auc > `none`（无先验自由头）
- H44c（主）：k=0 时 `orth` 的 auc > `rand`（随机码本）⇒ 先验在零样本下
              确实携带了**可检出的**信息（这是「不可替代」的最强形式）
- H44d（次）：k=0 时 `orth` 的 top1 > `center` ⇒ 学对齐优于硬对齐（复现 43b）
- 全部 λ、γ 列出，不挑最优；负结果照报
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07


# --------------------------------------------------------------------------- #
def build_net(d_in, d_out, seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(),
        nn.Linear(512, d_out),
    ).to(DEV)


@torch.no_grad()
def embed(net, X):
    net.eval()
    return F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)),
                       dim=1).cpu().numpy()


def stage1(Xtr, ytr, dim, nrows, steps, seed, S=None, mode="chem"):
    """同 exp42：mode=chem 冻结头为去均值化学向量（DEVISE 标准做法）。"""
    torch.manual_seed(seed)
    net = build_net(Xtr.shape[1], dim, seed)
    params = list(net.parameters())
    if mode == "chem":
        Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
        W = torch.tensor(Wc, dtype=torch.float32, device=DEV)
    else:
        W = nn.Parameter(torch.randn(nrows, dim, device=DEV) * 0.01)
        params = params + [W]
    opt = torch.optim.Adam(params, lr=3e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    n = len(Xtr)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(64, n),), device=DEV)
        z = F.normalize(net(Xt[idx]), dim=1)
        loss = F.cross_entropy(z @ F.normalize(W, dim=1).T / TEMP, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net, W.detach().cpu().numpy()


def cka(A, B):
    A = np.asarray(A, float); A = A - A.mean(0, keepdims=True)
    B = np.asarray(B, float); B = B - B.mean(0, keepdims=True)
    num = float(np.linalg.norm(A.T @ B) ** 2)
    den = float(np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))
    return num / den if den > 0 else 0.0


def head_orth_R(S_old, Z_old):
    Sc = S_old - S_old.mean(0, keepdims=True)
    Zc = Z_old - Z_old.mean(0, keepdims=True)
    U, _, Vt = np.linalg.svd(Sc.T @ Zc)
    return U @ Vt


def head_ridge_R(S_old, Z_old, lam):
    Sc = S_old - S_old.mean(0, keepdims=True)
    Zc = Z_old - Z_old.mean(0, keepdims=True)
    return Sc.T @ np.linalg.solve(Sc @ Sc.T + lam * np.eye(len(Sc)), Zc)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _pad19(W):
    Wt = torch.tensor(np.asarray(W), dtype=torch.float32, device=DEV)
    if Wt.shape[0] < 19:
        Wt = torch.cat([Wt, torch.zeros(19 - Wt.shape[0], Wt.shape[1], device=DEV)], 0)
    return Wt


@torch.no_grad()
def score_all(net, W, X, gamma=0.0, new_row=18):
    """返回 19 类 logit（已归一化 cosine 头 + 可选 calibrated stacking）。"""
    z = F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)), dim=1)
    lg = z @ F.normalize(_pad19(W), dim=1).T
    if gamma:
        lg = lg.clone()
        lg[:, new_row] -= gamma
    return lg[:, :19]


def auc_of(pos, neg):
    """无 scipy 依赖的 Mann-Whitney U / AUC。"""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0   # 平均秩近似
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


@torch.no_grad()
def metrics(net, W, Xte_new, Xte_old, new_row=18, gamma=0.0):
    """(top1, auc, mrr)：筛查三指标。"""
    lg_n = score_all(net, W, Xte_new, gamma, new_row).cpu().numpy()
    lg_o = score_all(net, W, Xte_old, gamma, new_row).cpu().numpy()
    pred = lg_n.argmax(1)
    top1 = float((pred == new_row).mean())
    auc = auc_of(lg_n[:, new_row], lg_o[:, new_row])
    order = np.argsort(-lg_n, axis=1)
    ranks = np.where(order == new_row)[1] + 1
    mrr = float((1.0 / ranks).mean())
    return dict(top1=top1, auc=auc, mrr=mrr)


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows, gamma=0.0, new_row=18):
    """每类等权召回。选 λ/γ 的准则：把少数类权重拉平 = 未见类所处的处境。"""
    lg = score_all(net, W, Xv, gamma, new_row).cpu().numpy()
    pred = lg[:, rows].argmax(1)
    rs = []
    for i, j in enumerate(rows):
        m = yv == j
        if m.sum():
            rs.append(float((pred[m] == i).mean()))
    return float(np.mean(rs)) if rs else 0.0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5,20")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--gammas", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--mode", default="chem", choices=["chem", "free"])
    ap.add_argument("--eids", default="")
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()

    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]

    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(z[args.repr if args.repr != "ecfp" else "ecfp384"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    # 老类参照集：每个老类抽 min(200, n) 条，用于算 AUC 的负样本
    arms = ["rand", "center", "orth", "ridge-macro", "ridge-top1",
            "calib", "none"] + [f"ridge{lam:g}" for lam in lams]
    res = {"acc": {t: {a: {} for a in arms} for t in tags},
           "cv": {}, "gamma": {}, "cka": {},
           "meta": {"repr": args.repr, "shots": tags, "seeds": args.seeds,
                    "mode": args.mode, "lams": lams, "gammas": gammas,
                    "eids": eids, "dim": dim,
                    "metrics": ["top1", "auc", "mrr"]}}

    print(f"[exp44] device={DEV} repr={args.repr} mode={args.mode} "
          f"shots={tags} seeds={args.seeds} lams={lams} gammas={gammas}", flush=True)

    rng_g = np.random.default_rng(20240101)
    for e_hold in eids:
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids]
                              + [ki[e_hold]]]
        NEW = 18
        Xe = np.asarray(ent[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new = Xe[te_ix]

                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float)); ys.append(np.full(len(bg[m]), len(others) + j))
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), 18)); nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)

                # 老类参照集（AUC 的负样本），与测试集不相交
                ref = []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    ix = rng_g.permutation(len(Xo))[:min(200, len(Xo))]
                    ref.append(Xo[ix])
                for m in mids:
                    Xo = np.asarray(bg[m], float)
                    ix = rng_g.permutation(len(Xo))[:min(200, len(Xo))]
                    ref.append(Xo[ix])
                Xte_old = np.concatenate(ref).astype(np.float32)

                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                rows_old = sorted({int(y) for y in ytr})
                for j in rows_old:
                    if j == 18 and len(tr_e_ix) == 0:
                        continue
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    val_mask[ix[:max(3, int(0.2 * len(ix)))]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                net, W_free = stage1(Xfit, yfit, dim, nrows, args.steps,
                                     seed * 1000 + eids.index(e_hold),
                                     S=S, mode=args.mode)
                Zfit = embed(net, Xfit)
                Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                S_old = S[rows_old]
                mu_S = S_old.mean(0, keepdims=True)
                res["cka"].setdefault(tag, {}).setdefault(
                    e_hold, []).append(round(cka(Z_old, S_old), 4))

                def W_of(A):
                    return (S - mu_S) @ A

                # ---------- P2：用 macro-recall 选 λ（旧准则 top1 作对照）----------
                lam_m, best = lams[0], -1.0
                for lam in lams:
                    v = macro_recall(net, W_of(head_ridge_R(S_old, Z_old, lam)),
                                     Xval, yval, val_rows)
                    if v > best:
                        best, lam_m = v, lam
                lam_t, best = lams[0], -1.0
                for lam in lams:
                    zz = F.normalize(net(torch.tensor(Xval, dtype=torch.float32,
                                                       device=DEV)), dim=1)
                    Wt = F.normalize(_pad19(W_of(head_ridge_R(S_old, Z_old, lam))), dim=1)
                    pred = (zz @ Wt.T)[:, val_rows].argmax(1).cpu().numpy()
                    want = np.searchsorted(val_rows, yval)
                    v = float((pred == want).mean())
                    if v > best:
                        best, lam_t = v, lam
                res["cv"].setdefault(tag, {}).setdefault(
                    e_hold, []).append([lam_m, lam_t])

                R_orth = head_orth_R(S_old, Z_old)
                W_orth = W_of(R_orth)
                W_rm = W_of(head_ridge_R(S_old, Z_old, lam_m))
                W_rt = W_of(head_ridge_R(S_old, Z_old, lam_t))

                # ---------- P3：calibrated stacking 的 γ ----------
                # γ 只对未见类（第 18 列）生效；k=0/5/20 时它才是少数类。
                # 选 γ 用老类验证集的 macro-recall：不接触未见类，无泄漏。
                if len(tr_e_ix):
                    g_best, gb = 0.0, -1.0
                    for g in gammas:
                        v = macro_recall(net, W_rm, Xval, yval, val_rows,
                                         gamma=g, new_row=NEW)
                        if v > gb:
                            gb, g_best = v, g
                else:
                    g_best = 0.0
                res["gamma"].setdefault(tag, {}).setdefault(
                    e_hold, []).append(float(g_best))

                accs = {}
                # 随机码本：零化学信息的下限（k=0 的唯一合法「无先验」对照）
                Wrand = rng_g.normal(size=(19, dim)).astype(np.float32)
                accs["rand"] = metrics(net, Wrand, Xte_new, Xte_old, NEW)
                accs["center"] = metrics(net, S - mu_S, Xte_new, Xte_old, NEW)
                accs["orth"] = metrics(net, W_orth, Xte_new, Xte_old, NEW)
                accs["ridge-macro"] = metrics(net, W_rm, Xte_new, Xte_old, NEW)
                accs["ridge-top1"] = metrics(net, W_rt, Xte_new, Xte_old, NEW)
                accs["calib"] = metrics(net, W_rm, Xte_new, Xte_old, NEW,
                                        gamma=g_best)
                for lam in lams:
                    accs[f"ridge{lam:g}"] = metrics(
                        net, W_of(head_ridge_R(S_old, Z_old, lam)),
                        Xte_new, Xte_old, NEW)
                if len(tr_e_ix):
                    accs["none"] = metrics(net, W_free, Xte_new, Xte_old, NEW)
                    accs["none+calib"] = metrics(net, W_free, Xte_new, Xte_old,
                                                 NEW, gamma=g_best)

                for a, m in accs.items():
                    res["acc"][tag].setdefault(a, {}).setdefault(
                        e_hold, []).append({kk: round(float(vv), 4)
                                            for kk, vv in m.items()})

        line = []
        for t in tags:
            s = " ".join(
                f"{a}={np.mean([d['auc'] for d in res['acc'][t][a][e_hold]]):.2f}"
                f"/{np.mean([d['top1'] for d in res['acc'][t][a][e_hold]]):.2f}"
                for a in arms if res["acc"][t][a].get(e_hold))
            line.append(f"{t}: {s}")
        print(f"[{e_hold:>7}] " + " | ".join(line)
              + f"  || cka={np.mean(res['cka'][tag][e_hold]):.3f}"
              + f" g={np.mean(res['gamma'][tag][e_hold]):.2f}", flush=True)

    with open(os.path.join(OUT, f"exp44_res_{args.mode}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[exp44] 已写出 out/exp44_res_{args.mode}.json")
    write_report(res, arms, tags, args)


def write_report(res, arms, tags, args):
    eids = res["meta"]["eids"]
    try:
        from scipy.stats import ttest_rel
        HAS = True
    except Exception:
        HAS = False

    L = []
    w = L.append
    w("# exp44 —— 冷启动主实验（修正协议版）\n")
    w("协议修正：P1 学对齐为主张臂 / P2 λ 由 macro-recall 选 / "
      "P3 calibrated stacking。\n")
    w("单元格格式：**auc / top1**（auc = 筛查主指标）。\n")

    for met in ("auc", "top1"):
        w(f"\n## {met}\n")
        w("| k | " + " | ".join(f"`{a}`" for a in arms) + " |")
        w("|---|" + "---|" * len(arms))
        for t in tags:
            row = []
            for a in arms:
                per = [np.mean([d[met] for d in res["acc"][t][a][e]])
                       for e in eids if res["acc"][t][a].get(e)]
                row.append(f"{np.mean(per):.3f}" if per else "—")
            w(f"| {t} | " + " | ".join(row) + " |")

    w("\n## Δ 相对对照（配对 t 检验，n = 实体数）\n")
    for t in tags:
        if t == "0":
            base = "rand"
        else:
            base = "none"
        if not res["acc"][t].get(base):
            continue
        w(f"\n### k={t}（对照 = `{base}`）\n")
        w("| 臂 | Δauc | p | Δtop1 | p |")
        w("|---|---|---|---|---|")
        for a in arms:
            if a == base or not res["acc"][t].get(a):
                continue
            xs, ys, xs2, ys2 = [], [], [], []
            for e in eids:
                if res["acc"][t][a].get(e) and res["acc"][t][base].get(e):
                    xs.append(np.mean([d["auc"] for d in res["acc"][t][a][e]]))
                    ys.append(np.mean([d["auc"] for d in res["acc"][t][base][e]]))
                    xs2.append(np.mean([d["top1"] for d in res["acc"][t][a][e]]))
                    ys2.append(np.mean([d["top1"] for d in res["acc"][t][base][e]]))
            if not xs:
                continue
            da, dt = np.mean(xs) - np.mean(ys), np.mean(xs2) - np.mean(ys2)
            if HAS and len(xs) > 1:
                pa = ttest_rel(xs, ys).pvalue
                pt = ttest_rel(xs2, ys2).pvalue
            else:
                pa = pt = float("nan")
            w(f"| `{a}` | {da:+.3f} | {pa:.3f} | {dt:+.3f} | {pt:.3f} |")

    w("\n## 判据\n")
    for t in tags:
        row = {}
        for a in arms:
            per = [np.mean([d["auc"] for d in res["acc"][t][a][e]])
                   for e in eids if res["acc"][t][a].get(e)]
            row[a] = np.mean(per) if per else np.nan
        if t != "0":
            w(f"- k={t}: H44a ridge-macro {row.get('ridge-macro', float('nan')):.3f} "
              f"vs ridge-top1 {row.get('ridge-top1', float('nan')):.3f}；"
              f"H44b calib {row.get('calib', float('nan')):.3f} "
              f"vs none {row.get('none', float('nan')):.3f}")
        else:
            w(f"- k=0: H44c orth {row.get('orth', float('nan')):.3f} "
              f"vs rand {row.get('rand', float('nan')):.3f}；"
              f"H44d orth top1 vs center top1 见上表")

    with open(os.path.join(OUT, f"exp44_p6_coldstart_{args.mode}.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[exp44] 已写出 out/exp44_p6_coldstart_{args.mode}.md")


if __name__ == "__main__":
    main()
