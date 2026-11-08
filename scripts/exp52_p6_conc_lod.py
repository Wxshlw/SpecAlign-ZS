"""exp52 —— 冷启动筛查的**浓度轴**：化学先验自己的「现实检测限」

=============================================================================
为什么这是全文最关键的缺口
=============================================================================
exp49/50 确立了 k=0 下化学先验的不可替代性（auc 0.856 / auc_dis 0.845）。
但那些数字是**把所有掺混浓度混在一起**算出来的。

本文标题主题是「FTIR 低浓度失效与现实检测限」。审稿人一定会问：
    「你这套零样本筛查，在**真实掺假浓度**（1%–5%）下还能用吗？」
若不能，前面的结论就悬空；若能，就得到全文最有力的统一结论：

    ★ 化学先验的筛查能力随掺混浓度下降而退化，
      且存在一个**可测的浓度检测限 c\***
      ——这正是「现实检测限」在**零样本冷启动**维度的对应物，
      与 exp13/15/18（浓度 LOD）构成完整闭环。

=============================================================================
设计（训练一次，按浓度分层评估 ⇒ 极省算力）
=============================================================================
对每个**有浓度标签**的 held-out 实体 T（A01–A07 共 7 个 + HSYRUP = 8 个）：
  - 训练集：其余实体（70%）+ 基质（70%），k=0 时不含 T 的任何样本
  - **固定 30% 的 T 样本作测试**（与 exp49 同协议），再按浓度分层
  - 老类负样本：老类 held-out 30%（与训练集不相交，无泄漏）

指标（按浓度档位 c 分层）：
  `auc(c)`      = AUC( T@c  vs  老类 held-out )
  `sens@FPR5(c)`= 在老类得分 95 分位处取阈值时，T@c 的命中率（筛查操作点）
  `top1(c)`     = 19 类 argmax 命中 T

判据（预注册）
--------------
- H52a：本文 `ridge` 的 auc(c) **随 c 单调上升**（Spearman ρ>0）
        ⇒ 先验的筛查能力确实受浓度支配，存在检测限
- H52b：定义 c\* = 使 auc(c) 首次 ≥ 0.80 的最低浓度档；
        报告本文与外部方法的 c\*，本文的 c\* **不高于**最强外部方法
- H52c：**低浓度段**（c ≤ 5%）本文 auc 显著高于最强外部方法
        ⇒ 先验的价值恰恰集中在"数据最不足以区分"的低浓度区
        （与 exp49 的 worst-case 稳健性、exp50 的定向性同一机制）
- H52d：低浓度段 `sens@FPR5` 本文 ≥ 外部方法

★ 关键对照：`rand`（零化学信息随机码本）同步分层，
  用于把「浓度效应」与「化学语义效应」分离开。
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07
OLD_FRAC = 0.70


def build_net(d_in, d_out, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(),
                         nn.Linear(512, d_out)).to(DEV)


@torch.no_grad()
def embed(net, X):
    net.eval()
    return F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)),
                       dim=1).cpu().numpy()


def stage1(Xtr, ytr, dim, nrows, steps, seed, S=None):
    torch.manual_seed(seed)
    net = build_net(Xtr.shape[1], dim, seed)
    Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
    W = torch.tensor(Wc, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(list(net.parameters()), lr=3e-4)
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


def head_ridge_R(S_old, Z_old, lam):
    Sc = S_old - S_old.mean(0, keepdims=True)
    Zc = Z_old - Z_old.mean(0, keepdims=True)
    return Sc.T @ np.linalg.solve(Sc @ Sc.T + lam * np.eye(len(Sc)), Zc)


def nrm(A):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def auc_of(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def cos_to(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    rs = [float((pred[yv == j] == i).mean()) for i, j in enumerate(rows)
          if (yv == j).sum()]
    return float(np.mean(rs)) if rs else 0.0


@torch.no_grad()
def score_col(net, W, X, col):
    z = F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    lg = (z @ Wt.T).cpu().numpy()[:, :W.shape[0]]
    return lg[:, col], lg.argmax(1)


BINS = [(2.0, "<=2"), (5.0, "3-5"), (10.0, "6-10"), (20.0, "11-20"),
        (1e9, ">20")]


def bin_of(c):
    for hi, lab in BINS:
        if c <= hi:
            return lab
    return ">20"


def strat_metrics(s_new, s_old, pred_new, new_row, conc_te):
    """按**绝对浓度分箱**分层（不同实体档位不同，必须分箱才能 macro 平均）"""
    thr = np.quantile(s_old, 0.95)
    out = {}
    for lab in [b[1] for b in BINS]:
        m = np.asarray([bin_of(c) == lab for c in conc_te])
        if m.sum() < 3:
            continue
        out[lab] = dict(
            auc=round(float(auc_of(s_new[m], s_old)), 4),
            sens5=round(float((s_new[m] > thr).mean()), 4),
            top1=round(float((pred_new[m] == new_row).mean()), 4),
            n=int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    eids = [e for e in sorted(ent) if conc.get(e) is not None]
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    ARMS = ["ridge", "rand", "rf", "svm-rbf", "centroid-cos"]
    res = {"acc": {t: {a: {} for a in ARMS} for t in tags},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "arms": ARMS, "lams": lams}}
    print(f"[exp52] device={DEV} 有浓度标签的实体={eids} shots={tags} "
          f"seeds={args.seeds}", flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        NEW = len(others) + len(mids)
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        Xe = np.asarray(ent[e_hold], float)
        ce = np.asarray(conc[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new, conc_te = Xe[te_ix], ce[te_ix]

                parts, ys, refs = [], [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                    refs.append(Xo[pp[ntr:]])
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    refs.append(Xo[pp[ntr:]])
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(refs).astype(np.float32)

                # ---------- 外部基线（k=0 时只能用异常分）----------
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xn_s, Xo_s = sc.transform(Xte_new), sc.transform(Xte_old)
                rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                         random_state=0).fit(Xtr_s, ytr)
                cj = np.unique(ytr)
                cent = np.stack([Xtr_s[ytr == j].mean(0) for j in cj])
                Dn, Do = cos_to(Xn_s, cent), cos_to(Xo_s, cent)
                if len(tr_e_ix):
                    s_rf = rf.predict_proba(Xn_s)[:, list(rf.classes_).index(NEW)]
                    o_rf = rf.predict_proba(Xo_s)[:, list(rf.classes_).index(NEW)]
                    s_sv = sv.predict_proba(Xn_s)[:, list(sv.classes_).index(NEW)]
                    o_sv = sv.predict_proba(Xo_s)[:, list(sv.classes_).index(NEW)]
                    ic = list(cj).index(NEW)
                    s_ce, o_ce = Dn[:, ic], Do[:, ic]
                    p_rf = rf.predict(Xn_s)
                else:
                    s_rf, o_rf = (1 - rf.predict_proba(Xn_s).max(1),
                                  1 - rf.predict_proba(Xo_s).max(1))
                    s_sv, o_sv = (1 - sv.predict_proba(Xn_s).max(1),
                                  1 - sv.predict_proba(Xo_s).max(1))
                    s_ce, o_ce = 1 - Dn.max(1), 1 - Do.max(1)
                    p_rf = np.full(len(Xn_s), -1)
                for nm, (sn, so, pn) in (
                        ("rf", (s_rf, o_rf, p_rf)),
                        ("svm-rbf", (s_sv, o_sv, np.full(len(Xn_s), -1))),
                        ("centroid-cos", (s_ce, o_ce, np.full(len(Xn_s), -1)))):
                    d = strat_metrics(sn, so, pn, NEW, conc_te)
                    res["acc"][tag][nm].setdefault(e_hold, []).append(d)

                # ---------- 本文 ----------
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in sorted({int(y) for y in ytr}):
                    if j == NEW and len(tr_e_ix) == 0:
                        continue
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    nv = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:nv]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                net, _ = stage1(Xfit, yfit, dim, nrows, args.steps,
                                seed * 1000 + eids.index(e_hold), S=S)
                Zfit = embed(net, Xfit)
                rows_old = sorted({int(y) for y in yfit})
                Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                S_old, mu_S = S[rows_old], S[rows_old].mean(0, keepdims=True)
                lam_b, bv = lams[0], -1.0
                for lam in lams:
                    Wc = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam)
                    v = macro_recall(net, Wc, Xval, yval, val_rows)
                    if v > bv:
                        bv, lam_b = v, lam
                W_chem = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)
                rng_r = np.random.default_rng(seed * 977 + 5)
                Wr = nrm(rng_r.normal(size=W_chem.shape).astype(np.float32))

                for nm, W in (("ridge", W_chem), ("rand", Wr)):
                    sn, pn = score_col(net, W, Xte_new, NEW)
                    so, _ = score_col(net, W, Xte_old, NEW)
                    res["acc"][tag][nm].setdefault(e_hold, []).append(
                        strat_metrics(sn, so, pn, NEW, conc_te))

        # 进度：按浓度档位打印 ridge / 最强外部
        lv = [c for c in [b[1] for b in BINS]
              if any(c in d for d in res["acc"][tags[0]]["ridge"].get(e_hold, []))]
        s = " ".join(
            f"{c}:{np.mean([d[c]['auc'] for d in res['acc'][tags[0]]['ridge'][e_hold] if c in d]):.2f}"
            for c in lv)
        print(f"[{e_hold:>7}] k={tags[0]} ridge auc 按浓度: {s}", flush=True)

    with open(os.path.join(OUT, "exp52_conc.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp52] 已写出 out/exp52_conc.json")
    write_report(res, ARMS, tags, eids)


def write_report(res, ARMS, tags, eids):
    try:
        from scipy.stats import spearmanr, ttest_rel
        HAS = True
    except Exception:
        HAS = False

    def agg(t, a, met):
        """macro 平均：每个实体先在自己 seeds 上求箱内均值，再跨实体等权。"""
        per_e = {}
        for e in eids:
            acc, cnt = {}, {}
            for d in res["acc"][t].get(a, {}).get(e, []):
                for lab, v in d.items():
                    if met in v:
                        acc[lab] = acc.get(lab, 0.0) + v[met]
                        cnt[lab] = cnt.get(lab, 0) + 1
            per_e[e] = {lab: acc[lab] / cnt[lab] for lab in acc}
        labs = [b[1] for b in BINS]
        out = {}
        for lab in labs:
            vs = [per_e[e][lab] for e in eids if lab in per_e[e]]
            if vs:
                out[lab] = (float(np.mean(vs)), len(vs))
        return out

    def agg_e(t, a, met):
        """返回 {实体: {浓度箱: seeds 均值}}——配对检验需要逐实体值。"""
        per_e = {}
        for e in eids:
            acc, cnt = {}, {}
            for d in res["acc"][t].get(a, {}).get(e, []):
                for lab, v in d.items():
                    if met in v:
                        acc[lab] = acc.get(lab, 0.0) + v[met]
                        cnt[lab] = cnt.get(lab, 0) + 1
            if acc:
                per_e[e] = {lab: acc[lab] / cnt[lab] for lab in acc}
        return per_e

    ORD = [b[1] for b in BINS]

    L = []
    w = L.append
    w("# exp52 —— 冷启动筛查的**浓度轴**：化学先验自己的「现实检测限」\n")
    w("按掺混浓度**绝对分箱**分层（不同实体浓度档位不同，必须分箱才能 macro 平均）。\n")
    w("`sens5` = 老类得分 95 分位处取阈值（FPR≈5%）时的命中率。`n` = 参与实体数。\n")

    for t in tags:
        w(f"\n## k={t}\n")
        for met in ("auc", "sens5", "top1"):
            w(f"\n### {met}\n")
            w("| 浓度箱% | " + " | ".join(f"`{a}`" for a in ARMS) + " | 实体数 |")
            w("|---|" + "---|" * len(ARMS) + "---|")
            d0 = agg(t, ARMS[0], met)
            for c in ORD:
                row, nn = [], 0
                for a in ARMS:
                    d = agg(t, a, met)
                    if c in d:
                        row.append(f"{d[c][0]:.3f}"); nn = max(nn, d[c][1])
                    else:
                        row.append("—")
                w(f"| {c} | " + " | ".join(row) + f" | {nn} |")

    w("\n## 判据\n")
    for t in tags:
        d = agg(t, "ridge", "auc")
        labs = [c for c in ORD if c in d]
        ys = np.array([d[c][0] for c in labs])
        xs = np.arange(len(labs))
        w(f"\n### k={t}")
        if HAS and len(labs) > 3:
            rho, p = spearmanr(xs, ys)
            w(f"- **H52a**：Spearman ρ(浓度箱序, ridge auc) = {rho:+.3f} (p={p:.4f}) "
              f"⇒ {'成立（浓度越高越好）' if rho > 0 and p < 0.05 else '不成立'}")
            w(f"   - ridge auc 逐箱："
              + ", ".join(f"{c}:{d[c][0]:.3f}" for c in labs))
        w("   - **c\\***（auc 首次 ≥0.80 的浓度箱）：")
        for a in ARMS:
            da = agg(t, a, "auc")
            hit = [c for c in ORD if c in da and da[c][0] >= 0.80]
            w(f"     - `{a}`: {hit[0] if hit else '未达到'}%")
        # 逐箱配对（实体为单位）
        if HAS:
            w("   - **逐箱配对（ridge − 外部，n=实体数）**：")
            for c in labs:
                re_ = agg_e(t, "ridge", "auc")
                seg = [f"{c} 箱:"]
                for a in ARMS:
                    if a == "ridge":
                        continue
                    oe = agg_e(t, a, "auc")
                    common = sorted(set(re_) & set(oe))
                    common = [e for e in common if c in re_[e] and c in oe[e]]
                    if len(common) < 3:
                        continue
                    x = np.array([oe[e][c] for e in common])
                    y = np.array([re_[e][c] for e in common])
                    seg.append(f"{a} Δ={np.mean(y - x):+.3f}(p={ttest_rel(y, x).pvalue:.3f},n={len(common)})")
                if len(seg) > 1:
                    w("     - " + " ".join(seg))

    with open(os.path.join(OUT, "exp52_p6_concentration.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp52] 已写出 out/exp52_p6_concentration.md")


if __name__ == "__main__":
    main()
