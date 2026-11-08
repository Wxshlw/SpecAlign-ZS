"""exp50 —— **定向筛查** vs **通用异常检测**：双重 held-out 判别实验

=============================================================================
为什么这是当前最关键的一个实验
=============================================================================
exp49（干净协议）暴露了一个对本文不利的事实：
    k=0 时 `rf`（1−max_prob 异常分）auc 0.925、`svm-rbf` 0.844，
    而本文 `ridge` 只有 0.797 ⇒ H49a 不成立。
审稿人会直接说：「用一个通用 novelty detector 就够了，要什么化学先验？」

**但这个对比在语义上是不对等的**，而这一点可以被实验证明：

    rf / svm / 1-NN 的「异常分」回答的是：
        「这条谱和我库里的东西都不一样吗？」（通用 novelty）
    本文的化学外推回答的是：
        「这条谱是不是**我怀疑的那个具体分子**？」（定向 screening）

真实监管场景里，实验室是**带着一个具体怀疑对象**去筛的
（例：听说出现了三聚氰胺的某个新结构类似物）。
食品基质本身存在批次、季节、产地带来的**正常新变异**。
通用异常检测会把这些**良性新变异一起报警** ⇒ 大量假阳性。

=============================================================================
实验设计：双重 held-out（target + novel distractor）
=============================================================================
同时留出**两个**实体：
    T  目标（suspect）—— 我们怀疑的分子
    D  干扰（distractor）—— 另一个同样不在库里的新化合物（良性新变异的替身）

训练集只包含：其余 11 个实体 + 6 个基质 + （k 条 T 的样本，若 k>0）
评估负样本 = 老类 held-out 30%  ∪  D 的全部样本

指标：
    `auc_all`   = AUC( T  vs  老类 ∪ D )        主指标（筛查真阳性率/假阳性率）
    `auc_old`   = AUC( T  vs  老类 )
    `auc_dis`   = AUC( T  vs  D )               ★ 判别性指标：能否把「怀疑的那个」
                                                  和「另一个新化合物」分开
    `top1`      = 17 类 argmax 命中 T

★ **核心预注册判据**
- H50a：k=0 时本文 `ridge` 的 `auc_dis` **显著 > 0.5**，
        而 `rf` / `svm-rbf` / `knn-cos` 的 `auc_dis` **≈ 0.5 或 < 0.5**
        ⇒ 通用异常检测**在原理上无法**区分两个都"新"的化合物；
          化学语义先验可以。**这是定向筛查不可替代的硬证据。**
- H50b：k=0 时本文 `auc_all` ≥ `rf` 的 `auc_all`
        ⇒ 加上"良性新变异"后，通用异常检测的优势消失甚至反转
- H50c：`auc_dis` 的本文−外部差值 ≥ `auc_old` 的差值
        ⇒ 先验的增量恰好出现在"都是新东西"这一最难的子问题上

=============================================================================
对照的公平性说明
=============================================================================
k=0 时外部方法没有 T 的任何样本，只能用异常分。这不是我们故意削弱它们——
这正是"零样本"的定义。我们把它们的 `auc_dis` 也**如实计算**：
若它们真的能区分 T 与 D，数字会显示出来。
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
def score_new(net, W, X, new_row):
    z = F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    lg = (z @ Wt.T).cpu().numpy()[:, :W.shape[0]]
    return lg[:, new_row], lg.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ndis", type=int, default=3, help="每个 target 用几个 distractor")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]

    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    ARMS = ["ridge", "rf", "svm-rbf", "knn-cos", "centroid-cos"]
    res = {"acc": {t: {a: {} for a in ARMS} for t in tags},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "ndis": args.ndis, "lams": lams}}
    print(f"[exp50] device={DEV} shots={tags} ndis={args.ndis} seeds={args.seeds}",
          flush=True)

    for tgt in eids:
        cands = [e for e in eids if e != tgt]
        rng_d = np.random.default_rng(20250101 + eids.index(tgt))
        dis_list = [str(x) for x in
                    rng_d.choice(cands, min(args.ndis, len(cands)), replace=False)]

        for dis in dis_list:
            others = [e for e in eids if e not in (tgt, dis)]
            n_old_ent = len(others)
            NEW = n_old_ent + len(mids)          # T 的行号
            S = np.asarray(T_all)[[ki[e] for e in others]
                                  + [ki[f"M_{m}"] for m in mids] + [ki[tgt]]]
            Xt_ = np.asarray(ent[tgt], float)
            Xd_ = np.asarray(ent[dis], float)

            for k in shots:
                tag = "full" if k is None else str(k)
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed * 7919 + 13)
                    perm = rng.permutation(len(Xt_))
                    n_te = int(min(max(5, round(0.3 * len(Xt_))), len(Xt_) - 1))
                    te_ix, pool_ix = perm[:n_te], perm[n_te:]
                    tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                    Xte_new = Xt_[te_ix]
                    Xte_dis = Xd_[rng.permutation(len(Xd_))[:len(te_ix)]]

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
                        ys.append(np.full(ntr, n_old_ent + j))
                        refs.append(Xo[pp[ntr:]])
                    nrows = n_old_ent + len(mids)
                    if len(tr_e_ix):
                        parts.append(Xt_[tr_e_ix])
                        ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                    Xtr = np.concatenate(parts).astype(np.float32)
                    ytr = np.concatenate(ys).astype(np.int64)
                    Xte_old = np.concatenate(refs).astype(np.float32)

                    # ---------------- 外部基线（同一协议）----------------
                    sc = StandardScaler().fit(Xtr)
                    Xtr_s = sc.transform(Xtr)
                    Xn_s = sc.transform(Xte_new)
                    Xo_s = sc.transform(Xte_old)
                    Xd_s = sc.transform(Xte_dis)
                    has_new = len(tr_e_ix) > 0

                    rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                                n_jobs=-1).fit(Xtr_s, ytr)
                    sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                             random_state=0).fit(Xtr_s, ytr)
                    cn_rf, cn_sv = list(rf.classes_), list(sv.classes_)
                    Pn, Po, Pd = (rf.predict_proba(Xn_s), rf.predict_proba(Xo_s),
                                  rf.predict_proba(Xd_s))
                    Qn, Qo, Qd = (sv.predict_proba(Xn_s), sv.predict_proba(Xo_s),
                                  sv.predict_proba(Xd_s))
                    if has_new:
                        ir, isv = cn_rf.index(NEW), cn_sv.index(NEW)
                        s_rf = (Pn[:, ir], Po[:, ir], Pd[:, ir])
                        s_sv = (Qn[:, isv], Qo[:, isv], Qd[:, isv])
                        t_rf = float((rf.predict(Xn_s) == NEW).mean())
                        t_sv = float((sv.predict(Xn_s) == NEW).mean())
                    else:
                        s_rf = (1 - Pn.max(1), 1 - Po.max(1), 1 - Pd.max(1))
                        s_sv = (1 - Qn.max(1), 1 - Qo.max(1), 1 - Qd.max(1))
                        t_rf = t_sv = 0.0
                    Cn, Co, Cd = (cos_to(Xn_s, Xtr_s), cos_to(Xo_s, Xtr_s),
                                  cos_to(Xd_s, Xtr_s))
                    if has_new:
                        s_kn = (Cn[:, ytr == NEW].max(1), Co[:, ytr == NEW].max(1),
                                Cd[:, ytr == NEW].max(1))
                        t_kn = float((ytr[Cn.argmax(1)] == NEW).mean())
                    else:
                        s_kn = (1 - Cn.max(1), 1 - Co.max(1), 1 - Cd.max(1))
                        t_kn = 0.0
                    cj = np.unique(ytr)
                    cent = np.stack([Xtr_s[ytr == j].mean(0) for j in cj])
                    Dn, Do, Dd = (cos_to(Xn_s, cent), cos_to(Xo_s, cent),
                                  cos_to(Xd_s, cent))
                    if has_new:
                        ic = list(cj).index(NEW)
                        s_ce = (Dn[:, ic], Do[:, ic], Dd[:, ic])
                        t_ce = float((cj[Dn.argmax(1)] == NEW).mean())
                    else:
                        s_ce = (1 - Dn.max(1), 1 - Do.max(1), 1 - Dd.max(1))
                        t_ce = 0.0

                    for nm, (sn, so, sd), tt in (
                            ("rf", s_rf, t_rf), ("svm-rbf", s_sv, t_sv),
                            ("knn-cos", s_kn, t_kn), ("centroid-cos", s_ce, t_ce)):
                        res["acc"][tag].setdefault(nm, {}).setdefault(
                            f"{tgt}|{dis}", []).append(dict(
                            top1=round(float(tt), 4),
                            auc_old=round(float(auc_of(sn, so)), 4),
                            auc_dis=round(float(auc_of(sn, sd)), 4),
                            auc_all=round(float(auc_of(sn, np.concatenate([so, sd]))), 4)))

                    # ---------------- 本文 ----------------
                    rng_v = np.random.default_rng(seed * 31 + 7)
                    val_mask = np.zeros(len(Xtr), bool)
                    rows_old = sorted({int(y) for y in ytr})
                    for j in rows_old:
                        if j == NEW and len(tr_e_ix) == 0:
                            continue
                        ix = rng_v.permutation(np.where(ytr == j)[0])
                        nv = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                        val_mask[ix[:nv]] = True
                    Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                    Xval, yval = Xtr[val_mask], ytr[val_mask]
                    val_rows = sorted({int(y) for y in yval})

                    net, _ = stage1(Xfit, yfit, dim, nrows, args.steps,
                                    seed * 1000 + eids.index(tgt), S=S)
                    Zfit = embed(net, Xfit)
                    Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                    S_old, mu_S = S[rows_old], S[rows_old].mean(0, keepdims=True)
                    lam_b, bv = lams[0], -1.0
                    for lam in lams:
                        Wc = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam)
                        v = macro_recall(net, Wc, Xval, yval, val_rows)
                        if v > bv:
                            bv, lam_b = v, lam
                    W_chem = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)
                    sn, pred_n = score_new(net, W_chem, Xte_new, NEW)
                    so, _ = score_new(net, W_chem, Xte_old, NEW)
                    sd, _ = score_new(net, W_chem, Xte_dis, NEW)
                    res["acc"][tag].setdefault("ridge", {}).setdefault(
                        f"{tgt}|{dis}", []).append(dict(
                        top1=round(float((pred_n == NEW).mean()), 4),
                        auc_old=round(float(auc_of(sn, so)), 4),
                        auc_dis=round(float(auc_of(sn, sd)), 4),
                        auc_all=round(float(auc_of(sn, np.concatenate([so, sd]))), 4)))

        line = []
        for t in tags:
            s = " ".join(
                f"{a}={np.nanmean([d['auc_dis'] for d in res['acc'][t][a].get(f'{tgt}|{dis_list[0]}', [])]):.2f}"
                for a in ARMS if res["acc"][t][a].get(f"{tgt}|{dis_list[0]}"))
            line.append(f"{t}: {s}")
        print(f"[{tgt:>7}] " + " | ".join(line), flush=True)

    with open(os.path.join(OUT, "exp50_distractor.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp50] 已写出 out/exp50_distractor.json")
    write_report(res, ARMS, tags)


def write_report(res, ARMS, tags):
    try:
        from scipy.stats import ttest_1samp, ttest_rel
        HAS = True
    except Exception:
        HAS = False

    def M(t, a, met):
        """返回 dict: pair -> 该 pair 的 seeds 均值"""
        out = {}
        for p, lst in res["acc"][t].get(a, {}).items():
            v = [d[met] for d in lst if d[met] is not None]
            if v:
                out[p] = float(np.mean(v))
        return out

    L = []
    w = L.append
    w("# exp50 —— 定向筛查 vs 通用异常检测（双重 held-out：target + novel distractor）\n")
    w("`auc_dis` = AUC(目标 T vs **另一个同样不在库里的新化合物 D**)。\n")
    w("★ 这是区分「定向筛查」与「通用 novelty detection」的判别性指标：\n")
    w("  通用异常检测在原理上**无法**区分两个都新的化合物 ⇒ auc_dis 应 ≈ 0.5。\n")

    for met in ("auc_dis", "auc_old", "auc_all", "top1"):
        w(f"\n## {met}\n")
        w("| k | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
        w("|---|" + "---|" * len(ARMS))
        for t in tags:
            row = []
            for a in ARMS:
                d = M(t, a, met)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w(f"| {t} | " + " | ".join(row) + " |")

    w("\n## 判据（k=0，主战场）\n")
    for t in tags:
        rr = M(t, "ridge", "auc_dis")
        w(f"- **k={t}**")
        for a in ARMS:
            d = M(t, a, "auc_dis")
            if not d:
                continue
            v = np.asarray(list(d.values()), float)
            msg = ""
            if HAS and len(v) > 1:
                msg = (f"，vs 0.5: p={ttest_1samp(v, 0.5).pvalue:.4f}"
                       f"（{'显著>' if np.mean(v) > 0.5 and ttest_1samp(v, 0.5).pvalue < 0.05 else '不显著'}）")
            w(f"   - `{a}` auc_dis = {np.mean(v):.3f}{msg}")
        # 本文 vs 每个外部方法的配对差
        if HAS and rr:
            for a in ("rf", "svm-rbf", "knn-cos", "centroid-cos"):
                d = M(t, a, "auc_dis")
                common = sorted(set(rr) & set(d))
                if len(common) > 2:
                    x = np.asarray([d[p] for p in common])
                    y = np.asarray([rr[p] for p in common])
                    w(f"   - ridge − {a}: Δ={np.mean(y - x):+.3f}, "
                      f"p={ttest_rel(y, x).pvalue:.4f} (n={len(common)})")

    with open(os.path.join(OUT, "exp50_p6_distractor.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp50] 已写出 out/exp50_p6_distractor.md")


if __name__ == "__main__":
    main()
