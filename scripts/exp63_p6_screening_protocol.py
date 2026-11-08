"""exp63 —— 「筛查」口径校正：负样本到底该是什么？

=============================================================================
exp62 暴露的更深问题（比阈值更根本）
=============================================================================
exp49/52/53 的 AUC 负样本 = **其余所有实体的 held-out 谱**（含其他掺杂物 +
基质）。这个任务的实际语义是「**多类判别**：这张谱是 T 还是别的掺杂物」，
而**不是**筛查。

筛查（screening）的真实语义是 Schymanski 的「检出」：
    「**这批基质里有没有掺东西**」 ⇒ 负样本 = **同基质的纯谱**。

把两者混为一谈，会让零样本方法被结构性惩罚：
  - 多类判别口径下，T 的最近邻实体（如 A01 的 A02，ECFP Tanimoto 0.815）
    必然占据高分负样本 ⇒ exp62 Stage A 实测富集 5.0×；
  - 筛查口径下不需要区分 A01 与 A02，只需区分「掺了」与「纯肉桂」。

★ 这不是挑数据：同一套模型、同一套训练、同一套阈值规则，
  只改**负样本的定义**，看结论是否由口径驱动。三种口径全部报告，
  不做选择性呈现。

=============================================================================
设计
=============================================================================
负样本口径（同一折内同时计算，无额外算力）：
  N1 `same-matrix`   T 所在基质的纯谱 held-out（真筛查语义，主口径）
  N2 `all-matrix`    全部 6 个基质纯谱 held-out（通用筛查）
  N3 `old-heldout`   其余实体 + 基质 held-out（exp52/53 现状，对照）

正样本：T 的 held-out 30%（k=0，T 的光谱不进训练集）
得分：raw / margin-hard / margin-lse（exp62 结论：margin 最优且合法）
对照：rf / svm-rbf / centroid-cos（1−maxprob，同口径）+ rand（零化学信息）

指标：auc / pauc10 / tpr5 / tpr10 / tpr20，整体 + 浓度分箱

预注册判据
  H63a：N1 口径下本文（margin-lse）auc 显著高于 N3 口径
        ⇒ 确认 exp52 的低分由「多类判别」这一口径负担造成
  H63b：N1 口径下本文 tpr5 显著高于 raw（复现 exp62 的 margin 增益）
  H63c：N1 口径下本文 tpr5 不低于 rf（公平口径下的筛查能力）
  H63d：**浓度检测限**：N1 口径下 tpr5 首次 ≥0.80 的浓度箱 c\*，并与 N3 对比
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
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

EID2MID = {"A01": "cinnamon", "A02": "cinnamon", "A03": "cinnamon",
           "A04": "turmeric", "A05": "oliveoil", "A06": "coffeeR",
           "A07": "juice", "HSYRUP": "honey"}


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


def tpr_at_fpr(pos, neg, alpha):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos > np.quantile(neg, 1.0 - alpha)).mean())


def pauc_std(pos, neg, alpha=0.10, ngrid=201):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    fpr = np.linspace(0.0, alpha, ngrid)
    tpr = np.array([(pos > t).mean() for t in np.quantile(neg, 1.0 - fpr)])
    area = TRAPZ(tpr, fpr) / alpha
    return float(0.5 * (1.0 + (area - alpha / 2) / (1.0 - alpha / 2)))


BINS = [(2.0, "<=2"), (5.0, "3-5"), (10.0, "6-10"), (20.0, "11-20"), (1e9, ">20")]


def bin_of(c):
    for hi, lab in BINS:
        if c <= hi:
            return lab
    return ">20"


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    rs = [float((pred[yv == j] == i).mean()) for i, j in enumerate(rows)
          if (yv == j).sum()]
    return float(np.mean(rs)) if rs else 0.0


def pack(sn, so, conc_te):
    """整体 + 浓度分箱。阈值取负样本自身分位 ⇒ 名义 FPR 精确。"""
    def one(p, q):
        return dict(auc=round(auc_of(p, q), 4),
                    pauc10=round(pauc_std(p, q, 0.10), 4),
                    tpr5=round(tpr_at_fpr(p, q, 0.05), 4),
                    tpr10=round(tpr_at_fpr(p, q, 0.10), 4),
                    tpr20=round(tpr_at_fpr(p, q, 0.20), 4),
                    n=int(len(p)))
    out = {"all": one(sn, so)}
    for _, lab in BINS:
        m = np.asarray([bin_of(c) == lab for c in conc_te])
        if m.sum() < 3:
            continue
        out[lab] = one(sn[m], so)
    return out


NEG = ["same-matrix", "all-matrix", "old-heldout"]
ARMS = ["raw", "margin-hard", "margin-lse", "rand", "rf", "svm-rbf",
        "centroid-cos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--out", default="exp63_screen_v2.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s.strip() == "full" else int(s)
             for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]

    ent, conc, bg, _ = G1.load_g1_conc()
    eids = [e for e in sorted(ent) if conc.get(e) is not None
            and e in EID2MID]
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    res = {t: {n: {a: {} for a in ARMS} for n in NEG} for t in tags}
    res["meta"] = dict(seeds=args.seeds, eids=eids, arms=ARMS, neg=NEG,
                       shots=tags)
    print(f"[exp63] device={DEV} 实体={eids} shots={tags} seeds={args.seeds} "
          f"基质={mids}", flush=True)
    print("        基质样本数: " + ", ".join(f"{m}={len(bg[m])}" for m in mids),
          flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        NEW = len(others) + len(mids)
        S = T_all[[ki[e] for e in others]
                  + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        Xe = np.asarray(ent[e_hold], float)
        ce = np.asarray(conc[e_hold], float)
        mid_h = EID2MID[e_hold]

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed * 7919 + 13)
            perm = rng.permutation(len(Xe))
            n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
            te_ix, pool_ix = perm[:n_te], perm[n_te:]
            Xte_new, conc_te = Xe[te_ix], ce[te_ix]

            for k in shots:
                tag = "full" if k is None else str(k)
                tr_e_ix = (pool_ix if k is None
                           else pool_ix[:min(k, len(pool_ix))])
                parts, ys, refs = [], [], []
                mid_ref = {}
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
                    refs.append(Xo[pp[ntr:]]); mid_ref[m] = Xo[pp[ntr:]]
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(refs).astype(np.float32)

                # —— 三种负样本口径 ——
                negs = {
                    "same-matrix": mid_ref[mid_h],
                    "all-matrix": np.concatenate([mid_ref[m] for m in mids]),
                    "old-heldout": Xte_old,
                }
                for n_name, Xn_neg in negs.items():
                    if len(Xn_neg) < 5:
                        continue

                    # ---------- 外部基线 ----------
                    sc = StandardScaler().fit(Xtr)
                    Xtr_s = sc.transform(Xtr)
                    Xn_s = sc.transform(Xte_new)
                    Xo_s = sc.transform(Xn_neg)
                    rf = RandomForestClassifier(n_estimators=300,
                                                random_state=0,
                                                n_jobs=-1).fit(Xtr_s, ytr)
                    sv = SVC(kernel="rbf", C=10, gamma="scale",
                             probability=True,
                             random_state=0).fit(Xtr_s, ytr)
                    cj = np.unique(ytr)
                    cent = np.stack([Xtr_s[ytr == j].mean(0) for j in cj])
                    Dn = nrm(Xn_s) @ nrm(cent).T
                    Do = nrm(Xo_s) @ nrm(cent).T
                    if len(tr_e_ix):
                        # k>0：T 已在标签集内 ⇒ 用 T 类的**有监督**概率
                        i_rf = list(rf.classes_).index(NEW)
                        i_sv = list(sv.classes_).index(NEW)
                        ic = list(cj).index(NEW)
                        trio = (
                            ("rf", (rf.predict_proba(Xn_s)[:, i_rf],
                                    rf.predict_proba(Xo_s)[:, i_rf])),
                            ("svm-rbf", (sv.predict_proba(Xn_s)[:, i_sv],
                                         sv.predict_proba(Xo_s)[:, i_sv])),
                            ("centroid-cos", (Dn[:, ic], Do[:, ic])))
                    else:
                        # k=0：T 不在标签集 ⇒ 只能用 1−maxprob 异常分
                        trio = (
                            ("rf", (1 - rf.predict_proba(Xn_s).max(1),
                                    1 - rf.predict_proba(Xo_s).max(1))),
                            ("svm-rbf", (1 - sv.predict_proba(Xn_s).max(1),
                                         1 - sv.predict_proba(Xo_s).max(1))),
                            ("centroid-cos", (1 - Dn.max(1), 1 - Do.max(1))))
                    for nm, (sn, so) in trio:
                        res[tag][n_name][nm].setdefault(e_hold, []).append(
                            pack(sn, so, conc_te))

                # ---------- 本文 ----------
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in sorted({int(y) for y in ytr}):
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
                W = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)
                rng_r = np.random.default_rng(seed * 977 + 5)
                Wr = nrm(rng_r.normal(size=W.shape).astype(np.float32))

                Zn = embed(net, Xte_new)
                for n_name, Xn_neg in negs.items():
                    if len(Xn_neg) < 5:
                        continue
                    Zo = embed(net, Xn_neg)
                    for nm, Wx in (("ridge", W), ("rand", Wr)):
                        Ln = Zn @ Wx.T
                        Lo = Zo @ Wx.T
                        mx_n = Ln[:, :NEW].max(1)
                        mx_o = Lo[:, :NEW].max(1)
                        cand = {
                            "raw": (Ln[:, NEW], Lo[:, NEW]),
                            "margin-hard": (Ln[:, NEW] - mx_n,
                                            Lo[:, NEW] - mx_o),
                        }
                        if nm == "ridge":
                            cand["margin-lse"] = (
                                Ln[:, NEW] - (mx_n + args.tau * np.log(
                                    np.exp(Ln[:, :NEW] / args.tau
                                           - mx_n[:, None] / args.tau).sum(1))),
                                Lo[:, NEW] - (mx_o + args.tau * np.log(
                                    np.exp(Lo[:, :NEW] / args.tau
                                           - mx_o[:, None] / args.tau).sum(1))))
                        for sn_name, (sn, so) in cand.items():
                            key = sn_name if nm == "ridge" else "rand"
                            res[tag][n_name][key].setdefault(e_hold, []).append(
                                pack(sn, so, conc_te))

        for tag in tags:
            line = []
            for n_name in ("same-matrix", "old-heldout"):
                d = res[tag][n_name]["margin-lse"].get(e_hold, [])
                if d:
                    line.append(f"{n_name[:4]}="
                                f"{np.mean([x['all']['tpr5'] for x in d]):.3f}")
            print(f"[{e_hold:>7}] mid={mid_h:<10} k={tag} " + " | ".join(line),
                  flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp63] 已写出 out/{args.out}")
    write_report(res, eids, tags, args.out)


def write_report(res, eids, tags=None, out_name="exp63_screen_v2.json"):
    from scipy.stats import ttest_rel
    ORD = ["all", "<=2", "3-5", "6-10", "11-20", ">20"]
    SHOW = ["margin-lse", "raw", "rand", "rf", "svm-rbf", "centroid-cos"]
    if tags is None:
        tags = res["meta"]["shots"]

    def agg(tag, n_name, arm, met, seg="all"):
        per = {}
        for e in eids:
            vs = [d[seg][met]
                  for d in res[tag][n_name].get(arm, {}).get(e, [])
                  if seg in d]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    L = []
    w = L.append
    w("# exp63 —— 「筛查」口径校正 × 冷启动强度（k）\n")
    w("同一模型、同一训练、同一阈值规则，**只改负样本定义与 k**。全部组合报告。\n")
    w("- `same-matrix`：T 所在基质纯谱（**真筛查语义**：这批基质有没有掺东西）\n"
      "- `all-matrix`：全部 6 个基质纯谱（通用筛查）\n"
      "- `old-heldout`：其余实体 + 基质（exp49/52/53 现状，实为多类判别）\n")
    w("- **k** = T 的掺混谱进训练集的条数；k=0 为纯冷启动。\n"
      "  注意：k>0 时 rf/svm 已把 T 作为一个类（用 T 类概率），"
      "k=0 时 T 不在其标签集内（只能用 1−maxprob 异常分）。\n")
    w("`pAUC10` = McClish 标准化部分 AUC（FPR≤0.10，随机 0.5、完美 1.0）。\n")

    for tag in tags:
        for n_name in NEG:
            w(f"\n## k={tag} · 口径 `{n_name}`（macro over {len(eids)} 实体）\n")
            for met in ("auc", "pauc10", "tpr5", "tpr10", "tpr20"):
                w(f"\n### {met}（整体）\n")
                w("| 臂 | " + " | ".join(f"`{a}`" for a in SHOW) + " |")
                w("|---|" + "---|" * len(SHOW))
                row = []
                for a in SHOW:
                    d = agg(tag, n_name, a, met)
                    row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
                w("| 值 | " + " | ".join(row) + " |")

    # ---- k=0 vs k=5 的 gains（核心问题：垫底是否只属于 k=0）----
    w("\n## 核心对比：本文的落后是否只属于 k=0？\n")
    w("| 口径 | k | 本文 margin-lse | rf | svm-rbf | 差值(本文−rf) | 差值(本文−svm) |")
    w("|---|---|---|---|---|---|---|")
    for n_name in NEG:
        for tag in tags:
            o = agg(tag, n_name, "margin-lse", "tpr5")
            r = agg(tag, n_name, "rf", "tpr5")
            s = agg(tag, n_name, "svm-rbf", "tpr5")
            if not (o and r and s):
                continue
            mo, mr, ms = (np.mean(list(o.values())), np.mean(list(r.values())),
                          np.mean(list(s.values())))
            w(f"| {n_name} | {tag} | {mo:.3f} | {mr:.3f} | {ms:.3f} | "
              f"{mo - mr:+.3f} | {mo - ms:+.3f} |")

    w("\n### 配对检验（本文 margin-lse − rf，n=实体数）\n")
    for n_name in NEG:
        for tag in tags:
            o = agg(tag, n_name, "margin-lse", "tpr5")
            r = agg(tag, n_name, "rf", "tpr5")
            com = sorted(set(o) & set(r))
            if len(com) < 3:
                continue
            x = np.array([r[e] for e in com]); y = np.array([o[e] for e in com])
            w(f"- `{n_name}` k={tag}：rf {x.mean():.3f} vs 本文 {y.mean():.3f} "
              f"(Δ={np.mean(y - x):+.3f}, p={ttest_rel(y, x).pvalue:.3g}, "
              f"n={len(com)})")

    # 判据
    w("\n## 判据\n")
    tag0 = tags[0]
    a1 = agg(tag0, "same-matrix", "margin-lse", "auc")
    a3 = agg(tag0, "old-heldout", "margin-lse", "auc")
    com = sorted(set(a1) & set(a3))
    if len(com) >= 3:
        x = np.array([a3[e] for e in com]); y = np.array([a1[e] for e in com])
        w(f"\n- **H63a**（口径负担，k={tag0}）：margin-lse auc "
          f"`old-heldout` {x.mean():.3f} → `same-matrix` {y.mean():.3f} "
          f"(Δ={np.mean(y - x):+.3f}, p={ttest_rel(y, x).pvalue:.4g}, "
          f"n={len(com)}) ⇒ {'成立' if np.mean(y - x) > 0 else '不成立'}")

    for tag in tags:
        for n_name in NEG:
            w(f"\n### k={tag} · `{n_name}`")
            for met in ("auc", "pauc10", "tpr5", "tpr20"):
                a = agg(tag, n_name, "raw", met)
                b = agg(tag, n_name, "margin-lse", met)
                c = agg(tag, n_name, "rf", met)
                d = agg(tag, n_name, "rand", met)
                com = sorted(set(a) & set(b) & set(c))
                if len(com) < 3:
                    continue
                xa = np.array([a[e] for e in com])
                xb = np.array([b[e] for e in com])
                xc = np.array([c[e] for e in com])
                w(f"- **{met}**：raw {xa.mean():.3f} | margin-lse {xb.mean():.3f} "
                  f"(Δvs raw={np.mean(xb - xa):+.3f}, "
                  f"p={ttest_rel(xb, xa).pvalue:.3g}) "
                  f"| rf {xc.mean():.3f} (Δvs rf={np.mean(xb - xc):+.3f}, "
                  f"p={ttest_rel(xb, xc).pvalue:.3g})"
                  + (f" | rand {np.mean([d[e] for e in com]):.3f}" if d else ""))
            w("   - **浓度分层 tpr5**（margin-lse / raw / rf）：")
            for seg in ORD:
                b = agg(tag, n_name, "margin-lse", "tpr5", seg)
                a = agg(tag, n_name, "raw", "tpr5", seg)
                c = agg(tag, n_name, "rf", "tpr5", seg)
                if not b:
                    continue
                w(f"     - {seg}%: margin {np.mean(list(b.values())):.3f} | "
                  f"raw {np.mean(list(a.values())):.3f} | "
                  f"rf {np.mean(list(c.values())):.3f} | n={len(b)}")
            hit = [s for s in ORD
                   if agg(tag, n_name, "margin-lse", "tpr5", s)
                   and np.mean(list(agg(tag, n_name, "margin-lse",
                                        "tpr5", s).values())) >= 0.80]
            hit_rf = [s for s in ORD
                      if agg(tag, n_name, "rf", "tpr5", s)
                      and np.mean(list(agg(tag, n_name, "rf",
                                           "tpr5", s).values())) >= 0.80]
            w(f"   - **c\\***（tpr5 首次 ≥0.80）：margin-lse "
              f"{hit[0] if hit else '未达到'}% | "
              f"rf {hit_rf[0] if hit_rf else '未达到'}%")

    md = out_name.replace(".json", "") + ".md"
    with open(os.path.join(OUT, md), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[exp63] 已写出 out/{md}")


if __name__ == "__main__":
    main()
