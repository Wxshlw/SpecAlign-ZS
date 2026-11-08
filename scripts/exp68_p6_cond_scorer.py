"""exp68 —— 化学条件化非线性打分器（SpecAlign-ZS v2）

病根（exp66 E 定位）：`score = z · w_T` 是**双线性**相似度 = 单个线性方向，
而「检出」需要非线性边界 ⇒ 同空间下 rf_z 0.896 vs zs 0.730 (p=0.0029)。

修复：不学 w_T，改学打分函数 g(z, S_T) → 「这条谱含不含 S_T 描述的成分」。
训练三元组（只用老类，T 完全不参与）：
    正例 (老类 j 掺混谱, S_j)          → 1
    负例1 (纯基质谱,   S_j)            → 0   没掺
    负例2 (老类 j 掺混谱, S_k), k≠j    → 0   掺的是别的东西
推断：(未知谱, S_T) —— T 一条谱都没见过，S_T 来自 SMILES ⇒ k=0 可用。
负例2 同时靶向 exp62 发现的「化学近邻混淆」。

臂：
    zs      现状线性头 z·w_T
    uad     通用检测器（条件恒为 S_T 但只在「掺/未掺」上训练，无化学区分）
    cond    化学条件化 MLP：g([z, S_T])
    cond-k  cond + k 条 T 真实谱微调（k>0）
    rf/svm  外部基线
协议：按样品 id 分组（exp67 定稿）
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
import p6_data as P
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, pauc_std,
                                         macro_recall, EID2MID)

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
SID2EID = {"D7w": "A01", "D7p": "A02", "D7d": "A03", "D8": "A04",
           "D1": "A05", "D2": "A06", "D3": "A07"}


class CondScorer(nn.Module):
    """g([z, S]) → 标量 logit。fiLM 变体：化学向量调制谱嵌入。"""

    def __init__(self, dz, ds, hid=256, film=True):
        super().__init__()
        self.film = film
        self.gam = nn.Linear(ds, dz)
        self.bet = nn.Linear(ds, dz)
        self.fc = nn.Sequential(nn.Linear(dz, hid), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(hid, 1))
        self.raw = nn.Sequential(nn.Linear(dz + ds, hid), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(hid, 1))

    def forward(self, z, s):
        if self.film:
            z = z * (1 + self.gam(s)) + self.bet(s)
            return self.fc(z).squeeze(-1)
        return self.raw(torch.cat([z, s], -1)).squeeze(-1)


def train_scorer(Ztr, Str, ytr, dz, ds, seed, film=True,
                 steps=800, lr=1e-3, bs=64):
    torch.manual_seed(seed)
    net = CondScorer(dz, ds, film=film).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=DEV)
    St = torch.tensor(Str, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    n = len(Zt)
    net.train()
    for it in range(steps):
        ix = torch.randint(0, n, (min(bs, n),), device=DEV)
        loss = F.binary_cross_entropy_with_logits(
            net(Zt[ix], St[ix]), yt[ix])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def score_of(net, Z, S_vec):
    Zt = torch.tensor(Z, dtype=torch.float32, device=DEV)
    St = torch.tensor(np.repeat(S_vec[None], len(Z), 0),
                      dtype=torch.float32, device=DEV)
    return net(Zt, St).cpu().numpy()


def load_groups():
    sets = P.load_all(verbose=False)
    grid = G1._grid()
    out = {}
    for sid, D in sets.items():
        if sid not in SID2EID:
            continue
        Xg = G1._resample(np.asarray(D["X"], float),
                          np.asarray(D["axis"], float), grid)
        Xp = G1._prep(Xg)
        y = np.asarray(D["y"], float)
        g = D.get("groups", np.arange(len(Xp)))
        g = np.asarray(g, dtype=object)
        m = y > 0
        ug = {v: i for i, v in enumerate(sorted(set(g[m].tolist())))}
        out[SID2EID[sid]] = dict(X=Xp[m], y=y[m],
                                 gid=np.array([ug[v] for v in g[m]]))
    return out


def split_by_group(X, gid, rng, te_frac=0.3):
    ug = np.unique(gid)
    ugp = rng.permutation(ug)
    nte = max(1, int(round(te_frac * len(ug))))
    te_g = set(ugp[:nte].tolist())
    te = np.where(np.isin(gid, list(te_g)))[0]
    tr = np.where(~np.isin(gid, list(te_g)))[0]
    return te, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--scorer-steps", type=int, default=800)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--film", type=int, default=1)
    ap.add_argument("--out", default="exp68_cond.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    dz = T_all.shape[1]
    ds = dz
    res = {"M": {}}
    print(f"[exp68] device={DEV} 实体={eids} shots={shots} "
          f"seeds={args.seeds} film={bool(args.film)}", flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]
        S_h = T_all[ki[e_hold]]
        S_all = T_all[[ki[e] for e in others]
                      + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = len(others) + len(mids)
        S_old_map = {e: T_all[ki[e]] for e in others}

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                te_ix, pool_ix = split_by_group(Xe, gid_e, rng)
                Xte_new = Xe[te_ix]
                tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                # ---- 老类训练/参考 + 基质（按组划分）----
                old_tr, old_ref, mid_ref = {}, {}, {}
                parts, ys = [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    gid_o = GR[e]["gid"] if e in GR else np.arange(len(Xo))
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    old_tr[e] = Xo[pp[:ntr]]; old_ref[e] = Xo[pp[ntr:]]
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    mid_ref[m] = Xo[pp[ntr:]]
                nrows = NEW
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(list(old_ref.values())
                                         + list(mid_ref.values())).astype(
                    np.float32)

                # ---- stage1 嵌入 ----
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in sorted({int(y) for y in ytr}):
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    nv = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:nv]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                net, _ = stage1(Xfit, yfit, dz, nrows, args.steps,
                                seed * 1000 + eids.index(e_hold), S=S_all)
                Zfit = embed(net, Xfit)
                rows = sorted({int(y) for y in yfit})
                Zc = np.stack([Zfit[yfit == j].mean(0) for j in rows])
                S_o, mu_S = S_all[rows], S_all[rows].mean(0, keepdims=True)
                lam_b, bv = lams[0], -1.0
                for lam in lams:
                    Wc = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam)
                    v = macro_recall(net, Wc, Xval, yval, rows)
                    if v > bv:
                        bv, lam_b = v, lam
                W = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam_b)
                Zn = embed(net, Xte_new)

                # ---- 化学条件化打分器训练集（只用老类）----
                # 批量 embed：把老类掺混谱与基质纯谱各拼成一批，避免逐类调用
                Zp, Sp, yp = [], [], []
                blk_e = [np.asarray(old_tr[e], float) for e in others]
                off = np.cumsum([0] + [len(b) for b in blk_e])
                Zold = embed(net, np.concatenate(blk_e).astype(np.float32))
                for i, e in enumerate(others):    # 正例：配自己的化学向量
                    Zo = Zold[off[i]:off[i + 1]]
                    Zp.append(Zo)
                    Sp.append(np.repeat(S_old_map[e][None], len(Zo), 0))
                    yp.append(np.ones(len(Zo)))
                Sstack = np.stack([S_old_map[e] for e in others])
                for i, e in enumerate(others):    # 负例2：配他人化学向量
                    Zo = Zold[off[i]:off[i + 1]]
                    msk = np.array([o != e for o in others])
                    cand = Sstack[msk]
                    Zp.append(Zo)
                    Sp.append(cand[rng.integers(0, len(cand), len(Zo))])
                    yp.append(np.zeros(len(Zo)))
                blk_m = [np.asarray(bg[m], float)[
                    rng.permutation(len(bg[m]))[:max(2, int(0.7 * len(bg[m])))]]
                    for m in mids]
                offm = np.cumsum([0] + [len(b) for b in blk_m])
                Zmid = embed(net, np.concatenate(blk_m).astype(np.float32))
                for i in range(len(mids)):        # 负例1：纯基质谱 × 随机老类向量
                    Zm = Zmid[offm[i]:offm[i + 1]]
                    Zp.append(Zm)
                    Sp.append(Sstack[rng.integers(0, len(Sstack), len(Zm))])
                    yp.append(np.zeros(len(Zm)))
                Zp = np.concatenate(Zp).astype(np.float32)
                Sp = np.concatenate(Sp).astype(np.float32)
                yp = np.concatenate(yp).astype(np.float32)

                sc_net = train_scorer(Zp, Sp, yp, dz, ds,
                                      seed * 17 + 3,
                                      film=bool(args.film),
                                      steps=args.scorer_steps)
                if len(tr_e_ix):                       # k>0：用 T 的真实谱微调
                    Zt_k = embed(net, Xe[tr_e_ix])
                    Zf = np.concatenate([Zt_k, Zp]).astype(np.float32)
                    Sf = np.concatenate([
                        np.repeat(S_h[None], len(Zt_k), 0), Sp]).astype(
                        np.float32)
                    yf = np.concatenate([np.ones(len(Zt_k)), yp]).astype(
                        np.float32)
                    sc_ft = train_scorer(Zf, Sf, yf, dz, ds, seed * 17 + 3,
                                         film=bool(args.film),
                                         steps=max(150, args.scorer_steps // 3),
                                         lr=3e-4)
                else:
                    sc_ft = None

                # ---- 外部基线 ----
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xn_s = sc.transform(Xte_new)
                rf = RandomForestClassifier(300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                         random_state=0).fit(Xtr_s, ytr)

                for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                       ("old-heldout", Xte_old)):
                    if len(Xn_neg) < 5:
                        continue
                    Zo = embed(net, Xn_neg)
                    Xo_s = sc.transform(Xn_neg)
                    if len(tr_e_ix):
                        i_rf = list(rf.classes_).index(NEW)
                        i_sv = list(sv.classes_).index(NEW)
                        base = (("rf", rf.predict_proba(Xn_s)[:, i_rf],
                                 rf.predict_proba(Xo_s)[:, i_rf]),
                                ("svm", sv.predict_proba(Xn_s)[:, i_sv],
                                 sv.predict_proba(Xo_s)[:, i_sv]))
                    else:
                        base = (("rf", 1 - rf.predict_proba(Xn_s).max(1),
                                 1 - rf.predict_proba(Xo_s).max(1)),
                                ("svm", 1 - sv.predict_proba(Xn_s).max(1),
                                 1 - sv.predict_proba(Xo_s).max(1)))
                    arms = {nm: (sn, so) for nm, sn, so in base}
                    arms["zs"] = (Zn @ W[NEW], Zo @ W[NEW])
                    arms["cond"] = (score_of(sc_net, Zn, S_h),
                                    score_of(sc_net, Zo, S_h))
                    if sc_ft is not None:
                        arms["cond-k"] = (score_of(sc_ft, Zn, S_h),
                                          score_of(sc_ft, Zo, S_h))
                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=round(auc_of(sn, so), 4),
                                 tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                 tpr20=round(tpr_at_fpr(sn, so, 0.20), 4),
                                 pauc10=round(pauc_std(sn, so, 0.10), 4)))
        line = []
        for n_name in ("same-matrix", "old-heldout"):
            for arm in ("zs", "cond", "rf"):
                d = res["M"].get(tag, {}).get(n_name, {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{n_name[:4]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp68] 已写出 out/{args.out}")
    write_report(res, eids, args.out)


def write_report(res, eids, out_name):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp68 —— 化学条件化非线性打分器（SpecAlign-ZS v2）\n")
    w("`cond` = g([z, S_T])：只用老类训练（正例 老类掺混谱×自身化学向量；"
      "负例1 纯基质谱×随机老类向量；负例2 老类掺混谱×**他人**化学向量），"
      "推断时配 S_T ⇒ **k=0 无需 T 的任何谱**。\n")
    w("协议：按样品 id 分组划分（exp67 定稿）。\n")

    def agg(tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(n_name, {})
                  .get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    ARMS = ["zs", "cond", "cond-k", "rf", "svm"]
    for met in ("auc", "tpr5", "tpr20"):
        w(f"\n## {met}（macro over {len(eids)} 实体）\n")
        w("| 口径 | k | " + " | ".join(f"`{a}`" for a in ARMS if a) + " |")
        w("|---|---|" + "---|" * len(ARMS))
        for n_name in ("same-matrix", "old-heldout"):
            for tag in ("0", "5"):
                d = res["M"].get(tag, {}).get(n_name, {})
                if not d:
                    continue
                cells = []
                for a in ARMS:
                    v = agg(tag, n_name, a, met)
                    cells.append(f"{np.mean(list(v.values())):.3f}"
                                 if v else "—")
                w(f"| {n_name} | {tag} | " + " | ".join(cells) + " |")

    w("\n## 判据（配对 t，n=实体数）\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for met in ("auc", "tpr5"):
                z = agg(tag, n_name, "zs", met)
                c = agg(tag, n_name, "cond", met)
                r = agg(tag, n_name, "rf", met)
                ck = agg(tag, n_name, "cond-k", met)
                com = sorted(set(z) & set(c) & set(r))
                if len(com) < 3:
                    continue
                xz = np.array([z[e] for e in com])
                xc = np.array([c[e] for e in com])
                xr = np.array([r[e] for e in com])
                msg = (f"- **{n_name} k={tag} {met}**: zs {xz.mean():.3f} → "
                       f"cond {xc.mean():.3f} (Δvs zs={np.mean(xc-xz):+.3f}, "
                       f"p={ttest_rel(xc,xz).pvalue:.4g}) | rf {xr.mean():.3f} "
                       f"(Δcond-rf={np.mean(xc-xr):+.3f}, "
                       f"p={ttest_rel(xc,xr).pvalue:.4g})")
                if ck:
                    xk = np.array([ck[e] for e in com if e in ck])
                    xr2 = np.array([r[e] for e in com if e in ck])
                    if len(xk) >= 3:
                        msg += (f" | cond-k {xk.mean():.3f} "
                                f"(Δvs rf={np.mean(xk-xr2):+.3f}, "
                                f"p={ttest_rel(xk, xr2).pvalue:.4g})")
                w(msg + f" (n={len(com)})")

    w("\n## 分实体（same-matrix, auc）\n")
    w("| 实体 | " + " | ".join(f"{a}" for a in ARMS) + " |")
    w("|---|" + "---|" * len(ARMS))
    for e in eids:
        cells = []
        for a in ARMS:
            v = agg("5", "same-matrix", a, "auc")
            cells.append(f"{v[e]:.3f}" if e in v else "—")
        w(f"| {e} | " + " | ".join(cells) + " |")

    with open(os.path.join(OUT, "exp68_p6_cond_scorer.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
