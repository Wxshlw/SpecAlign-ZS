"""exp69 —— SpecAlign-ZS v3：margin 型双头（cond 检出 + zs 识别）

exp68 教训：cond 在「检出」上 +0.069（n.s.），但在「识别」上 −0.228 (p=0.0076)。
三个可疑原因，本实验一并修：
  (a) 训练正负比失衡（负例 = 2×正例 + 基质）⇒ 模型偏向判负；
  (b) 未做 margin 扣减（exp62 已证 margin 是唯一有效的非单调修复）；
  (c) 单头通吃两个任务——而双分离表明检出/识别需要**不同的头**。

v3 设计：
  识别头 = zs（化学外推，线性）+ margin      —— old-heldout
  检出头 = cond（化学条件化，非线性）+ margin  —— same-matrix
  融合臂 = z(cond) 与 z(zs) 标准化后加权
协议：按样品 id 分组（exp67 定稿）。
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
from exp68_p6_cond_scorer import CondScorer, load_groups, split_by_group

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7


def train_balanced(Ztr, Str, ytr, dz, ds, seed, film=True,
                   steps=800, lr=1e-3, bs=64):
    """按类别均衡采样（修复 exp68 的偏置判负）。"""
    torch.manual_seed(seed)
    net = CondScorer(dz, ds, film=film).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=DEV)
    St = torch.tensor(Str, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    ix1 = torch.where(yt > 0.5)[0]
    ix0 = torch.where(yt < 0.5)[0]
    net.train()
    for _ in range(steps):
        a = ix1[torch.randint(0, len(ix1), (bs // 2,), device=DEV)]
        b = ix0[torch.randint(0, len(ix0), (bs // 2,), device=DEV)]
        ix = torch.cat([a, b])
        loss = F.binary_cross_entropy_with_logits(net(Zt[ix], St[ix]), yt[ix])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def g_all(net, Z, Smat, chunk=4096):
    """返回 [n, n_cand] = g(z_i, S_j)。一次性展开成大批，避免逐候选调用。"""
    n, m = len(Z), len(Smat)
    Zb = np.repeat(np.asarray(Z, np.float32), m, 0)
    Sb = np.tile(np.asarray(Smat, np.float32), (n, 1))
    out = np.empty(n * m, np.float32)
    for st in range(0, n * m, chunk):
        Zt = torch.tensor(Zb[st:st + chunk], device=DEV)
        St = torch.tensor(Sb[st:st + chunk], device=DEV)
        out[st:st + chunk] = net(Zt, St).cpu().numpy()
    return out.reshape(n, m)


def zscore_pair(pos, neg):
    """把两组分数按负样本分布标准化后返回（用于融合）。"""
    mu, sd = float(np.mean(neg)), float(np.std(neg)) + 1e-8
    return (pos - mu) / sd, (neg - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--scorer-steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--film", type=int, default=1)
    ap.add_argument("--alpha", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--out", default="exp69_v3.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]
    alphas = [float(x) for x in args.alpha.split(",")]

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
    res = {"M": {}}
    print(f"[exp69] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
          flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]
        S_all = T_all[[ki[e] for e in others]
                      + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = len(others) + len(mids)
        S_old_map = {e: T_all[ki[e]] for e in others}
        # 候选化学向量（含 T），用于 margin
        cand_names = others + [e_hold]
        S_cand = T_all[[ki[e] for e in cand_names]]
        iT = len(others)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                te_ix, pool_ix = split_by_group(Xe, gid_e, rng)
                Xte_new = Xe[te_ix]
                tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                old_tr, old_ref, mid_ref = {}, {}, {}
                parts, ys = [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
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

                # ---- 平衡训练集 ----
                blk_e = [np.asarray(old_tr[e], float) for e in others]
                off = np.cumsum([0] + [len(b) for b in blk_e])
                Zold = embed(net, np.concatenate(blk_e).astype(np.float32))
                Sstack = np.stack([S_old_map[e] for e in others])
                Zp, Sp, yp = [], [], []
                for i, e in enumerate(others):       # 正例
                    Zo = Zold[off[i]:off[i + 1]]
                    Zp.append(Zo)
                    Sp.append(np.repeat(S_old_map[e][None], len(Zo), 0))
                    yp.append(np.ones(len(Zo)))
                for i, e in enumerate(others):       # 负例2：配他人化学向量
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
                for i in range(len(mids)):           # 负例1：纯基质谱
                    Zm = Zmid[offm[i]:offm[i + 1]]
                    Zp.append(Zm)
                    Sp.append(Sstack[rng.integers(0, len(Sstack), len(Zm))])
                    yp.append(np.zeros(len(Zm)))
                Zp = np.concatenate(Zp).astype(np.float32)
                Sp = np.concatenate(Sp).astype(np.float32)
                yp = np.concatenate(yp).astype(np.float32)
                sc_net = train_balanced(Zp, Sp, yp, dz, dz, seed * 17 + 3,
                                        film=bool(args.film),
                                        steps=args.scorer_steps)
                sc_ft = None
                if len(tr_e_ix):
                    Zt_k = embed(net, Xe[tr_e_ix])
                    Zf = np.concatenate([Zt_k, Zp]).astype(np.float32)
                    Sf = np.concatenate([
                        np.repeat(T_all[ki[e_hold]][None], len(Zt_k), 0),
                        Sp]).astype(np.float32)
                    yf = np.concatenate([np.ones(len(Zt_k)), yp]).astype(
                        np.float32)
                    sc_ft = train_balanced(Zf, Sf, yf, dz, dz, seed * 17 + 3,
                                           film=bool(args.film),
                                           steps=max(150,
                                                     args.scorer_steps // 3),
                                           lr=3e-4)

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
                        pn, po = (rf.predict_proba(Xn_s)[:, i_rf],
                                  rf.predict_proba(Xo_s)[:, i_rf])
                        qn, qo = (sv.predict_proba(Xn_s)[:, i_sv],
                                  sv.predict_proba(Xo_s)[:, i_sv])
                        # rf/svm 的 margin 版（同类内扣除最大竞争者）
                        Pn = rf.predict_proba(Xn_s); Po = rf.predict_proba(Xo_s)
                        rn = Pn[:, i_rf] - np.delete(Pn, i_rf, 1).max(1)
                        ro = Po[:, i_rf] - np.delete(Po, i_rf, 1).max(1)
                        Qn = sv.predict_proba(Xn_s); Qo = sv.predict_proba(Xo_s)
                        tn = Qn[:, i_sv] - np.delete(Qn, i_sv, 1).max(1)
                        to = Qo[:, i_sv] - np.delete(Qo, i_sv, 1).max(1)
                        base = {"rf": (pn, po), "svm": (qn, qo),
                                "rf-m": (rn, ro), "svm-m": (tn, to)}
                    else:
                        base = {"rf": (1 - rf.predict_proba(Xn_s).max(1),
                                       1 - rf.predict_proba(Xo_s).max(1)),
                                "svm": (1 - sv.predict_proba(Xn_s).max(1),
                                        1 - sv.predict_proba(Xo_s).max(1))}

                    # ---- zs：raw 与 margin ----
                    # W 行序 = others + mids + [T]；候选 = others + [T]
                    W_cand = np.concatenate([W[:len(others)],
                                             W[NEW:NEW + 1]], 0)
                    Ln = Zn @ W_cand.T
                    Lo = Zo @ W_cand.T
                    arms = dict(base)
                    arms["zs"] = (Ln[:, iT], Lo[:, iT])
                    arms["zs-m"] = (Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                                    Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                    # ---- cond：raw 与 margin ----
                    Gn = g_all(sc_net, Zn, S_cand)
                    Go = g_all(sc_net, Zo, S_cand)
                    arms["cond"] = (Gn[:, iT], Go[:, iT])
                    arms["cond-m"] = (Gn[:, iT] - np.delete(Gn, iT, 1).max(1),
                                      Go[:, iT] - np.delete(Go, iT, 1).max(1))
                    if sc_ft is not None:
                        Fn = g_all(sc_ft, Zn, S_cand)
                        Fo = g_all(sc_ft, Zo, S_cand)
                        arms["cond-k"] = (Fn[:, iT], Fo[:, iT])
                        arms["cond-k-m"] = (
                            Fn[:, iT] - np.delete(Fn, iT, 1).max(1),
                            Fo[:, iT] - np.delete(Fo, iT, 1).max(1))
                    # ---- 融合：α·z(cond-m) + (1−α)·z(zs-m) ----
                    for al in alphas:
                        if al in (0.0, 1.0):
                            continue
                        cn, co = zscore_pair(arms["cond-m"][0],
                                             arms["cond-m"][1])
                        zn, zo = zscore_pair(arms["zs-m"][0], arms["zs-m"][1])
                        arms[f"fuse{al}"] = (al * cn + (1 - al) * zn,
                                             al * co + (1 - al) * zo)
                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=round(auc_of(sn, so), 4),
                                 tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                 tpr20=round(tpr_at_fpr(sn, so, 0.20), 4)))
        line = []
        for n_name in ("same-matrix", "old-heldout"):
            for arm in ("zs-m", "cond-m", "fuse0.5", "rf"):
                d = res["M"].get(tag, {}).get(n_name, {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{n_name[:4]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp69] 已写出 out/{args.out}")
    write_report(res, eids, args.out, alphas)


def ki_names(e_hold, others, mids, ki, e):
    return ki[e]


def write_report(res, eids, out_name, alphas):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp69 —— SpecAlign-ZS v3：margin 型双头 + 融合\n")
    w("- `zs` 化学外推线性头；`zs-m` 其 margin 版 `s_T − max_{j≠T} s_j`\n"
      "- `cond` 化学条件化非线性 g([z,S_T])（**平衡采样**）；`cond-m` 其 margin 版\n"
      "- `fuse α` = α·z(cond-m) + (1−α)·z(zs-m)\n"
      "- `rf-m`/`svm-m` = rf/svm 的 margin 版（公平对比：都扣最大竞争者）\n")
    w("协议：按样品 id 分组划分。\n")

    def agg(tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(n_name, {})
                  .get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    for met in ("auc", "tpr5", "tpr20"):
        w(f"\n## {met}\n")
        arms = ["zs", "zs-m", "cond", "cond-m", "cond-k-m"] + \
               [f"fuse{a}" for a in alphas if a not in (0.0, 1.0)] + \
               ["rf", "rf-m", "svm"]
        w("| 口径 | k | " + " | ".join(f"`{a}`" for a in arms) + " |")
        w("|---|---|" + "---|" * len(arms))
        for n_name in ("same-matrix", "old-heldout"):
            for tag in ("0", "5"):
                if not res["M"].get(tag, {}).get(n_name):
                    continue
                cells = []
                for a in arms:
                    v = agg(tag, n_name, a, met)
                    cells.append(f"{np.mean(list(v.values())):.3f}"
                                 if v else "—")
                w(f"| {n_name} | {tag} | " + " | ".join(cells) + " |")

    w("\n## 判据（配对 t）\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for met in ("auc", "tpr5"):
                z = agg(tag, n_name, "zs", met)
                zm = agg(tag, n_name, "zs-m", met)
                cm = agg(tag, n_name, "cond-m", met)
                r = agg(tag, n_name, "rf", met)
                rm = agg(tag, n_name, "rf-m", met)
                fu = agg(tag, n_name, "fuse0.5", met)
                com = sorted(set(zm) & set(cm) & set(r))
                if len(com) < 3:
                    continue
                xz = np.array([z[e] for e in com])
                xzm = np.array([zm[e] for e in com])
                xcm = np.array([cm[e] for e in com])
                xr = np.array([r[e] for e in com])
                msg = (f"- **{n_name} k={tag} {met}**: zs {xz.mean():.3f} → "
                       f"zs-m {xzm.mean():.3f} (Δ={np.mean(xzm-xz):+.3f}, "
                       f"p={ttest_rel(xzm,xz).pvalue:.3g}) | cond-m "
                       f"{xcm.mean():.3f} | rf {xr.mean():.3f}")
                if rm:
                    xrm = np.array([rm[e] for e in com])
                    msg += f" | rf-m {xrm.mean():.3f}"
                    msg += (f" ⇒ cond-m vs rf-m "
                            f"{np.mean(xcm - np.array([rm[e] for e in com])):+.3f}"
                            f" (p={ttest_rel(xcm, np.array([rm[e] for e in com])).pvalue:.3g})")
                if fu:
                    xf = np.array([fu[e] for e in com])
                    msg += (f" | fuse.5 {xf.mean():.3f} vs rf "
                            f"{np.mean(xf - xr):+.3f} "
                            f"(p={ttest_rel(xf, xr).pvalue:.3g})")
                w(msg + f" (n={len(com)})")

    w("\n## 分实体（auc）\n")
    for n_name in ("same-matrix", "old-heldout"):
        w(f"\n### {n_name}\n")
        w("| 实体 | zs-m | cond-m | fuse.5 | rf | rf-m |")
        w("|---|---|---|---|---|---|")
        for e in eids:
            cells = []
            for a in ("zs-m", "cond-m", "fuse0.5", "rf", "rf-m"):
                for tag in ("5", "0"):
                    v = agg(tag, n_name, a, "auc")
                    if e in v:
                        cells.append(f"{v[e]:.3f}")
                        break
                else:
                    cells.append("—")
            w(f"| {e} | " + " | ".join(cells) + " |")

    with open(os.path.join(OUT, "exp69_p6_v3_margin.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
