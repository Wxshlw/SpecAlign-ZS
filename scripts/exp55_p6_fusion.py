"""exp55 —— 对齐方式的**集成融合**：从"持平略优"到"显著领先"的尝试

=============================================================================
动机（由 exp53 实测驱动，不是拍脑袋）
=============================================================================
exp53 的逐实体结果显示：**没有任何单一对齐方式在多数实体上夺冠**
（本文 ridge-c 只在 3/13 实体上最高；A04 是 ale-c 赢，A06 是 ridge-f 赢，
A05 是 conse 赢，COFR 是 eszsl 赢）。
这种"各赢各的"结构正是**集成**能发挥作用的前提。

同时 exp53 的结论是"持平略优"（Δ=+0.017, p=0.62 不显著），
本文需要一个**真正领先**的配置。集成是检验这一点的直接手段。

=============================================================================
融合规则（预注册，无超参调优）
=============================================================================
每种对齐 m 给出一个未见类得分 s_m(x)。不同方法的得分量纲完全不同
（ridge 是内积、devise/conse 是余弦、ale 是非归一化内积），
直接平均没有意义。统一用**老类验证集**做标准化：

    z_m(x) = (s_m(x) − μ_m) / σ_m ，  μ_m, σ_m 取自 Xval（训练时留出的老类）

融合 = 各成员 z_m 的**等权平均**。无权重、无超参、无需额外标签。
（这是标准的 score-level fusion / calibration，不是调出来的。）

成员组（预注册）：
    `ens-chem`  ridge + eszsl + devise + ale + conse（全在同一 chem 编码器上）
    `ens-all`   上述 + clip（跨编码器）
    `ens-top3`  ridge + conse + clip（exp53 中 macro 最高的三个）
    `ens-2`     ridge + clip
    `ens-rand`  ridge-rand + eszsl-crand（零化学信息集成，因果对照）

=============================================================================
预注册判据
=============================================================================
- H55a（主）：最好的集成组在 k=0 的 macro auc **显著 >** `ridge-c`（本文单对齐）
- H55b（主）：最好的集成组 **显著 >** exp53 中最好的单一基线（`conse-c` / `clip`）
        ⇒ 若能成立，才真正实现"领先"，而不只是"持平略优"
- H55c（稳健）：集成组的 **worst-case** auc ≥ `ridge-c` 的 worst-case
        （融合不能牺牲本文最看重的最差实体表现）
- H55d（因果）：`ens-rand` 显著低于真化学集成
- H55e（边界）：集成的 top1 是否仍 ≈0（若仍如此，再次确认识别不可解）
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp49_p6_clean_protocol as E49
import exp53_p6_zsl_baselines as E53

DEV = E49.DEV
OUT = E49.OUT
OLD_FRAC_TRAIN = E49.OLD_FRAC_TRAIN

SINGLE = ["ridge", "eszsl", "devise", "ale", "conse"]
GROUPS = {"ens-chem": SINGLE,
          "ens-all": SINGLE + ["clip"],
          "ens-top3": ["ridge", "conse", "clip"],
          "ens-2": ["ridge", "clip"],
          "ens-rand": ["ridge-rand", "eszsl-crand"]}
ARMS = SINGLE + ["clip", "ridge-rand", "eszsl-crand"] + list(GROUPS)


@torch.no_grad()
def embed(net, X):
    net.eval()
    return E53.embed(net, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]

    ent, bg = E53.load_data()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    res = {"acc": {t: {a: {} for a in ARMS} for t in tags},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "arms": ARMS, "groups": {k: v for k, v in GROUPS.items()}}}
    print(f"[exp55] device={DEV} shots={tags} seeds={args.seeds}", flush=True)

    for ei, e_hold in enumerate(eids):
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        S_rand = E53.rand_like(S, np.random.default_rng(1000 + ei))
        NEW = 18
        Xe = np.asarray(ent[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)   # 与 exp49/53/54 同切分
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new = Xe[te_ix]
                parts, ys, refs = [], [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC_TRAIN * len(Xo))))
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                    refs.append(Xo[pp[ntr:]])
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC_TRAIN * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    refs.append(Xo[pp[ntr:]])
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), 18)); nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(refs).astype(np.float32)

                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                rows_old = sorted({int(y) for y in ytr})
                for j in rows_old:
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    n_val = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:n_val]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                # ---------- 编码器 ----------
                net_c, _ = E49.stage1(Xfit, yfit, dim, nrows, args.steps,
                                      seed * 1000 + ei, S=S, mode="chem")
                Zf, Zv = embed(net_c, Xfit), embed(net_c, Xval)
                Zn, Zo = embed(net_c, Xte_new), embed(net_c, Xte_old)
                Z_old = np.stack([Zf[yfit == j].mean(0) for j in rows_old])
                net_p, W_clip = E53.train_clip(Xtr, ytr, S, dim,
                                               seed * 1000 + ei, steps=args.steps)
                Zn_p, Zo_p, Zv_p = embed(net_p, Xte_new), embed(net_p, Xte_old), embed(net_p, Xval)
                del net_p
                torch.cuda.empty_cache()

                # ---------- 各单一对齐的完整 19 类打分矩阵 ----------
                def pick_W(lam):
                    return (E53.head_ridge(S, rows_old, Z_old, lam),
                            E53.head_eszsl(S, rows_old, Zf, yfit, lam),
                            E53.head_ridge(S_rand, rows_old, Z_old, lam),
                            E53.head_eszsl(S_rand, rows_old, Zf, yfit, lam))
                best, bv = lams[0], -1.0
                for lam in lams:
                    Wr_, We_ = pick_W(lam)[:2]
                    v = max(E53.macro_val(Zv, Wr_, yval, val_rows),
                            E53.macro_val(Zv, We_, yval, val_rows))
                    if v > bv:
                        bv, best = v, lam
                Wr, We, Wr_rand, We_rand = pick_W(best)
                Wd = E53.train_devise(Zf, yfit, S, rows_old, S.shape[1], 1.0,
                                      steps=max(150, args.steps // 2))
                Wa = E53.train_ale(Zf, yfit, S, rows_old, 0.01,
                                   steps=max(150, args.steps // 2))
                bt, bv2 = 0.07, -1.0
                for tau in (0.01, 0.07, 0.5):
                    p = E53.conse_scores(Zv, S, rows_old, Z_old, tau)[:, val_rows].argmax(1)
                    rs = [float((p[yval == j] == i).mean())
                          for i, j in enumerate(val_rows) if (yval == j).sum()]
                    v = float(np.mean(rs)) if rs else 0.0
                    if v > bv2:
                        bv2, bt = v, tau

                def full(Zc):
                    return {"ridge": E53.lin_scores(Zc, Wr),
                            "eszsl": E53.lin_scores(Zc, We),
                            "devise": E53.lin_scores(Zc, Wd),
                            "ale": E53.lin_scores(Zc, Wa),
                            "ridge-rand": E53.lin_scores(Zc, Wr_rand),
                            "eszsl-crand": E53.lin_scores(Zc, We_rand),
                            "conse": E53.conse_scores(Zc, S, rows_old, Z_old, bt)}
                Mn, Mo, Mv = full(Zn), full(Zo), full(Zv)
                Mn["clip"] = E53.lin_scores(Zn_p, W_clip)
                Mo["clip"] = E53.lin_scores(Zo_p, W_clip)
                Mv["clip"] = E53.lin_scores(Zv_p, W_clip)

                # ---------- 标准化：用 Xval 上「未见类列」的 μ/σ，对整个矩阵做标量变换 ----------
                # 标量平移缩放不改变行内 argmax ⇒ 各方法自身的 top1 不受影响
                Zs_n, Zs_o = {}, {}
                for nm in Mn:
                    sv = Mv[nm][:, NEW]
                    mu, sd = float(sv.mean()), float(sv.std())
                    sd = sd if sd > 1e-9 else 1.0
                    Zs_n[nm] = (Mn[nm] - mu) / sd
                    Zs_o[nm] = (Mo[nm] - mu) / sd

                out = {nm: (Mn[nm], Mo[nm]) for nm in Mn}
                for g, mem in GROUPS.items():
                    out[g] = (np.mean([Zs_n[m] for m in mem], 0),
                              np.mean([Zs_o[m] for m in mem], 0))

                for arm, (Sn_, So_) in out.items():
                    res["acc"][tag].setdefault(arm, {}).setdefault(
                        e_hold, []).append(E53.scores_to_metrics(Sn_, So_, NEW))
                del net_c
                torch.cuda.empty_cache()

        line = [f"[{e_hold}]"]
        for t in tags:
            for a in ("ridge", "clip", "conse", "ens-chem", "ens-all", "ens-top3", "ens-rand"):
                d = res["acc"][t].get(a, {}).get(e_hold, [])
                if d:
                    line.append(f"{t}:{a}={np.mean([x['auc'] for x in d if x['auc'] is not None]):.3f}")
        print(" ".join(line), flush=True)

    SFX = args.suffix
    with open(os.path.join(OUT, f"exp55_fusion{SFX}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"[exp55] 已写出 out/exp55_fusion{SFX}.json", flush=True)
    write_report(res, tags, eids, SFX)


def per_entity(res, t, a, met="auc"):
    return {e: float(np.mean([x[met] for x in res["acc"][t].get(a, {}).get(e, [])
                              if x[met] is not None]))
            for e in res["meta"]["eids"] if res["acc"][t].get(a, {}).get(e)}


def write_report(res, tags, eids, SFX=""):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp55 —— 对齐方式的集成融合（score-level fusion，无超参）\n")
    w("标准化：z_m(x) = (s_m(x) − μ_m)/σ_m，μ/σ 取自老类验证集 Xval；融合 = 等权平均。\n")
    for t in tags:
        w(f"\n## k={t} —— macro auc（融合组只有 auc）\n")
        w("| 臂 | " + " | ".join(eids) + " | **macro** | **worst** |")
        w("|---|" + "---|" * (len(eids) + 2))
        rows = []
        for a in ARMS:
            d = per_entity(res, t, a)
            if not d:
                continue
            rows.append((a, np.mean(list(d.values())),
                         f"| `{a}` | " +
                         " | ".join(f"{d[e]:.3f}" if e in d else "—" for e in eids) +
                         f" | **{np.mean(list(d.values())):.3f}** |"
                         f" **{min(d.values()):.3f}** |"))
        for _, m, line in sorted(rows, key=lambda x: -x[1]):
            w(line)
    w("\n## 判据\n")
    for t in tags:
        w(f"\n### k={t}")
        base = per_entity(res, t, "ridge")
        if not base:
            continue
        x = np.array([base[e] for e in eids])
        for g in GROUPS:
            d = per_entity(res, t, g)
            if not d:
                continue
            y = np.array([d[e] for e in eids])
            p = ttest_rel(y, x).pvalue
            w(f"- `{g}` {y.mean():.3f} vs `ridge` {x.mean():.3f}: "
              f"Δ={np.mean(y - x):+.3f} p={p:.4f} "
              f"{'✅ 显著优于本文单对齐' if np.mean(y - x) > 0 and p < 0.05 else ''}"
              f"{'❌ 显著更差' if np.mean(y - x) < 0 and p < 0.05 else ''}"
              f"   worst={y.min():.3f} (ridge worst {x.min():.3f})")
        # H55b：最好的集成 vs 最好的单一基线
        singles = [a for a in SINGLE + ["clip"] if per_entity(res, t, a)]
        bs = max(singles, key=lambda a: np.mean(list(per_entity(res, t, a).values())))
        bg_ = max(GROUPS, key=lambda a: np.mean(list(per_entity(res, t, a).values())))
        y = np.array([per_entity(res, t, bg_)[e] for e in eids])
        z = np.array([per_entity(res, t, bs)[e] for e in eids])
        p = ttest_rel(y, z).pvalue
        w(f"- **H55b** 最好集成 `{bg_}` ({y.mean():.3f}) vs 最好单一 `{bs}` "
          f"({z.mean():.3f}): Δ={np.mean(y - z):+.3f} p={p:.4f}")
        # H55d 因果
        a1 = per_entity(res, t, "ens-chem"); a2 = per_entity(res, t, "ens-rand")
        if a1 and a2:
            y = np.array([a1[e] for e in eids]); z = np.array([a2[e] for e in eids])
            w(f"- **H55d** ens-chem − ens-rand: Δ={np.mean(y - z):+.3f} p={ttest_rel(y, z).pvalue:.4f}")

    p = os.path.join(OUT, f"exp55_p6_fusion{SFX}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[exp55] 已写出 {p}", flush=True)


if __name__ == "__main__":
    main()
