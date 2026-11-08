"""exp54 —— 化学先验的**知识表示**消融：CLM vs ECFP vs 拼接 vs 零信息对照

=============================================================================
为什么必须做
=============================================================================
到目前为止所有"化学先验"实验都用同一种表示：CLM（化学语言模型嵌入，384 维）。
审稿人可以问：
    "你的结论是 CLM 的功劳，还是**随便一种分子指纹**都能做到？"
本实验把先验的知识源换掉，检验结论的**稳健性**与**表示依赖性**。

臂（知识源 × 对齐方式）：
    `clm`    CLM 嵌入（384 维）—— 本文主张的化学语言模型语义
    `ecfp`   ECFP 折叠指纹（384 位）—— 经典子结构指纹，无语言模型
    `concat` [CLM | ECFP]（768 维）—— 检验两者是否互补
    `rand`   同形状同范数的高斯随机码本（零化学信息，因果下限）

对齐方式各跑两个（不押注单一方法）：
    `ridge`  本文（W=(S−μ)·A，A 由老类 ridge 闭式解）
    `eszsl`  ESZSL 闭式解（exp53 中可能更强）

=============================================================================
预注册判据
=============================================================================
- H54a（主）：`clm` 的 k=0 macro auc ≥ `ecfp`
        ⇒ 化学语言模型的语义**优于**子结构指纹（若反之，说明只是"有分子信息"就够）
- H54b：`concat` ≥ max(clm, ecfp) ⇒ 两者互补（若成立，主张臂应换成 concat）
- H54c（因果，必成立）：三者均显著 > `rand`
- H54d（次要）：表示之间的排序在 ridge / eszsl 两种对齐下**一致**
        ⇒ 结论不依赖于具体对齐方式
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp49_p6_clean_protocol as E49
import exp53_p6_zsl_baselines as E53

DEV = E49.DEV
OUT = E49.OUT
OLD_FRAC_TRAIN = E49.OLD_FRAC_TRAIN

REPRS = ["clm", "ecfp", "concat"]
METHODS = ["ridge", "eszsl"]
ARMS = [f"{m}-{r}" for m in METHODS for r in REPRS] + \
       [f"{m}-rand" for m in METHODS]


@torch.no_grad()
def embed(net, X):
    net.eval()
    return F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)),
                       dim=1).cpu().numpy()


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
    CLM = np.asarray(zc["clm"], float)
    EC = np.asarray(zc["ecfp384"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    idx = [ki[e] for e in sorted(ent.keys())] + [ki[f"M_{m}"] for m in mids]
    MATS = {"clm": CLM[idx], "ecfp": EC[idx],
            "concat": np.concatenate([CLM[idx], EC[idx]], 1)}
    dim = CLM.shape[1]

    res = {"acc": {t: {a: {} for a in ARMS} for t in tags},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "arms": ARMS, "lams": lams,
                    "repr_dims": {k: v.shape[1] for k, v in MATS.items()}}}
    print(f"[exp54] device={DEV} shots={tags} seeds={args.seeds} "
          f"dims={res['meta']['repr_dims']}", flush=True)

    for ei, e_hold in enumerate(eids):
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        order = [ki[e] for e in others] + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]
        S_all = {k: v[order] for k, v in MATS.items()}
        S_rand = E53.rand_like(S_all["clm"], np.random.default_rng(2000 + ei))
        S_all["rand"] = S_rand
        NEW = 18
        Xe = np.asarray(ent[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)   # 与 exp49/53 同切分
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

                # 先为每个知识源训一个编码器（stage1 的冻结头必须是 dim 维）
                nets, Zs = {}, {}
                for rk in REPRS:
                    S = S_all[rk]
                    if S.shape[1] == dim:
                        S_head = S                       # 与 exp49/53 完全一致
                    else:
                        # 768 维 → 384 维：固定种子的随机投影（确定性，不改变 clm/ecfp）
                        P0 = np.random.default_rng(0).standard_normal(
                            (S.shape[1], dim)) / np.sqrt(S.shape[1])
                        S_head = S @ P0
                    nets[rk], _ = E49.stage1(Xfit, yfit, dim, nrows, args.steps,
                                             seed * 1000 + ei, S=S_head, mode="chem")
                # rand 对照复用 CLM 编码器 ⇒ 唯一的差别是"码本有没有化学信息"
                nets["rand"] = nets["clm"]

                for rk in REPRS + ["rand"]:
                    S = S_all[rk]
                    net = nets[rk]
                    Zf, Zv = embed(net, Xfit), embed(net, Xval)
                    Zn, Zo = embed(net, Xte_new), embed(net, Xte_old)
                    Z_old = np.stack([Zf[yfit == j].mean(0) for j in rows_old])

                    for meth in METHODS:
                        arm = f"{meth}-{rk}"

                        def build(lam, S=S, Zf=Zf, Z_old=Z_old):
                            return (E53.head_ridge(S, rows_old, Z_old, lam)
                                    if meth == "ridge" else
                                    E53.head_eszsl(S, rows_old, Zf, yfit, lam))
                        best, bv = lams[0], -1.0
                        for lam in lams:
                            v = E53.macro_val(Zv, build(lam), yval, val_rows)
                            if v > bv:
                                bv, best = v, lam
                        W = build(best)
                        res["acc"][tag].setdefault(arm, {}).setdefault(
                            e_hold, []).append(E53.scores_to_metrics(
                            E53.lin_scores(Zn, W), E53.lin_scores(Zo, W), NEW))
                del nets
                torch.cuda.empty_cache()

        line = [f"[{e_hold}]"]
        for t in tags:
            for a in ("ridge-clm", "ridge-ecfp", "ridge-concat", "ridge-rand",
                      "eszsl-clm", "eszsl-ecfp", "eszsl-concat"):
                d = res["acc"][t].get(a, {}).get(e_hold, [])
                if d:
                    line.append(f"{t}:{a}={np.mean([x['auc'] for x in d if x['auc'] is not None]):.3f}"
                                f"/{np.mean([x['top1'] for x in d]):.3f}")
        print(" ".join(line), flush=True)

    SFX = args.suffix
    with open(os.path.join(OUT, f"exp54_repr{SFX}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"[exp54] 已写出 out/exp54_repr{SFX}.json", flush=True)
    write_report(res, tags, eids, SFX)


def per_entity(res, t, a, met):
    out = {}
    for e in res["meta"]["eids"]:
        v = [x[met] for x in res["acc"][t].get(a, {}).get(e, []) if x[met] is not None]
        if v:
            out[e] = float(np.mean(v))
    return out


def write_report(res, tags, eids, SFX=""):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp54 —— 化学先验的**知识表示**消融（CLM / ECFP / 拼接 / 零信息）\n")
    w("同一套干净切分；对齐方式各跑 ridge 与 eszsl 两种，避免结论依赖单一对齐。\n")

    for t in tags:
        w(f"\n## k={t}\n")
        for met in ("auc", "top1"):
            w(f"\n### {met}（macro）\n| 臂 | " + " | ".join(eids) + " | **macro** |")
            w("|---|" + "---|" * (len(eids) + 1))
            rows = []
            for a in ARMS:
                d = per_entity(res, t, a, met)
                if not d:
                    continue
                rows.append((a, np.mean(list(d.values())),
                             f"| `{a}` | " +
                             " | ".join(f"{d[e]:.3f}" if e in d else "—" for e in eids) +
                             f" | **{np.mean(list(d.values())):.3f}** |"))
            for _, m, line in sorted(rows, key=lambda x: -x[1]):
                w(line)

    w("\n## 判据\n")
    for t in tags:
        w(f"\n### k={t}")
        for meth in METHODS:
            dd = {r: per_entity(res, t, f"{meth}-{r}", "auc")
                  for r in REPRS + ["rand"]}
            if not dd["clm"]:
                continue
            mm = {r: np.mean(list(dd[r].values())) for r in dd if dd[r]}
            w(f"- `{meth}`：" +
              "，".join(f"{r}={v:.3f}" for r, v in mm.items()))
            for r in REPRS:
                if r == "clm" or not dd[r]:
                    continue
                kk = [e for e in dd["clm"] if e in dd[r]]
                x = np.array([dd["clm"][e] for e in kk]); y = np.array([dd[r][e] for e in kk])
                w(f"   - clm − {r}: Δ={np.mean(x - y):+.3f} p={ttest_rel(x, y).pvalue:.4f}")
            for r in REPRS:
                if not dd[r] or not dd["rand"]:
                    continue
                kk = [e for e in dd[r] if e in dd["rand"]]
                x = np.array([dd[r][e] for e in kk]); y = np.array([dd["rand"][e] for e in kk])
                w(f"   - H54c {r} − rand: Δ={np.mean(x - y):+.3f} p={ttest_rel(x, y).pvalue:.4f}")

    p = os.path.join(OUT, f"exp54_p6_repr_ablation{SFX}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[exp54] 已写出 {p}", flush=True)


if __name__ == "__main__":
    main()
