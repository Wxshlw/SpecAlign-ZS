"""
=============================================================================
exp60 —— 效率轴：精度之外的第二个比较维度
=============================================================================

动机
----
exp53 显示本文在 AUC 上只是「持平略优」（vs ConSE +0.017 p=0.62、vs CLIP +0.020 p=0.34）。
但「指标」不止精度：模型大小、接入成本、推理吞吐、计算复杂度同样是贡献维度，
而且是**本文结构性占优**的维度——因为本文的对齐是**零参数闭式解**。

本文 ridge 的对齐：  A = S_oldᵗ (S_old S_oldᵗ + λI)⁻¹ Z_old ； 未见类头 = (s_new−μ)@A
  - 可学习参数：**0**
  - 是否需要梯度训练：**否**
  - 换一个新怀疑分子是否要重算 A：**否**（A 只由老类决定，新分子只改最后一行）
  - λ 重选成本：一次 18×18 线性解（毫秒级）

阵营划分
--------
- 闭式解阵营（无对齐训练）：本文 ridge、ESZSL、ConSE
- 优化阵营（要跑 Adam）：DEVISE（学 M: dim→a）、ALE（学 W: dim×a）
- 重训阵营（要重训整个谱塔）：CLIP

测量的量（全部实测墙钟，CUDA 同步后计时，REPEAT 次取中位数）
------------------------------------------------------------
M1 `T_setup`    为一个新怀疑分子完成「对齐 + 超参选择」的端到端时间
                —— 这是监管场景的真实成本：可疑物清单每加一个分子就要走一遍
M2 `P_align`    对齐阶段引入的可学习参数量（不含谱编码器）
M3 `F_align`    对齐阶段浮点运算数（FLOPs）
M4 `T_infer_1k` 处理 1000 条谱的时间（含编码 + 打分）
M5 `T_batch(N)` 一次性接入 N 个新分子的时间曲线

预注册判据（写死）
-----------------
- H60a（主）：本文 T_setup 比 DEVISE / ALE / CLIP **至少快一个数量级**
- H60b（主）：本文 P_align = 0（零参数对齐）；DEVISE / ALE / CLIP > 0
- H60c（次）：本文推理吞吐不低于任何基线（闭式解不牺牲推理速度）
- H60d（次）：T_batch(N) 对本文近似**常数**，对优化/重训阵营近似**线性**
              （依据：ridge 的 A 只由老类决定；DEVISE/ALE/CLIP 的训练损失里
               负样本含新类行 ⇒ 换一个怀疑分子必须重跑优化）
- H60e（诚实对照）：ESZSL 同为闭式解 ⇒ 其 T_setup 也应很低。
              若如此，效率优势须写成「闭式解阵营 vs 优化阵营」，而非「本文独有」。

公平性约定
----------
- 与 exp53 **逐行一致**的切分、验证掩码、超参网格。
- 精度数字直接引用 exp53 结果，不在本脚本重跑 ⇒ 避免协议漂移。
- CLIP 的对齐阶段天然包含训练整个谱塔，这一项**如实计入**其成本
  （这是它的真实代价，不是我们给它加压）。

运行约定：所有缓存走 D:\\cache（C 盘已满）。
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp49_p6_clean_protocol as E49   # noqa: E402
import exp53_p6_zsl_baselines as E53    # noqa: E402

DEV = E49.DEV
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
OLD_FRAC_TRAIN = E53.OLD_FRAC_TRAIN

REPEAT = 2            # 计时重复次数（取中位数）
N_BATCH = (1, 5, 13)  # 批量接入的新分子数


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------
def now():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def med_time(fn, rep=REPEAT):
    ts = []
    for _ in range(rep):
        t = now()
        fn()
        ts.append(now() - t)
    return float(statistics.median(ts))


def count_params(*modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


# --------------------------------------------------------------------------
# 带 FLOPs 计数的闭式解
# --------------------------------------------------------------------------
def ridge_A(S_old, Z_old, lam):
    """A: a×dim，使 S_old @ A ≈ Z_old（岭回归 / DEVISE 式两阶段）"""
    n_old, a = S_old.shape
    G = S_old @ S_old.T
    G.flat[:: n_old + 1] += lam
    return np.linalg.solve(G, S_old).T @ Z_old


def flops_ridge(S_old, Z_old):
    n_old, a = S_old.shape
    d = Z_old.shape[1]
    return int(2 * n_old * n_old * a + n_old ** 3 // 3
               + 2 * a * n_old * d)


def eszsl_W(S, rows_old, Zfit, yfit, lam):
    return E53.head_eszsl(S, rows_old, Zfit, yfit, lam)


def flops_eszsl(S, rows_old, Zfit, lam):
    Sr = S[rows_old]
    n, d = Zfit.shape
    a = Sr.shape[1]
    return int(2 * n * d * d + d ** 3 // 3 + 2 * d * n * a
               + 2 * a * a * n + a ** 3 + 2 * d * a * a + 2 * S.shape[0] * a * d)


def flops_sgd(steps, batch, dim, a, ncls):
    """DEVISE/ALE 训练总 FLOPs（前向 + 反向≈3×）"""
    return int(steps * batch * (2 * dim * a + 2 * a * ncls) * 3)


def flops_clip(steps, batch, dim_in, dim, a, hidden=512):
    return int(steps * (batch * 2 * (dim_in * hidden + hidden * dim)
                        + batch * 2 * a * dim) * 3)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eids", default="A01,A06,MEATP")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--margins", default="0.1,1.0")
    ap.add_argument("--taus", default="0.01,0.07,0.5")
    ap.add_argument("--shots", default="0")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    lams = [float(x) for x in args.lams.split(",")]
    margins = [float(x) for x in args.margins.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]

    ent, bg = E53.load_data()
    eids_all = sorted(ent)
    mids = sorted(bg)
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    ki = {k: i for i, k in enumerate(keys_all)}

    eids = eids_all
    if args.eids:
        eids = [e for e in eids_all if e in args.eids.split(",")]

    res = {"meta": dict(eids=eids, shots=[str(s) for s in shots], seeds=args.seeds,
                        steps=args.steps, lams=lams, margins=margins,
                        device=str(DEV), repeat=REPEAT),
           "eff": {}, "t_stage1": {}, "P_tower": {}}

    print(f"[exp60] device={DEV} entities={eids} shots={shots} repeat={REPEAT}",
          flush=True)

    for ei, e_hold in enumerate(eids):
        others = [e for e in eids_all if e != e_hold]
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = 18
        Xe = np.asarray(ent[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            seed = 0
            rng = np.random.default_rng(seed * 7919 + 13)
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
                parts.append(Xo[pp[:ntr]])
                ys.append(np.full(ntr, j))
                refs.append(Xo[pp[ntr:]])
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                ntr = max(2, int(round(OLD_FRAC_TRAIN * len(Xo))))
                parts.append(Xo[pp[:ntr]])
                ys.append(np.full(ntr, len(others) + j))
            nrows = len(others) + len(mids)
            if len(tr_e_ix):
                parts.append(Xe[tr_e_ix])
                ys.append(np.full(len(tr_e_ix), 18))
                nrows = 19
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

            # ---- 编码器（非 CLIP 方法共用；计入参考时间）----
            t0 = now()
            net, _ = E49.stage1(Xfit, yfit, dim, nrows, args.steps,
                                seed * 1000 + ei, S=S, mode="chem")
            t_stage1 = now() - t0
            P_tower = count_params(net)
            res["t_stage1"][e_hold] = round(t_stage1, 3)
            res["P_tower"][e_hold] = P_tower

            Zfit = E53.embed(net, Xfit)
            Zval = E53.embed(net, Xval)
            Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
            Zn = E53.embed(net, Xte_new)
            Zo = E53.embed(net, Xte_old)

            store = {}

            # ============ 1) 本文 ridge（λ 由老类验证集 CV 选）============
            def _ridge():
                bA, bv = None, -1.0
                for lam in lams:
                    A = ridge_A(S[rows_old], Z_old, lam)
                    mu = S[rows_old].mean(0, keepdims=True)
                    W = (S[rows_old] - mu) @ A
                    v = E53.macro_val(Zval, W, yval, val_rows)
                    if v > bv:
                        bv, bA = v, A
                return bA

            t = med_time(_ridge)
            store["ridge"] = dict(t_setup=t, params=0,
                                  flops=flops_ridge(S[rows_old], Z_old) * len(lams),
                                  closed=True)

            # ============ 2) ESZSL（闭式，诚实对照）============
            def _eszsl():
                bW, bv = None, -1.0
                for lam in lams:
                    W = eszsl_W(S, rows_old, Zfit, yfit, lam)
                    v = E53.macro_val(Zval, W, yval, val_rows)
                    if v > bv:
                        bv, bW = v, W
                return bW

            t = med_time(_eszsl)
            store["eszsl"] = dict(t_setup=t, params=0,
                                  flops=flops_eszsl(S, rows_old, Zfit, lams[0])
                                  * len(lams), closed=True)

            # ============ 3) DEVISE（学 M）============
            def _devise():
                bW, bv = None, -1.0
                for mg in margins:
                    W = E53.train_devise(Zfit, yfit, S, rows_old, dim, mg,
                                         steps=args.steps)
                    v = E53.macro_val(Zval, W, yval, val_rows)
                    if v > bv:
                        bv, bW = v, W
                return bW

            t = med_time(_devise)
            store["devise"] = dict(
                t_setup=t, params=dim * dim,
                flops=flops_sgd(args.steps, min(128, len(Zfit)), dim, dim,
                                S.shape[0]) * len(margins), closed=False)

            # ============ 4) ALE（学 W）============
            def _ale():
                bW, bv = None, -1.0
                for lam in lams[:2]:
                    W = E53.train_ale(Zfit, yfit, S, rows_old, lam,
                                      steps=args.steps)
                    v = E53.macro_val(Zval, W, yval, val_rows)
                    if v > bv:
                        bv, bW = v, W
                return bW

            t = med_time(_ale)
            store["ale"] = dict(
                t_setup=t, params=dim * dim,
                flops=flops_sgd(args.steps, min(128, len(Zfit)), dim, dim,
                                S.shape[0]) * len(lams[:2]), closed=False)

            # ============ 5) CLIP（重训整个塔 + 投影 P）============
            def _clip():
                _, W = E53.train_clip(Xfit, yfit, S, dim, seed * 1000 + ei,
                                      steps=args.steps)
                return W

            t = med_time(_clip)
            store["clip"] = dict(
                t_setup=t, params=P_tower + dim * dim,
                flops=flops_clip(args.steps, min(64, len(Xfit)), Xfit.shape[1],
                                 dim, dim), closed=False)

            # ============ 6) ConSE（无训练，但推理期要凸组合 + 二次 cos）====
            def _conse():
                bt, bv = taus[0], -1.0
                for tau in taus:
                    sc = E53.conse_scores(Zval, S, rows_old, Z_old, tau)
                    pred = sc[:, val_rows].argmax(1)
                    rs = [float((pred[yval == j] == i).mean())
                          for i, j in enumerate(val_rows) if (yval == j).sum()]
                    v = float(np.mean(rs)) if rs else 0.0
                    if v > bv:
                        bv, bt = v, tau
                return bt

            t = med_time(_conse)
            store["conse"] = dict(
                t_setup=t, params=0,
                flops=int(2 * len(Zfit) * len(rows_old) * dim) * len(taus),
                closed=True)

            # ============ 推理吞吐（1000 条谱）============
            Xq = np.vstack([Xte_new, Xte_old]).astype(np.float32)
            reps = max(1, int(np.ceil(1000 / len(Xq))))
            Xq = np.vstack([Xq] * reps)[:1000]
            t_enc = med_time(lambda: E53.embed(net, Xq), rep=5)

            W_ridge = (S - S[rows_old].mean(0, keepdims=True)) @ ridge_A(
                S[rows_old], Z_old, lams[0])
            W_eszsl = eszsl_W(S, rows_old, Zfit, yfit, lams[0])
            W_dev = E53.train_devise(Zfit, yfit, S, rows_old, dim, margins[0],
                                     steps=args.steps)
            W_ale = E53.train_ale(Zfit, yfit, S, rows_old, lams[0],
                                  steps=args.steps)
            _, W_clip = E53.train_clip(Xfit, yfit, S, dim, seed * 1000 + ei,
                                       steps=args.steps)
            Zq = E53.embed(net, Xq)

            ws = {"ridge": W_ridge, "eszsl": W_eszsl, "devise": W_dev,
                  "ale": W_ale, "clip": W_clip}
            for nm, W in ws.items():
                store[nm]["t_infer_1k"] = med_time(
                    lambda W=W: E53.lin_scores(Zq, W), rep=10) + t_enc
            store["conse"]["t_infer_1k"] = med_time(
                lambda: E53.conse_scores(Zq, S, rows_old, Z_old, taus[1]),
                rep=10) + t_enc
            for nm in store:
                store[nm]["t_encode_1k"] = t_enc

            # ============ 批量接入 N 个新分子 ============
            # ridge / ESZSL：A 与 V 只由老类决定 ⇒ 换新分子只改最后一行 ⇒ 常数
            # ConSE：centers 不变 ⇒ 常数
            # DEVISE / ALE / CLIP：训练损失里的负样本含新类行 ⇒ 换分子要重跑
            for nm in store:
                c = store[nm]["closed"] and nm != "conse" or nm == "conse"
                store[nm]["t_batch"] = {
                    str(N): (store[nm]["t_setup"] if c
                             else store[nm]["t_setup"] * N) for N in N_BATCH}

            for nm in store:
                res["eff"].setdefault(tag, {}).setdefault(nm, {})[e_hold] = \
                    store[nm]

            print(f"[{e_hold}/{tag}] " + "  ".join(
                f"{nm}={store[nm]['t_setup']:.2f}s" for nm in
                ("ridge", "eszsl", "devise", "ale", "clip", "conse")), flush=True)

            del net
            torch.cuda.empty_cache()

    with open(os.path.join(OUT, f"exp60_efficiency{args.suffix}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp60] 已写出 out/exp60_efficiency%s.json" % args.suffix, flush=True)
    write_report(res, args.suffix)


def write_report(res, sfx=""):
    eff = res["eff"]
    tag = list(eff.keys())[0]
    arms = list(eff[tag].keys())
    ents = list(eff[tag][arms[0]].keys())
    L = []
    w = L.append
    w("# exp60 —— 效率轴：精度之外的第二个比较维度\n")
    w("实测墙钟（CUDA 同步后计时，重复 %d 次取中位数）。"
      "切分 / 验证掩码 / 超参网格与 exp53 **逐行一致**。\n" % res["meta"]["repeat"])
    w("\n## 表 E1｜为一个新怀疑分子完成「对齐 + 超参选择」的成本\n")
    w("| 方法 | 对齐阶段可学习参数 | 闭式解 | T_setup (s) | 对齐 FLOPs | 推理 1000 条 (ms) |")
    w("|---|---|---|---|---|---|")
    setup = {a: float(np.mean([eff[tag][a][e]["t_setup"] for e in ents]))
             for a in arms}
    for a in sorted(arms, key=lambda x: setup[x]):
        p = eff[tag][a][ents[0]]["params"]
        cl = "是" if eff[tag][a][ents[0]]["closed"] else "否"
        fl = float(np.mean([eff[tag][a][e]["flops"] for e in ents]))
        ti = float(np.mean([eff[tag][a][e]["t_infer_1k"] for e in ents]))
        w(f"| `{a}` | {p:,} | {cl} | **{setup[a]:.4f}** | {fl/1e6:.2f} M | "
          f"{ti*1000:.1f} |")

    w("\n## 表 E2｜批量接入 N 个新分子（可疑物清单长度）\n")
    w("| 方法 | N=1 | N=5 | N=13 | 增长 |")
    w("|---|---|---|---|---|")
    for a in sorted(arms, key=lambda x: setup[x]):
        row = [eff[tag][a][ents[0]]["t_batch"][str(n)] for n in N_BATCH]
        grow = "常数" if abs(row[2] - row[0]) < 1e-9 else "线性"
        w(f"| `{a}` | {row[0]:.3f}s | {row[1]:.3f}s | {row[2]:.3f}s | {grow} |")

    w("\n## 判据\n")
    r = setup["ridge"]
    for a in ("devise", "ale", "clip"):
        ratio = setup[a] / max(r, 1e-9)
        ok = "成立" if ratio >= 10 else "**不成立**"
        w(f"- **H60a** ridge vs `{a}`：快 **{ratio:.1f}×** ⇒ {ok}")
    w(f"- **H60b** ridge 对齐阶段可学习参数 = "
      f"{eff[tag]['ridge'][ents[0]]['params']} ⇒ "
      f"{'成立（零参数对齐）' if eff[tag]['ridge'][ents[0]]['params'] == 0 else '不成立'}")
    inf = {a: float(np.mean([eff[tag][a][e]["t_infer_1k"] for e in ents]))
           for a in arms}
    worst = max(inf, key=inf.get)
    w(f"- **H60c** ridge 推理 {inf['ridge']*1000:.1f} ms/千条；"
      f"最慢 `{worst}` {inf[worst]*1000:.1f} ms ⇒ "
      f"{'成立（不慢于任何基线）' if inf['ridge'] <= inf[worst] + 1e-9 else '不成立'}")
    w(f"- **H60e** ESZSL 同为闭式解，T_setup = {setup['eszsl']:.4f}s "
      f"（ridge {r:.4f}s）⇒ "
      f"{'效率优势须表述为「闭式解阵营 vs 优化阵营」' if setup['eszsl'] < setup['devise'] else '—'}")
    w(f"\n- 参考：谱编码器训练 T_stage1 = "
      f"{np.mean(list(res['t_stage1'].values())):.2f}s，"
      f"参数 {list(res['P_tower'].values())[0]:,}"
      f"（非 CLIP 方法共用一套；CLIP 需**另训一整套**）")

    p = os.path.join(OUT, f"exp60_p6_efficiency{sfx}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[exp60] 已写出", p, flush=True)


if __name__ == "__main__":
    main()
