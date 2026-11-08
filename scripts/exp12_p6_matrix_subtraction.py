# -*- coding: utf-8 -*-
"""exp12 —— 论文6 Step 4｜基质扣除（differential spectroscopy）与掺混型 zero-shot 可辨识性。

## 动机（exp11 的负面结果给出的线索）
exp08/exp11 的 13 折呈双峰：
  - authentication 型（MEATCH/MEATP/MEATT/COFA/COFR）与"高信噪比掺混"型（A05 葵花籽油、
    A07 苹果汁、HSYRUP）表现良好
  - 低信噪比掺混型（A02 花生粉、A03 木屑、A06 麦芽粉）在所有臂、所有对照下均为 0.000

exp11 证明**文本侧等距化（ABTT / 学习式投影 / 对称 InfoNCE）无法提升 zero-shot top-1**，
即使 alignment 0.35→0.85、uniformity −0.47→−1.71。说明瓶颈不在文本表示空间。

## 真正的死因（数据层证据）
ENTITY_META 显示 A01–A07 + HSYRUP 为 adulteration 型：query 谱 = **基质 + 掺杂物**的混合谱
（如"肉桂 + 核桃粉"），而文本先验是**掺杂物**的组分谱。基质在混合谱中占绝对主导，
掺杂物仅贡献微弱残差 → 谱侧几乎不可分，文本侧再怎么拉开也无用。
同基质姊妹（A01/A02/A03 同为 cinnamon）尤其严重。

## 本实验的解法
差谱法（FTIR 掺假检测的常规手段）：
    X_res = X_mixture − α · X_matrix
把"肉桂+核桃粉"变成近似纯核桃粉的残差谱，使其与掺杂物级文本先验直接对齐。
零样本性质不变：基质背景谱在训练时作为背景类提供，扣除不需要知道掺杂物身份。

三臂：
  raw     : 不扣除（= exp08 baseline）
  sub     : 直接扣除基质均值谱
  sub_fit : 逐条谱最小二乘拟合非负系数 α 后扣除（适应掺混浓度差异）
"""
from __future__ import annotations
import os, sys, csv, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp05_p6_g0 as E5
import exp08_p6_g1 as E8
from exp11_p6_contrastive import train_joint, eval_fold

OUT = r"D:/2026sci论文实验设计/论文6/out"
ARMS = ["raw", "sub", "sub_fit"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60


# ----------------------------------------------------------------------------
def build_residual(ent, bg, mode):
    """mode: raw | sub | sub_fit。authentication 型与无背景者保持原谱。"""
    out = {}
    for e, X in ent.items():
        name, mid, kind = G1.ENTITY_META.get(e, (e, None, "authentication"))
        if mode == "raw" or kind != "adulteration" or mid not in bg:
            out[e] = X
            continue
        mu = np.asarray(bg[mid]).mean(0)
        if mode == "sub":
            out[e] = X - mu
        elif mode == "sub_fit":
            den = float(mu @ mu)
            if den <= 1e-12:
                out[e] = X
                continue
            alpha = np.clip((X @ mu) / den, 0.0, None)[:, None]
            out[e] = X - alpha * mu
        else:
            raise ValueError(mode)
    return out


def cos_matrix(ent, bg):
    """诊断：实体谱与基质谱 / 实体谱之间的余弦（扣除前后对照）。"""
    def u(X):
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    rows = []
    for e, X in ent.items():
        name, mid, kind = G1.ENTITY_META.get(e, (e, None, "authentication"))
        Xe = u(X).mean(0)
        Xe /= np.linalg.norm(Xe)
        if mid in bg:
            mu = u(np.asarray(bg[mid])).mean(0)
            mu /= np.linalg.norm(mu)
            c_mat = float(Xe @ mu)
        else:
            c_mat = float("nan")
        others = [float(Xe @ (u(ent[o]).mean(0) / np.linalg.norm(u(ent[o]).mean(0))))
                  for o in ent if o != e]
        rows.append((e, kind, c_mat, float(np.max(others))))
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    arms = args.arms.split(",")
    tag = args.tag or f"s{len(seeds)}"

    ent0, bg, grid = G1.load_g1()
    eids = sorted(ent0.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    print(f"[exp12] device={DEV} | arms={arms} | seeds={seeds}", flush=True)

    V = E8.build_text_variants_g1(eids, mids)
    T_full = np.stack([V["Original"][k] for k in keys]).astype(np.float32)
    T_full = T_full / np.linalg.norm(T_full, axis=1, keepdims=True)
    ZERO = np.zeros((len(keys), len(keys)), np.float32)

    # --- 诊断：扣除前后 实体–基质 余弦 ---
    print("\n=== 诊断：扣除前后「实体谱 vs 自身基质谱」余弦 ===", flush=True)
    print(f"{'eid':9s} {'kind':14s} {'raw':>8s} {'sub':>8s} {'sub_fit':>8s}")
    diag = {m: {r[0]: r for r in cos_matrix(build_residual(ent0, bg, m), bg)} for m in ARMS}
    for e in eids:
        row = f"{e:9s} {diag['raw'][e][1]:14s} "
        row += " ".join(f"{diag[m][e][2]:8.4f}" if np.isfinite(diag[m][e][2]) else f"{'—':>8s}"
                        for m in ARMS)
        print(row, flush=True)

    results = {a: {} for a in arms}
    ENT = {a: build_residual(ent0, bg, a) for a in arms}
    folds = eids if args.folds <= 0 else eids[:args.folds]

    for arm in arms:
        ent = ENT[arm]
        for fold, e_hold in enumerate(folds):
            eids_tr = [e for e in eids if e != e_hold]
            tr_ix = {e: i for i, e in enumerate(eids_tr)}
            parts = [ent[e] for e in eids_tr] + [bg[m] for m in mids]
            ys = np.concatenate([np.full(len(ent[e]), tr_ix[e]) for e in eids_tr] +
                                [np.full(len(bg[m]), len(eids_tr) + j) for j, m in enumerate(mids)])
            Xs = np.concatenate(parts)
            ix_seen = [keys.index(e) for e in eids_tr] + [keys.index(f"M_{m}") for m in mids]
            T_seen = T_full[ix_seen]
            LW = ZERO[np.ix_(ix_seen, ix_seen)]
            Xh = ent[e_hold]
            true_ix = eids.index(e_hold)
            accs = []
            for seed in seeds:
                spec, text_fn = train_joint(Xs, ys, T_seen, LW, "baseline", args.epochs,
                                            seed * 1000 + fold)
                acc, _ = eval_fold(spec, text_fn, Xh, T_full[:len(eids)], true_ix)
                accs.append(acc)
                del spec
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            for s, a in zip(seeds, accs):
                results[arm][(e_hold, s)] = a
            print(f"[{arm}] {e_hold}: top1={np.mean(accs):.3f} (n_q={len(Xh)})", flush=True)

    if len(folds) == len(eids):
        write_report(eids, arms, results, diag, seeds, tag)
    else:
        print(f"[smoke] 只跑 {len(folds)} 折，跳过报告", flush=True)


# ----------------------------------------------------------------------------
def write_report(eids, arms, results, diag, seeds, tag):
    chance = 1.0 / len(eids)
    fold_top1 = {a: {e: np.mean([results[a][(e, s)] for s in seeds]) for e in eids} for a in arms}
    mean_top1 = {a: float(np.mean(list(fold_top1[a].values()))) for a in arms}
    kinds = {e: G1.ENTITY_META.get(e, (e, None, "?"))[2] for e in eids}
    grp = {}
    for k in ("adulteration", "authentication"):
        sel = [e for e in eids if kinds[e] == k]
        if sel:
            grp[k] = {a: float(np.mean([fold_top1[a][e] for e in sel])) for a in arms}

    p_raw = {}
    for a in arms:
        if a == "raw":
            continue
        o = [fold_top1[a][e] for e in eids]
        c = [fold_top1["raw"][e] for e in eids]
        p_raw[a], _ = E5.perm_paired_greater(o, c)

    lines = [f"# exp12 基质扣除与掺混型 zero-shot 可辨识性（{len(seeds)} seeds）", "",
             "> 差谱法：X_res = X_mixture − α·X_matrix，使掺杂物残差与掺杂物级文本先验直接对齐。",
             "> 零样本性质不变：基质背景谱训练时作为背景类提供，扣除不需知道掺杂物身份。",
             "> 编码器/文本/温度/损失与 exp08 baseline 完全一致，唯一变量＝谱侧预处理。", "",
             "## 1. 主表（LOAO top-1）", "",
             "| 臂 | 扣除方式 | top-1 均值 | vs raw 配对置换 p |", "|---|---|---|---|"]
    dsc = {"raw": "不扣除（= exp08 baseline）", "sub": "扣除基质均值谱",
           "sub_fit": "逐条谱拟合非负 α 后扣除"}
    for a in arms:
        p = "—" if a == "raw" else f"{p_raw[a]:.4f}"
        lines.append(f"| **{a}** | {dsc[a]} | {mean_top1[a]:.3f} | {p} |")

    lines += ["", f"chance = {chance:.3f}。", "",
              "## 2. 按实体类型分组（核心假设：掺混型受益，认证型不受害）", "",
              "| 类型 | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for k, v in grp.items():
        lines.append(f"| {k} | {len([e for e in eids if kinds[e] == k])} | " +
                     " | ".join(f"{v[a]:.3f}" for a in arms) + " |")

    lines += ["", "## 3. 逐折明细（top-1，seed 均值）", "",
              "| fold | 类型 | " + " | ".join(arms) + " |", "|---|" + "---|" * (len(arms) + 1)]
    for e in eids:
        lines.append(f"| {e} | {kinds[e]} | " +
                     " | ".join(f"{fold_top1[a][e]:.3f}" for a in arms) + " |")

    lines += ["", "## 4. 诊断：扣除前后「实体谱 vs 自身基质谱」余弦（应下降＝基质被剥离）", "",
              "| eid | kind | " + " | ".join(arms) + " |", "|---|" + "---|" * (len(arms) + 1)]
    for e in eids:
        cells = []
        for a in arms:
            c = diag[a][e][2] if e in diag[a] else float("nan")
            cells.append(f"{c:.4f}" if np.isfinite(c) else "—")
        lines.append(f"| {e} | {kinds[e]} | " + " | ".join(cells) + " |")

    best = max(arms, key=lambda a: mean_top1[a])
    lines += ["", "## 5. 结论", "",
              f"- 最佳臂：**{best}**（top-1 = {mean_top1[best]:.3f}，raw = {mean_top1['raw']:.3f}）"]
    if best != "raw":
        lines.append(f"- 相对 raw 提升 {mean_top1[best] - mean_top1['raw']:+.3f}，"
                     f"配对置换 p={p_raw[best]:.4f}")
        if "adulteration" in grp:
            g = grp["adulteration"][best] - grp["adulteration"]["raw"]
            lines.append(f"- **掺混型子集提升 {g:+.3f}**（这是假设检验的核心：提升应集中在掺混型）")
    lines.append("- 若提升集中在 adulteration 型而 authentication 型基本不变，"
                 "则「基质支配→差谱剥离」的机理得到证实，可作为论文的机制性发现")

    md = os.path.join(OUT, f"exp12_p6_matrix_subtraction{'_' + tag if tag else ''}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT, f"exp12_p6_matrix_subtraction{'_' + tag if tag else ''}.csv"),
              "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "eid", "seed", "top1"])
        for a in arms:
            for (e, s), v in results[a].items():
                w.writerow([a, e, s, v])
    print("\n".join(lines), flush=True)
    print(f"[written] {md}", flush=True)


if __name__ == "__main__":
    main()
