#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp28 Part C —— 部署杀招：新批次**真品**的假阳性率

Part A 显示：新批次下有监督 ML 的 AUC 完全不降（0.93–0.96）。
这暴露了更本质的问题——它们判的是「与历史真品库的差异」，
于是**一批新批次的真品样品**同样会被判成掺假。

本实验：模型用**旧批次**拟合（未平移），然后对
  (a) 新批次真品  B + s·σ_null·δ̂   → 假阳性率 FPR_new
  (b) 新批次掺假  X + s·σ_null·δ̂   → 命中率
打分；阈值仍取旧真品库的 α=0.01 分位（实验室的既有做法）。

预注册：
  H28f  新批次真品的 FPR 显著高于 α=0.01（对多数方法成立）
  H28g  nas 的 FPR_new 可由 Δz_off = s·⟨δ̂,û_b⟩ 解析预测：
        FPR = 1 − Φ(z_α − Δz_off)；有监督方法无解析预测
  H28h  NAS-C 校正后，nas 的 FPR_new 回到 ≈ α
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1                      # noqa: E402
import exp18_p6_nas_lod as E18               # noqa: E402
import exp22_p6_batch_correction as E22      # noqa: E402
import exp28_p6_deploy_and_whiten as E28     # noqa: E402

TARGETS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "HSYRUP"]
NFOLD = 5
ALPHA = 0.01
SEED = 2026
METHODS = ["plsda", "plsr", "svm", "rf", "lr", "ocsvm", "hotelling"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
RES_JSON = os.path.join(OUT_DIR, "exp28c_res.json")
OUT_MD = os.path.join(OUT_DIR, "exp28c_newbatch_falsealarm.md")


def nas_fold(B, X, c, pr, s_vec, ub, corr="none"):
    """nas 在给定平移下的 (FPR_new, TPR)；corr=nasc 时做 NAS-C。"""
    mu_m = pr["mu_m"]
    folds = E22.fold_split(c, NFOLD, SEED)
    a0_full, b_full = E18.lstsq_resp(X, c)
    bs_full = E18.resid_vec(b_full, pr["V"])
    ub_full = bs_full / max(np.linalg.norm(bs_full), 1e-12)
    fprs, tps, cs = [], [], []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(c)), f)
        a0, b = E18.lstsq_resp(X[tr], c[tr])
        bs = E18.resid_vec(b, pr["V"])
        nb = float(np.linalg.norm(bs))
        ub_f = bs / max(nb, 1e-12)
        tn = E22.null_scores(B, b, pr)
        sd0 = float(tn.std(ddof=1))
        thr = float(np.quantile(tn, 1 - ALPHA))
        if corr == "nasc":
            # 校准折同样被新批次平移（真实部署：整批都偏）
            a0c, _ = E18.lstsq_resp(X[tr] + s_vec[None, :], c[tr])
            zhat = float((a0c - mu_m) @ ub_f) / max(sd0, 1e-12)
            dz = -zhat
        else:
            dz = 0.0
        # 新批次真品
        tB = (B + s_vec[None, :] - mu_m) @ ub_f + dz * sd0
        # 新批次掺假
        tX = (X[f] + s_vec[None, :] - mu_m) @ ub_f + dz * sd0
        fprs.append(float(np.mean(tB > thr)))
        tps.append(tX > thr)
        cs.append(c[f])
    tp = np.concatenate(tps); cc = np.concatenate(cs)
    tpr_c = {float(l): float(np.mean(tp[np.isclose(cc, l)])) for l in np.unique(cc)}
    return dict(fpr_new=float(np.mean(fprs)), tpr_c=tpr_c,
                auc_new=float(np.mean([E18.auc_of(
                    (X[f] + s_vec[None, :] - mu_m) @ ub_f + dz * 0,
                    (B + s_vec[None, :] - mu_m) @ ub_f) for f in folds])),
                dz_pred=float(s_vec @ ub_full / max(
                    float(E22.null_scores(B, b_full, pr).std(ddof=1)), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--shifts", default="-4,-2,2,4")
    ap.add_argument("--scale", default="sigma",
                    help="sigma: 以 σ_null 为单位；real: 以实测批次偏移的光谱量级 "
                         "‖a0−μ_m‖ 为单位（真实量级）")
    ap.add_argument("--from-json", default="")
    args = ap.parse_args()

    if args.from_json and os.path.exists(args.from_json):
        report(json.load(open(args.from_json, encoding="utf-8")))
        return

    ent, conc, bg, grid = G1.load_g1_conc()
    targets = [e for e in TARGETS if e in ent]
    if args.only:
        targets = [e.strip() for e in args.only.split(",") if e.strip() in ent]
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]

    RES = {"targets": targets, "shifts": shifts, "res": {}, "zoff": {}, "donor": {}}
    for e in targets:
        mid = G1.ENTITY_META[e][1]
        B = np.asarray(bg[mid], float)
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float)
        k = c < 100
        X, c = X[k], c[k] / 100.0
        pr = E22.prep(B)
        dvec, kind = E28.donor_direction(ent, conc, bg, e, targets)
        RES["donor"][e] = kind
        if dvec is None:
            continue
        # 实测批次偏移的光谱量级（未归一化）
        mid_e = G1.ENTITY_META[e][1]
        sibs = [x for x in targets if x != e and x in ent
                and G1.ENTITY_META[x][1] == mid_e]
        pool = sibs if sibs else [x for x in targets if x != e and x in ent]
        raw = []
        for x in pool:
            m2 = G1.ENTITY_META[x][1]
            X2 = np.asarray(ent[x], float)
            c2 = np.asarray(conc[x], float)
            k2 = c2 < 100
            a0x, _ = E18.lstsq_resp(X2[k2], c2[k2] / 100.0)
            raw.append(a0x - np.asarray(bg[m2], float).mean(0))
        dmag = float(np.linalg.norm(np.mean(raw, axis=0)))
        RES["donor"][e] = kind + f"（‖δ‖={dmag:.3g}）"
        a0, b = E18.lstsq_resp(X, c)
        bs = E18.resid_vec(b, pr["V"])
        ub = bs / max(np.linalg.norm(bs), 1e-12)
        sd0 = float(E22.null_scores(B, b, pr).std(ddof=1))
        RES["zoff"][e] = float((a0 - pr["mu_m"]) @ ub / max(sd0, 1e-12))
        RES["res"][e] = {}
        for s in shifts:
            sv = s * sd0 * dvec if args.scale == "sigma" else s * dmag * dvec
            RES["res"][e][f"s{s:+.0f}"] = {}
            # ---- nas / nas+NAS-C ----
            for tag, corr in [("nas", "none"), ("nas_nasc", "nasc")]:
                try:
                    RES["res"][e][f"s{s:+.0f}"][tag] = nas_fold(
                        B, X, c, pr, sv, ub, corr=corr)
                except Exception as ex:                        # noqa: BLE001
                    RES["res"][e][f"s{s:+.0f}"][tag] = {"error": str(ex)}
            # ---- ML：每折只拟合一次，多次平移复用 ----
            folds = E22.fold_split(c, NFOLD, SEED)
            for m in METHODS:
                fprs, aucs = [], []
                for f in folds:
                    tr = np.setdiff1d(np.arange(len(c)), f)
                    try:
                        _sp, sn, fn = E28.ml_fit_score(m, X[tr], c[tr], B, X[f], SEED)
                        if sn is None:
                            sn = E28.ml_fit_score(m, X[tr], c[tr], B, B, SEED)[0]
                        thr = float(np.quantile(sn, 1 - ALPHA))
                        sB = fn(B + sv[None, :])
                        sX = fn(X[f] + sv[None, :])
                        fprs.append(float(np.mean(sB > thr)))
                        aucs.append(float(E18.auc_of(sX, sB)))
                    except Exception:                          # noqa: BLE001
                        pass
                RES["res"][e][f"s{s:+.0f}"][m] = dict(
                    fpr_new=float(np.mean(fprs)) if fprs else None,
                    auc_new=float(np.mean(aucs)) if aucs else None)
            print(f"  [C/{e} s={s:+.0f}] ok", flush=True)
        json.dump(RES, open(RES_JSON, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    json.dump(RES, open(RES_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report(RES)


def report(RES):
    A = []
    add = A.append
    targets = RES["targets"]
    shifts = RES["shifts"]
    ms = ["nas", "nas_nasc"] + METHODS
    add("# exp28 Part C —— 新批次**真品**的假阳性率（部署杀招）")
    add("")
    add("模型用**旧批次**拟合；阈值取旧真品库 α=0.01 分位。")
    add("然后把真品库与掺假系列**一起**平移 `s·σ_null·δ̂`，"
        "模拟「一批新批次的样品送进实验室」。")
    add("")
    add("## 1. 新批次真品的假阳性率 FPR_new（α = 0.01）")
    add("")
    add("| s·σ_null | " + " | ".join(ms) + " |")
    add("|---" * (len(ms) + 1) + "|")
    for s in shifts:
        key = f"s{s:+.0f}"
        row = [f"| {s:+.0f} |"]
        for m in ms:
            v = [RES["res"][e][key].get(m, {}).get("fpr_new") for e in targets
                 if isinstance(RES["res"].get(e, {}).get(key, {}).get(m), dict)]
            v = [x for x in v if x is not None]
            row.append(f"**{np.mean(v):.3f}**" if v else "—")
        add(" | ".join(row) + " |")
    add("")
    add("## 2. 新批次真品 vs 新批次掺假：AUC_new")
    add("")
    add("| s·σ_null | " + " | ".join(ms) + " |")
    add("|---" * (len(ms) + 1) + "|")
    for s in shifts:
        key = f"s{s:+.0f}"
        row = [f"| {s:+.0f} |"]
        for m in ms:
            v = [RES["res"][e][key].get(m, {}).get("auc_new") for e in targets
                 if isinstance(RES["res"].get(e, {}).get(key, {}).get(m), dict)]
            v = [x for x in v if x is not None]
            row.append(f"{np.mean(v):.3f}" if v else "—")
        add(" | ".join(row) + " |")
    add("")

    # H28f / H28g
    hi = {m: 0 for m in ms}
    tot = {m: 0 for m in ms}
    for e in targets:
        for s in shifts:
            key = f"s{s:+.0f}"
            for m in ms:
                v = RES["res"].get(e, {}).get(key, {}).get(m, {})
                if isinstance(v, dict) and v.get("fpr_new") is not None:
                    tot[m] += 1
                    hi[m] += int(v["fpr_new"] > 5 * ALPHA)
    add("## 3. 判决")
    add("")
    add("| 方法 | FPR_new > 5α 的组合 | 占比 |")
    add("|---|---|---|")
    for m in ms:
        if tot[m]:
            add(f"| {m} | {hi[m]}/{tot[m]} | {hi[m]/tot[m]*100:.0f}% |")
    add("")
    zz, ff, pp = [], [], []
    for e in targets:
        for s in shifts:
            key = f"s{s:+.0f}"
            v = RES["res"].get(e, {}).get(key, {}).get("nas", {})
            if isinstance(v, dict) and v.get("fpr_new") is not None:
                dz = v.get("dz_pred")
                if dz is not None:
                    za = float(stats.norm.ppf(1 - ALPHA))
                    pp.append(1 - stats.norm.cdf(za - dz))
                    ff.append(v["fpr_new"])
                    zz.append(dz)
    if len(ff) >= 6:
        rr = stats.spearmanr(pp, ff)
        add(f"**H28g**：nas 的 FPR_new 与解析预测 `1 − Φ(z_α − Δz_off)` 的 "
            f"Spearman ρ = **{rr.statistic:+.3f}**（p = {rr.pvalue:.4f}），"
            f"n = {len(ff)}")
        add("")
        add(f"- 实测 FPR_new 均值 **{np.mean(ff):.3f}**，解析预测均值 **{np.mean(pp):.3f}**，"
            f"平均绝对偏差 **{np.mean(np.abs(np.array(ff) - np.array(pp))):.3f}**")
    add("")
    os.makedirs(OUT_DIR, exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(A) + "\n")
    print(f"[exp28c] -> {OUT_MD}")


if __name__ == "__main__":
    main()
