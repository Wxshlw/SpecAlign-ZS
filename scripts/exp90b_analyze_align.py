# -*- coding: utf-8 -*-
"""exp90b —— 对齐消融分析：固定码本、只换对齐，杠杆有多大？

配对单元 = (tag, 候选对) for auc_dis；(tag, 候选) for top-1。
逐格与同一码本的 `lin` 基线配对 t 检验（df = 单元数−1）。

三件事：
  1. 每个臂的 Δ vs lin，看哪些臂真的能改变决策
  2. 按 A/B/C 三类汇总（标定 / 竞争 / 几何）
  3. 码本不变性：同一臂在 clm / physchem / ecfp2048 上的 Δ 是否同号同量级
     —— 若一致，则「对齐是码本无关的通用组件」成立（这是它作为方法创新的资格）
"""
from __future__ import annotations
import json
import os
import itertools

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")

ARMS = ["lin", "cq", "zq", "iqr", "margin", "zC", "cosC", "mf"]
GROUP = {"lin": "A 标定", "cq": "A 标定", "zq": "A 标定", "iqr": "A 标定",
         "zC": "A 标定", "margin": "B 竞争", "cosC": "C 几何", "mf": "C 几何"}
DESC = {
    "lin": "原始分数（基线）",
    "cq": "减 query 内均值",
    "zq": "除 query 内标准差",
    "iqr": "median/IQR 重标定",
    "zC": "老类分布标定 (s−μ_c)/σ_c",
    "margin": "m_c = s_c − max_{j≠c} s_j",
    "cosC": "候选方向归一化 w/‖w‖",
    "mf": "白化匹配滤波 Σ⁻¹",
}


def load(nhu):
    with open(os.path.join(OUT, f"exp90_align_nhu{nhu}.json"), encoding="utf-8") as f:
        return json.load(f)


def cells(d, prior, arm, kind):
    """返回 {(key,tag): value}。"""
    b = d["res"][prior][arm][kind]
    rk = "auc_dis" if kind == "dis" else "top1"
    out = {}
    for key, recs in b.items():
        for r in recs:
            out[(key, r["tag"])] = r[rk]
    return out


def compare(d, prior, arm, kind):
    base = cells(d, prior, "lin", kind)
    cur = cells(d, prior, arm, kind)
    keys = sorted(set(base) & set(cur))
    a = np.array([base[k] for k in keys], float)
    b = np.array([cur[k] for k in keys], float)
    if len(keys) < 3 or np.allclose(a, b):
        return dict(n=len(keys), delta=float((b - a).mean()) if len(keys) else np.nan,
                    p=float("nan") if len(keys) < 3 else 1.0,
                    dz=float("nan"), maxabs=float(np.max(np.abs(b - a))) if len(keys) else np.nan)
    t, p_ = stats.ttest_rel(b, a)
    dd = b - a
    return dict(n=len(keys), delta=float(dd.mean()), p=float(p_),
                dz=float(dd.mean() / (dd.std(ddof=1) + 1e-12)),
                maxabs=float(np.max(np.abs(dd))))


def main():
    md = ["# exp90 · 对齐消融：固定码本，只换「怎么比」", "",
          "> 镜像实验：exp82 轴 A 是「固定对齐、换码本」；本实验是「固定码本、换对齐」。",
          "> 两者相减 = 「创新应归到哪一层」的定量依据。",
          "> 口径：定向归因（T、D 均未见），λ=1e-4，group 分组，3 seeds × 4 reps。", ""]
    summary = {}

    for nhu in (3, 5):
        d = load(nhu)
        priors = d["meta"]["priors"]
        summary[nhu] = {}
        md.append(f"## |U|={nhu}")
        md.append("")
        for kind, kname in [("dis", "auc_dis（成对判别）"), ("top1", "top-1（候选集 argmax）")]:
            md.append(f"### {kname}")
            md.append("")
            hdr = "| 类别 | 臂 | 说明 | " + " | ".join(priors) + " | 码本同号 |"
            sep = "|---|---|---|" + "---|" * len(priors) + "---|"
            md.append(hdr)
            md.append(sep)
            for arm in ARMS:
                cells_s = []
                signs = []
                rec = {}
                for p in priors:
                    c = compare(d, p, arm, kind)
                    rec[p] = c
                    if np.isnan(c["delta"]):
                        cells_s.append("—")
                        continue
                    if abs(c["delta"]) < 1e-9:
                        cells_s.append("≡ 0")
                    else:
                        star = "*" if c["p"] < 0.05 else ""
                        cells_s.append(f"{c['delta']:+.4f}{star} ({c['p']:.3f})")
                    if abs(c["delta"]) > 1e-9:
                        signs.append(np.sign(c["delta"]))
                same = "✅" if len(signs) == len(priors) and len(set(signs)) == 1 else \
                       ("—" if not signs else "❌")
                md.append(f"| {GROUP[arm]} | `{arm}` | {DESC[arm]} | "
                          + " | ".join(cells_s) + f" | {same} |")
                summary[nhu][f"{kind}|{arm}"] = rec
            md.append("")
            md.append("> Δ 相对同一码本的 `lin` 基线；* = p<0.05。`≡ 0` 表示与基线逐位相同。")
            md.append("")

    # ---- 杠杆对比：换码本 vs 换对齐 ----
    md.append("## 杠杆对比：换码本 vs 换对齐")
    md.append("")
    for kind, kname in [("dis", "auc_dis"), ("top1", "top-1")]:
        md.append(f"**{kname}**")
        md.append("")
        md.append("| 来源 | 幅度 | 显著性 | 数据 |")
        md.append("|---|---|---|---|")
        for nhu in (3, 5):
            s = summary[nhu]
            vals = []
            for p in ["clm", "physchem", "ecfp2048"]:
                c = s.get(f"{kind}|lin", {}).get(p)
                vals.append(c["delta"] if c else np.nan)
            # 换先验：同臂（lin）下三个码本之间的散布 = 上界
            md.append(f"| 换码本（|U|={nhu}，同 arm=lin） | 见 exp89：0/15 显著 | — | exp89 |")
            rows = []
            for arm in ARMS:
                if arm == "lin":
                    continue
                ds = [s[f"{kind}|{arm}"][p]["delta"] for p in s[f"{kind}|{arm}"]]
                ps = [s[f"{kind}|{arm}"][p]["p"] for p in s[f"{kind}|{arm}"]]
                ds = [x for x in ds if not np.isnan(x)]
                if not ds:
                    continue
                rows.append((arm, float(np.mean(ds)), float(np.min(ps))))
            rows.sort(key=lambda r: -abs(r[1]))
            for arm, dm, pm in rows[:3]:
                md.append(f"| 换对齐 `{arm}`（|U|={nhu}，3 码本均值） | {dm:+.4f} | 最小 p={pm:.3f} | exp90 |")
        md.append("")

    with open(os.path.join(OUT, "exp90_align.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, "exp90_align_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
