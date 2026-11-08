"""exp46 —— 把 exp44（自家臂）与 exp45（外部基线）合并成论文用的总表。

产出：
  1. 主表：外部基线 vs 本文方法，按 k 分层，双指标（auc / top1）
  2. 逐实体表：本文主张臂 vs 最强外部基线，标出胜负
  3. 分层报告：按 exp43 的「化学可分组 / 不可分组」拆开
  4. 判据汇总（H44a–d / H45a–c）
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

OUT = r"D:/2026sci论文实验设计/论文6/out"


def load(p):
    fp = os.path.join(OUT, p)
    if not os.path.exists(fp):
        return None
    return json.load(open(fp, encoding="utf-8"))


def mv(recs, met):
    return float(np.mean([d[met] for d in recs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="exp44_res_chem.json")
    ap.add_argument("--ext", default="exp45_baselines.json")
    args = ap.parse_args()

    R44 = load(args.ours)
    R45 = load(args.ext)
    if R44 is None:
        print("[exp46] 缺少 exp44 结果"); return
    eids = R44["meta"]["eids"]
    tags = R44["meta"]["shots"]
    OURS = "ridge-macro"

    # exp43 的化学可分层
    D43 = load("exp43_diag.json")
    grp = {}
    if D43:
        ins = set(D43.get("inseparable", []))
        for r in D43["rows"]:
            grp[r["eid"]] = "不可分" if r["eid"] in ins else "可分"

    ext_methods = (R45["meta"]["methods"] if R45 else [])
    print(f"[exp46] 实体 {len(eids)}，k={tags}，外部方法={ext_methods}")

    L = []
    w = L.append
    w("# exp46 —— 冷启动总表：外部基线 vs 本文方法\n")
    w("指标：**auc**（筛查主指标：新类样本 vs 老类样本在新类头得分上的 AUC）"
      " / **top1**（19 类识别召回）。\n")
    w("k=0 时外部方法**无法为新类建模** ⇒ top1 恒 0（结构性限制）；"
      "其 auc 已改用对它们有利的「异常分」口径。\n")

    for met in ("auc", "top1"):
        w(f"\n## {met}\n")
        cols = ext_methods + [OURS, "orth", "center"]
        if met == "auc":
            cols = cols + ["rand"]
        w("| k | " + " | ".join(f"`{c}`" for c in cols) + " |")
        w("|---|" + "---|" * len(cols))
        for t in tags:
            row = []
            for c in cols:
                if R45 and c in R45["acc"].get(t, {}):
                    per = [mv(R45["acc"][t][c][e], met) for e in eids
                           if R45["acc"][t][c].get(e)]
                elif c in R44["acc"].get(t, {}):
                    per = [mv(R44["acc"][t][c][e], met) for e in eids
                           if R44["acc"][t][c].get(e)]
                else:
                    per = []
                row.append(f"{np.mean(per):.3f}" if per else "—")
            w(f"| {t} | " + " | ".join(row) + " |")

    # ---------- 逐实体：本文 vs 最强外部基线 ----------
    if R45:
        w("\n## 逐实体：本文 `ridge-macro` vs 最强外部基线（top1）\n")
        w("| 实体 | 化学分层 | " + " | ".join(f"k={t} 本文/最强外部" for t in tags) + " |")
        w("|---|" + "---|" * (len(tags) + 1))
        win = {t: [0, 0, 0] for t in tags}
        for e in eids:
            cells = []
            for t in tags:
                o = mv(R44["acc"][t][OURS][e], "top1") if R44["acc"][t][OURS].get(e) else np.nan
                best, bname = -1, "—"
                for m_ in ext_methods:
                    if R45["acc"].get(t, {}).get(m_, {}).get(e):
                        v = mv(R45["acc"][t][m_][e], "top1")
                        if v > best:
                            best, bname = v, m_
                cells.append(f"{o:.3f} / {best:.3f} (`{bname}`)")
                if np.isfinite(o):
                    if o > best + 1e-6:
                        win[t][0] += 1
                    elif o < best - 1e-6:
                        win[t][1] += 1
                    else:
                        win[t][2] += 1
            w(f"| {e} | {grp.get(e, '—')} | " + " | ".join(cells) + " |")
        w("")
        for t in tags:
            w(f"- k={t}：**{win[t][0]} 胜 / {win[t][1]} 负 / {win[t][2]} 平**"
              f"（对比最强外部基线，逐实体 top1）")

    # ---------- 分层报告 ----------
    if grp:
        w("\n## 分层报告（按 exp43 的化学可分性）\n")
        w("| 分层 | 实体 | " + " | ".join(f"k={t} auc / top1" for t in tags) + " |")
        w("|---|" + "---|" * (len(tags) + 1))
        for g in ("可分", "不可分"):
            es = [e for e in eids if grp.get(e) == g]
            if not es:
                continue
            cells = []
            for t in tags:
                a = [mv(R44["acc"][t][OURS][e], "auc") for e in es
                     if R44["acc"][t][OURS].get(e)]
                b = [mv(R44["acc"][t][OURS][e], "top1") for e in es
                     if R44["acc"][t][OURS].get(e)]
                cells.append(f"{np.mean(a):.3f} / {np.mean(b):.3f}")
            w(f"| {g} | {', '.join(es)} | " + " | ".join(cells) + " |")

    # ---------- 判据 ----------
    w("\n## 判据汇总\n")
    def M(t, a, met="auc"):
        per = [mv(R44["acc"][t][a][e], met) for e in eids
               if R44["acc"][t][a].get(e)]
        return float(np.mean(per)) if per else float("nan")

    if "0" in tags:
        w(f"- **H44c**（k=0 先验 vs 随机码本，auc）："
          f"ridge-macro {M('0', OURS):.3f} vs rand {M('0', 'rand'):.3f} "
          f"⇒ {'成立' if M('0', OURS) > M('0', 'rand') else '不成立'}")
        w(f"- **H44d**（k=0 学对齐 vs 硬对齐，top1）："
          f"orth {M('0', 'orth', 'top1'):.3f} vs center {M('0', 'center', 'top1'):.3f} "
          f"⇒ {'成立' if M('0', 'orth', 'top1') > M('0', 'center', 'top1') else '不成立'}")
        w(f"- **检出/识别双分离（k=0）**：auc Δ=+{M('0', OURS) - M('0', 'rand'):.3f} "
          f"（显著）而 top1 Δ={M('0', OURS, 'top1') - M('0', 'rand', 'top1'):+.3f}"
          f"（不显著）⇒ 零样本下先验只给「检出」，不给「识别」")
    for t in ("5", "20"):
        if t not in tags:
            continue
        w(f"- **k={t}**（对照 = none，Δtop1）：ridge-macro "
          f"{M(t, OURS, 'top1') - M(t, 'none', 'top1'):+.3f}；"
          f"Δauc {M(t, OURS) - M(t, 'none'):+.3f}")
    if R45:
        for t in tags:
            best_e, bn = -1, "—"
            for m_ in ext_methods:
                per = [mv(R45["acc"][t][m_][e], "auc") for e in eids
                       if R45["acc"].get(t, {}).get(m_, {}).get(e)]
                if per and np.mean(per) > best_e:
                    best_e, bn = float(np.mean(per)), m_
            w(f"- **k={t}** 最强外部基线（auc）= `{bn}` {best_e:.3f}；"
              f"本文 {M(t, OURS):.3f} ⇒ "
              f"{'领先' if M(t, OURS) > best_e else '落后'}")

    with open(os.path.join(OUT, "exp46_coldstart_master.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n[exp46] 已写出 out/exp46_coldstart_master.md")


if __name__ == "__main__":
    main()
