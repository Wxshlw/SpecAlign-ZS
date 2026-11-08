"""exp93b —— 分析组成分辨率消融

三个判定
--------
J1 有效秩是不是机制：`rand384` / `onehot`（各向同性，参与率 17-18）截断到 r≈2-3 后
   能不能被"救活"到结构类水平？
   - 能 ⇒ 机制纯粹是「候选间协方差不均匀」，化学内容零贡献（对 claim 是坏消息，但必须报）
   - 不能 ⇒ 低秩只是必要条件，"低秩方向要对"才是化学的作用

J2 截断斜率是否与先验类型有关（机制的有无）：
   若所有先验同斜率 ⇒ 只是"砍维度越狠越差"，无机制。

J3 physchem 的组成块归因：elem(13 维元素计数) vs desc(12 维理化描述符) vs full。
"""
from __future__ import annotations
import os, sys, json, itertools
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def load(fn):
    with open(os.path.join(OUT, fn), encoding="utf-8") as f:
        return json.load(f)


def cells(d, arm, kind):
    """{key: [records]} → {(key, tag): value}"""
    blk = d.get(arm, {}).get(kind, {})
    out = {}
    fld = "auc_dis" if kind == "dis" else "top1"
    for k, recs in blk.items():
        for r in recs:
            v = r.get(fld)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out[(k, r["tag"])] = float(v)
    return out


def align(d, arms, kind):
    D = {a: cells(d, a, kind) for a in arms}
    keys = sorted(set.intersection(*[set(D[a]) for a in arms]))
    return keys, {a: np.array([D[a][k] for k in keys]) for a in arms}


def main():
    lines = []
    P = lines.append
    base_pri = ["clm", "physchem", "charbag", "ecfp2048"]
    iso_pri = ["rand384", "onehot"]
    ranks = ["r1", "r2", "r3", "r4", "r5", "r8", "rfull"]

    for nhu in (3, 5):
        fn = f"exp93_comp_nhu{nhu}.json"
        if not os.path.exists(os.path.join(OUT, fn)):
            print(f"[skip] {fn} 不存在")
            continue
        d = load(fn)
        meta = d["META"]
        arms = [a for a in d if a != "META"]
        pr = meta["rank_info"]

        P(f"# exp93 组成分辨率消融 · |U|={nhu}")
        P("")
        P(f"seeds={meta['seeds']} reps={meta['nrep']} λ={meta['lam']}")
        P("")

        # ---------- 表 1：秩截断曲线 ----------
        for kind, lab in (("dis", "auc_dis"), ("top1", "top-1")):
            P(f"## 表1 · {lab}（行=先验，列=候选间保留秩）")
            P("")
            hdr = "| 先验 | 参与率 | " + " | ".join(
                r.replace("r", "r=").replace("full", "full") for r in ranks) + " | 斜率(每维) |"
            P(hdr)
            P("|---|---|" + "---|" * (len(ranks) + 1))
            for p in base_pri + iso_pri:
                row, xs, ys = [], [], []
                for r in ranks:
                    a = f"{p}|{r}"
                    if a not in d:
                        row.append("—")
                        continue
                    v = [rec.get("auc_dis" if kind == "dis" else "top1")
                         for recs in d[a].get(kind, {}).values() for rec in recs]
                    v = [x for x in v if x is not None]
                    m = float(np.mean(v)) if v else np.nan
                    row.append(f"{m:.3f}")
                    if r != "rfull" and not np.isnan(m):
                        xs.append(int(r[1:]))
                        ys.append(m)
                if len(xs) >= 3:
                    sl = np.polyfit(xs, ys, 1)[0]
                    sl_s = f"{sl:+.4f}"
                else:
                    sl_s = "—"
                prs = pr.get(p, {}).get("pr", np.nan)
                P(f"| {p} | {prs:.2f} | " + " | ".join(row) + f" | {sl_s} |")
            P("")

        # ---------- J1：各向同性对照能否被救活 ----------
        P("## J1 · 各向同性对照（rand384 / onehot）截断后能否救活")
        P("")
        keys, V = align(d, [f"{p}|{r}" for p in base_pri + iso_pri for r in ranks
                            if f"{p}|{r}" in d], "dis")
        P(f"配对单元 n = {len(keys)}（候选对 × 重复）")
        P("")
        P("| 秩 | " + " | ".join(base_pri + iso_pri) + " | 结构类均值−rand |")
        P("|---|" + "---|" * (len(base_pri + iso_pri) + 1))
        for r in ranks:
            row = []
            for p in base_pri + iso_pri:
                a = f"{p}|{r}"
                row.append(f"{V[a].mean():.3f}" if a in V else "—")
            sm = np.mean([V[f"{p}|{r}"] for p in base_pri if f"{p}|{r}" in V], axis=0)
            if f"rand384|{r}" in V:
                dd = sm - V[f"rand384|{r}"]
                t, pv = stats.ttest_rel(sm, V[f"rand384|{r}"])
                row.append(f"{dd.mean():+.3f} (p={pv:.3f})")
            else:
                row.append("—")
            P(f"| {r} | " + " | ".join(row) + " |")
        P("")

        # ---------- J2：斜率是否与先验类型有关 ----------
        P("## J2 · 截断斜率（r=1..8 线性拟合，含 full 作对照）")
        P("")
        for p in base_pri + iso_pri:
            xs, ys = [], []
            for r in ranks[:-1]:
                a = f"{p}|{r}"
                if a not in d:
                    continue
                v = [rec["auc_dis"] for recs in d[a].get("dis", {}).values() for rec in recs]
                if v:
                    xs.append(int(r[1:]))
                    ys.append(float(np.mean(v)))
            if len(xs) >= 3:
                k, b = np.polyfit(xs, ys, 1)
                P(f"- **{p}**（参与率 {pr.get(p, {}).get('pr', np.nan):.2f}）："
                  f"斜率 {k:+.4f}/维，r=1 时 {ys[0]:.3f}，r=8 时 {ys[-1]:.3f}")
        P("")

        # ---------- J3：physchem 块消融 ----------
        pv = [a for a in arms if a.startswith("physchem|") and a.split("|")[1] in ("elem", "desc")]
        if pv:
            P("## J3 · physchem 组成块归因（elem=13 元素计数 / desc=12 理化描述符）")
            P("")
            P("| 变体 | auc_dis | top-1 |")
            P("|---|---|---|")
            for a in sorted(pv) + ["physchem|rfull"]:
                if a not in d:
                    continue
                dv = [r["auc_dis"] for recs in d[a].get("dis", {}).values() for r in recs]
                tv = [r["top1"] for recs in d[a].get("top1", {}).values() for r in recs]
                P(f"| {a.split('|')[1]} | {np.mean(dv):.4f} | {np.mean(tv):.4f} |")
            P("")

    txt = "\n".join(lines)
    path = os.path.join(OUT, "exp93_comp_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[exp93b] 写入 {path}")


if __name__ == "__main__":
    main()
