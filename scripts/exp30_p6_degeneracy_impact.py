# -*- coding: utf-8 -*-
"""
exp30 — 候选库表示退化的影响盘点（Degeneracy Impact Audit）

触发：exp29 发现 CLM 的「独家命中」56 个中有 52 个来自 A04 一折，
      怀疑该优势并非化学语义的普适价值，而是候选库构造缺陷造成的局部现象。

核心发现（本脚本固化）：
  1. 候选库存在**完全相同的分子表示**（exp00 已记录，但从未传达到 exp14/正文）：
       G1 = A01 ≡ A05 ≡ A13（亚油酸 C18H32O2）
       G2 = A02 ≡ A15      （油酸  C18H34O2）
       G3 = A04 ≡ A06      （麦芽糖 C12H22O11）
     在本实验的 13 候选池内 ⇒ **A01≡A05、A04≡A06**。
  2. 剔除退化折后 pooled top-1 从 0.350/0.368 **塌到 0.210/0.219**
     ⇒ 现报告的核心数字被退化折（A01/A05 等）虚高支撑。
  3. 剔除后呈**极端双峰**：9 折中 5 折 = 0.000、3 折 = 1.000
     ⇒ pooled 均值不是合适的总结统计量，「全有或全无」才是真实形态。
  4. A04 的 CLM 优势 = 在 A03/A04 这对「同为 C12H22O11 多糖」近邻间的有限分辨率
     （CLM 57/300 vs ECFP 5/300），**不足以支撑「化学语义」招牌**。

输出：out/exp30_p6_degeneracy_impact.md / exp30_res.json
"""

import os
import json
import collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

DEGEN = ["A01", "A05", "A04", "A06"]      # 候选池内 Tanimoto = 1.0000 的组
POLY = ["A03"]                            # 同为 C12H22O11 多糖，近邻混淆源（非严格退化）
ARMS = ["Original", "Descriptor", "Shuffle-Random", "Shuffle-Mismatch", "Random-Encoder"]


def top1(arm, S, folds, eids):
    hits = {}
    for e in folds:
        i = eids.index(e)
        sim = np.asarray(S[arm][e], float)
        hits[e] = (sim.argmax(axis=1) == i)
    pooled = float(np.concatenate([hits[e] for e in folds]).mean())
    n = int(sum(len(hits[e]) for e in folds))
    return {e: float(hits[e].mean()) for e in folds}, pooled, n


def main():
    S = np.load(os.path.join(OUT, "exp08_sims.npy"), allow_pickle=True).item()
    eids = sorted(S["Original"].keys())
    deg = json.load(open(os.path.join(OUT, "exp00_p6_degeneracy.json"), encoding="utf-8"))
    groups = deg.get("identical_repr_groups", {})

    clean = [e for e in eids if e not in DEGEN]
    clean2 = [e for e in eids if e not in DEGEN + POLY]

    # ---------- A. 退化盘点 ----------
    rows = {}
    for a in ARMS:
        f_all, p_all, n_all = top1(a, S, eids, eids)
        f_c1, p_c1, n_c1 = top1(a, S, clean, eids)
        f_c2, p_c2, n_c2 = top1(a, S, clean2, eids)
        rows[a] = {"all": (p_all, n_all), "clean": (p_c1, n_c1), "clean2": (p_c2, n_c2),
                   "fold_all": f_all, "fold_clean": f_c1, "fold_clean2": f_c2}

    # ---------- B. 二值形态诊断 ----------
    o_clean = rows["Original"]["fold_clean"]
    d_clean = rows["Descriptor"]["fold_clean"]
    bins = {"=0.000": 0, "(0, 0.2]": 0, "(0.2, 0.8)": 0, "[0.8, 1.0)": 0, "=1.000": 0}

    def bincount(d):
        c = collections.Counter()
        for v in d.values():
            if v == 0.0:
                c["=0.000"] += 1
            elif v <= 0.2:
                c["(0, 0.2]"] += 1
            elif v < 0.8:
                c["(0.2, 0.8)"] += 1
            elif v < 1.0:
                c["[0.8, 1.0)"] += 1
            else:
                c["=1.000"] += 1
        return c

    bin_o, bin_d = bincount(o_clean), bincount(d_clean)

    # ---------- C. A04 混淆去向 ----------
    conf = {}
    for a in ["Original", "Descriptor"]:
        sim = np.asarray(S[a]["A04"], float)
        cnt = collections.Counter(sim.argmax(axis=1))
        conf[a] = {eids[k]: int(v) for k, v in cnt.most_common()}

    # ---------- 报告 ----------
    L = []
    A = L.append
    A("# exp30 — 候选库表示退化的影响盘点\n")
    A("> 触发：exp29 发现 CLM 的 56 个「独家命中」有 52 个来自 A04 一折，")
    A("> 追查其机制，结果发现的是一个**更严重的、且早有记录却未被传达的候选库缺陷**。\n")

    A("## 1. 候选库里存在完全相同的分子\n")
    A("`exp00_p6_degeneracy.json`（项目第 0 号实验）已经记录了完全相同的表示组：\n")
    A("| 组 | 成员 | 分子 | 是否在本实验 13 候选池内 |")
    A("|---|---|---|---|")
    names = {"G1": "亚油酸 C18H32O2", "G2": "油酸 C18H34O2", "G3": "麦芽糖 C12H22O11"}
    for g, mem in groups.items():
        inside = [m for m in mem if m in eids]
        if len(inside) >= 2:
            cell = "**" + " ≡ ".join(inside) + "**（退化成立）"
        elif len(inside) == 1:
            cell = f"仅 {inside[0]}（伙伴不在池内 ⇒ 不退化）"
        else:
            cell = "否"
        A(f"| {g} | {', '.join(mem)} | {names.get(g,'')} | {cell} |")
    A("")
    A("⇒ 本实验的 13 候选池内有两组严格退化：**A01 ≡ A05**（亚油酸）、**A04 ≡ A06**（麦芽糖）。")
    A("rdkit 复核（ECFP4/2048bit）：A01-A05 Tanimoto = **1.0000**，A04-A06 Tanimoto = **1.0000**；")
    A("开/关 `useChirality` 结果一致 ⇒ **不是立体化学问题，是同一 SMILES 被用了两次**。\n")
    A("⚠️ 该缺陷在 exp00 即已发现，但**从未进入 exp14 的结论与正文**。\n")

    A("## 2. 剔除退化折后，核心数字塌了多少\n")
    A("| 口径 | CLM (Original) | ECFP (Descriptor) | Shuffle-Random |")
    A("|---|---|---|---|")
    for key, label in [("all", "全 13 折（现报告值）"), ("clean", "剔除退化折 A01/A05/A04/A06"),
                       ("clean2", "再剔除多糖近邻 A03")]:
        A(f"| {label} | {rows['Original'][key][0]:.3f} (n={rows['Original'][key][1]}) | "
          f"{rows['Descriptor'][key][0]:.3f} (n={rows['Descriptor'][key][1]}) | "
          f"{rows['Shuffle-Random'][key][0]:.3f} |")
    A("")
    A(f"> CLM：0.350 → **{rows['Original']['clean'][0]:.3f}**；ECFP：0.368 → **{rows['Descriptor']['clean'][0]:.3f}**\n")
    A("**⇒ 现报告的 0.350 / 0.368 被退化折显著虚高支撑。** 这不是措辞问题，是数字本身要改。\n")

    A("## 3. 真实形态：全有或全无，不是连续能力谱\n")
    A("剔除退化后的 9 折逐折 top-1：\n")
    A("| fold | n | CLM | ECFP |")
    A("|---|---|---|---|")
    nq = {e: np.asarray(S["Original"][e], float).shape[0] for e in eids}
    for e in clean:
        A(f"| {e} | {nq[e]} | {o_clean[e]:.3f} | {d_clean[e]:.3f} |")
    A("")
    A("| 区间 | CLM 折数 | ECFP 折数 |")
    A("|---|---|---|")
    for k in bins:
        A(f"| {k} | {bin_o.get(k,0)} | {bin_d.get(k,0)} |")
    A("")
    A("**9 折中 5 折 = 0.000、3 折 = 1.000，中间态仅 1 折。**")
    A("⇒ **pooled 均值（0.210/0.219）是一个由 0 和 1 加权出来的数，不能被读成「21% 的情况下能认出来」。**")
    A("正文必须改用「N/13 个实体可被零样本锁定」的表述，并给出二值形态本身作为发现。\n")

    A("## 4. A04 的 CLM 优势到底是什么\n")
    A("A04（姜黄粉←小麦粉/麦芽糖）折上 argmax 的去向：\n")
    A("| 被投给的候选 | CLM | ECFP |")
    A("|---|---|---|")
    keys = sorted(set(list(conf["Original"].keys()) + list(conf["Descriptor"].keys())),
                  key=lambda k: -(conf["Original"].get(k, 0) + conf["Descriptor"].get(k, 0)))
    for k in keys[:8]:
        mark = " ✅ 正确" if k == "A04" else ""
        A(f"| {k}{mark} | {conf['Original'].get(k,0)} | {conf['Descriptor'].get(k,0)} |")
    A("")
    A("判读：\n")
    A("- 两者都把**最多票投给 A03**（木屑，同为 C12H22O11 多糖）⇒ 主要混淆源是**多糖近邻**，不是严格退化对；")
    A("- CLM 把 57/300 投给正确的 A04，ECFP 只有 5/300 ⇒ CLM 在 A03/A04 之间**确有一点点分辨率**；")
    A("- 但 ECFP 另有 118 票投给 A01（脂肪酸，完全跑偏）⇒ ECFP 在该折是灾难性的。\n")
    A("**结论：这是 1 个折、1 类结构（寡糖/多糖近邻）上的有限优势，不是化学语义的普适价值。**")
    A("不足以支撑「CLM 提供化学语义增量」的 claim；可作为 limitation 中「语义编码可能仅在局部子结构")
    A("高度重复的分子族上有价值」的**观察性线索**，但须明确标注 n=1。\n")

    A("## 5. 必须写进论文的披露清单\n")
    A("1. **候选库含 2 组完全相同的分子表示**（A01≡A05、A04≡A06），须在方法/附录中列出；")
    A("2. **核心 top-1 须同时报告全折与剔除退化折两个口径**，不得只报有利的一个；")
    A("3. **pooled 均值须改为「可被锁定的实体数 / 总数」**，并说明二值形态；")
    A("4. **CLM vs ECFP 须同时报告聚类 bootstrap（不显著）与 query 级 McNemar（ECFP 显著更优，p=2.2e-05）**；")
    A("5. A04 的 CLM 优势须标注为 **单折观察**，不得泛化为化学语义的证据。\n")

    A("## 6. 对「化学语义」招牌的最终判定\n")
    A("| 证据 | 方向 |")
    A("|---|---|")
    A("| exp14 CLM vs ECFP（聚类 bootstrap） | 不可区分 |")
    A("| exp29 query 级 McNemar | **ECFP 显著更优** |")
    A("| exp29 融合 / α 扫描 | **零互补，α*=0** |")
    A("| exp29 union oracle | 互补空间仅 +0.018 |")
    A("| exp30 A04 单折 | CLM 仅有局部优势，n=1 |")
    A("")
    A("**⇒ 「化学语义 / 化学语言模型」作为卖点，证据链已全面为负。走 B 路线：撤掉该招牌。**")

    md = "\n".join(L)
    p_md = os.path.join(OUT, "exp30_p6_degeneracy_impact.md")
    open(p_md, "w", encoding="utf-8").write(md)

    res = {"identical_groups": groups,
           "degen_in_pool": DEGEN, "poly_neighbor": POLY,
           "top1": {a: {"all": rows[a]["all"][0], "clean": rows[a]["clean"][0],
                        "clean2": rows[a]["clean2"][0],
                        "n_all": rows[a]["all"][1], "n_clean": rows[a]["clean"][1],
                        "fold_clean": rows[a]["fold_clean"]} for a in ARMS},
           "bimodal_bins": {"CLM": dict(bin_o), "ECFP": dict(bin_d)},
           "a04_confusion": conf}
    p_json = os.path.join(OUT, "exp30_res.json")
    json.dump(res, open(p_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(md)
    print(f"\n[saved] {p_md}\n[saved] {p_json}")


if __name__ == "__main__":
    main()
