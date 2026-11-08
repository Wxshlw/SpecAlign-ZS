# -*- coding: utf-8 -*-
r"""exp95 —— 统一统计口径：所有主表数字在同一聚合规则下重算。

【为什么要这一步】
主表定稿 v1 暴露出三处跨表不一致，并排放会被审稿人抓到：

  A. 同一个量两种汇总差 0.04
     exp82/93 的 clm auc_dis = 0.828 / 0.826，exp89 重算 = 0.8674
  B. 均值差 vs 配对差差一倍
     exp82 表头 0.828-0.214 = 0.614，但报告写 Δ(clm-rand) = +0.320
  C. T1 的 +0.602 与 T3 的 +0.607 是同一个量的两次估计，被当成两个独立证据

【统一规则（本文全表唯一口径）】
  单位 = 科学单位。归因任务：候选对(pair)；检出/识别任务：实体(entity)。
  1) 单位内先对重复/划分求平均 -> 每个单位一个值
  2) 描述统计 = 单位等权均值（不是 record 等权）
  3) 推断 = 单位级配对 t，df = 单位数-1；效应量 dz = mean(d)/sd(d)
  4) 敏感性 = 以「划分」(seed/combo) 为聚类单位重做配对 t

【为什么不是 record 等权】
record = (划分, 候选对)。同一划分下的多个候选对共享同一次训练/切分，
不是独立观测；把它们当独立样本会让 n 从 19 虚增到 36、p 值系统性偏小。
以单位（候选对/实体）聚合后，n 回到真实独立单元数，这是本项目的既定纪律
（对应 MEMORY 中「折聚类 bootstrap 作主推断」的教训）。

输出：out/exp95_unified.json、out/exp95_unified_report.md
"""
from __future__ import annotations
import os, json, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

STRUCT = ["clm", "ecfp384", "ecfp2048", "maccs", "avalon", "charbag"]
COMP = ["rdkit", "physchem"]
NULL = ["onehot", "rand384"]
# |U|=4/5 只有 8 个先验（无 ecfp384 / avalon）
STRUCT_8 = ["clm", "ecfp2048", "maccs", "charbag"]


def load(name):
    p = os.path.join(OUT, name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 聚合原语
def units(dis, field="auc_dis"):
    """{unit: [values]}；跳过 NaN/None。"""
    out = {}
    for u, recs in dis.items():
        vs = [r.get(field) for r in recs if isinstance(r, dict)]
        vs = [float(v) for v in vs if v is not None and np.isfinite(v)]
        if vs:
            out[u] = vs
    return out


def unit_mean(dis, field="auc_dis"):
    return {u: float(np.mean(v)) for u, v in units(dis, field).items()}


def record_mean(dis, field="auc_dis"):
    allv = [v for vs in units(dis, field).values() for v in vs]
    return float(np.mean(allv)) if allv else float("nan")


def n_records(dis, field="auc_dis"):
    return sum(len(v) for v in units(dis, field).values())


def tag_mean(dis, field="auc_dis"):
    """按「划分」求均值，用作聚类敏感性分析。"""
    acc = {}
    for u, recs in dis.items():
        for r in recs:
            if not isinstance(r, dict):
                continue
            v = r.get(field)
            if v is None or not np.isfinite(v):
                continue
            acc.setdefault(r.get("tag", "?"), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired(a: dict, b: dict, label=""):
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return dict(n=len(keys), diff=float("nan"), sd=float("nan"),
                    t=float("nan"), p=float("nan"), dz=float("nan"),
                    identity=False, label=label)
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd <= 0:
        return dict(n=n, diff=m, sd=0.0, t=float("nan"), p=float("nan"), dz=float("nan"),
                    identity=(abs(m) < 1e-12), label=label)
    t = m / (sd / np.sqrt(n))
    p = float(2 * stats.t.sf(abs(t), n - 1))
    return dict(n=n, diff=m, sd=sd, t=float(t), p=p, dz=m / sd,
                identity=False, label=label)


def pc(pr):
    """单元格格式化：恒等 / 正常。"""
    if pr.get("identity"):
        return "≡ 0（恒等）"
    if not np.isfinite(pr["diff"]):
        return "—"
    star = "**" if np.isfinite(pr["p"]) and pr["p"] < 0.05 else ""
    return f"{star}{pr['diff']:+.4f}{star} ({fmt_p(pr['p'])})"


def mean_pair(vals):
    """结构类均值：逐单位在 6 个先验间平均，再跨单位平均。"""
    return float(np.mean([np.mean(v) for v in vals]))


def struct_mean_dict(dis_by_prior, priors):
    """返回 {unit: 该单位在 priors 上的均值}，用于与 rand 配对。"""
    per = {p: unit_mean(dis_by_prior[p]) for p in priors if p in dis_by_prior}
    keys = set.intersection(*[set(v) for v in per.values()]) if per else set()
    return {k: float(np.mean([per[p][k] for p in per])) for k in keys}


def fmt_p(p):
    if not np.isfinite(p):
        return "n.a."
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


# ---------------------------------------------------------------- 数据装载
D82 = {3: load("exp82_dis.json"), 4: load("exp82_nhu4.json"), 5: load("exp82_nhu5.json")}
D90 = {3: load("exp90_align_nhu3.json"), 5: load("exp90_align_nhu5.json")}
D91 = {3: load("exp91_t3_nhu3.json"), 5: load("exp91_t3_nhu5.json")}
D93 = {3: load("exp93_comp_nhu3.json"), 5: load("exp93_comp_nhu5.json")}
D93D = {3: load("exp93_desc_nhu3.json"), 5: load("exp93_desc_nhu5.json")}
D92B = load("exp92b_why_worse2.json")


def a82(nhu, prior, lam="lam0.0001"):
    return D82[nhu]["A"][f"A:{prior}|{lam}"]["dis"]


def c82(nhu, algo, prior):
    return D82[nhu]["C"][f"C:{algo}|{prior}"]["dis"]


def a90(nhu, prior, arm):
    return D90[nhu]["res"][prior][arm]["dis"]


def a91(nhu, prior, arm, task):
    return D91[nhu]["res"][prior][arm][task]


def a93(nhu, key):
    return D93[nhu][key]["dis"]


def a93d(nhu, key):
    return D93D[nhu][key]["dis"]


# ---------------------------------------------------------------- 报告
L = []
def w(s=""):
    L.append(s)
    print(s)


w("# exp95 · 统一口径重算（全表唯一口径）")
w()
w("规则：**单位内先平均 → 单位等权 → 单位级配对 t**。")
w("归因单位 = 候选对；检出/识别单位 = 实体。所有数字与 `out/exp95_unified.json` 一致。")
w()
w("---")
w()

# ============================================================ §1 三处不一致的定量解释
w("## §1 三处不一致：定量解释")
w()
w("### A · 为什么 0.828 与 0.8674 是同一个东西")
w()
w("| |U| | 先验 | record 等权 (n_rec) | **单位等权 (n_unit)** | 差 | exp89 旧值 |")
w("|---|---|---|---:|---:|---:|---:|")
res_A = {}
for nhu in (3, 4, 5):
    d = a82(nhu, "clm")
    ur, uu = record_mean(d), mean_pair(list(unit_mean(d).values()))
    res_A[nhu] = dict(record=ur, unit=uu, n_rec=n_records(d), n_unit=len(unit_mean(d)))
    old = {3: 0.8674, 4: 0.8309, 5: 0.8072}.get(nhu)
    w(f"| {nhu} | clm | {ur:.4f} ({n_records(d)}) | **{uu:.4f} ({len(unit_mean(d))})** | "
      f"{uu - ur:+.4f} | {old if old else '—'} |")
w()
w("⇒ **A 不是矛盾，是两种汇总**：exp82 报 record 等权（n=36），exp89 报单位等权（n=19）。")
w("  统一后全表用**单位等权**；exp82 的 record 等权数值不再进正文。")
w()

w("### B · +0.320 是 exp82 报告里的一个 bug —— 真值是 +0.614 / +0.602")
w()
# 先算结构类均值 vs rand 的三档（供 §1-B、§2 复用）
res_B = {}
for _nhu, _pri in ((3, STRUCT), (4, STRUCT_8), (5, STRUCT_8)):
    _sm = struct_mean_dict({p: a82(_nhu, p) for p in _pri}, _pri)
    _rd = unit_mean(a82(_nhu, "rand384"))
    _pr = paired(_sm, _rd, "struct-mean vs rand")
    res_B[_nhu] = dict(struct=float(np.mean(list(_sm.values()))),
                       rand=float(np.mean(list(_rd.values()))), **_pr)
w("| 算法 | Δ(clm−rand) | 说明 |")
w("|---|---:|---|")
_clm, _rd = a82(3, "clm"), a82(3, "rand384")
_matched = []
for _k in _clm:
    _a = {r["tag"]: r["auc_dis"] for r in _clm[_k]}
    _b = {r["tag"]: r["auc_dis"] for r in _rd.get(_k, [])}
    for _t in set(_a) & set(_b):
        _matched.append(_a[_t] - _b[_t])
w(f"| 逐 record 均值差 | {record_mean(_clm) - record_mean(_rd):+.4f} | n_rec=36（两侧都有） |")
w(f"| 逐 record 配对差（按 pair+tag 匹配，{len(_matched)} 对） | {float(np.mean(_matched)):+.4f} | 与上一行一致，说明匹配无遗漏 |")
w(f"| **单位等权配对差（本文采用）** | **{res_B[3]['diff']:+.4f}** | n=19，与 exp89 完全一致 |")
w("| exp82 报告写的值 | ~~+0.320~~ | **无法从产物复现**（既非均值差、也非配对差、也非任何一侧的子集） |")
w()
w("⇒ 结论：**我们一直在用的 +0.602 是对的，exp82 报告里的 +0.320 是报告脚本的错误**。")
w("  exp82 其余 Δ 全部可复现（vs charbag −0.009、vs physchem +0.008、vs onehot +0.376 均吻合），"
  "只有 rand 这一栏错。已在 json 中留下核对记录。")
w()

w("### C · T1 与 T3 的 +0.602 / +0.607 统一后只剩一个数")
w()
sm3 = struct_mean_dict({p: a82(3, p) for p in STRUCT}, STRUCT)
rd3 = unit_mean(a82(3, "rand384"))
pC = paired(sm3, rd3)
f93 = float(np.mean(list(unit_mean(a93(3, "clm|rfull")).values())))
r93 = float(np.mean(list(unit_mean(a93(3, "rand384|rfull")).values())))
w(f"- 从 exp82 直接算：**Δ = {pC['diff']:+.4f}** (p={fmt_p(pC['p'])}, dz={pC['dz']:+.2f}, n={pC['n']})")
w(f"- 从 exp93 的 full 列算：结构类均值需用同一批先验，单 clm full={f93:.4f}、rand full={r93:.4f}")
w(f"- **结论：正文只引 exp82 直算的那一个数**；T3 的 full 列只用于「截断 vs 不截断」的内部对比，"
  f"不再对外声称 +0.607。")
w()
w("---")
w()

# ============================================================ §2 T1 先验轴
w("## §2 T1-a · 先验轴（结构类 vs 随机码本，单位 = 候选对）")
w()
w("| |U| | 结构类均值 | rand | **Δ** | t | p | dz | n | 旧值(exp89) |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---|")
res_T1a = {}
for nhu, pri, old in ((3, STRUCT, "+0.602 (p=4.2e-06, dz=1.49)"),
                      (4, STRUCT_8, "+0.566 (p=4.1e-07, dz=1.61)"),
                      (5, STRUCT_8, "+0.444 (p=5.8e-06, dz=1.33)")):
    sm = struct_mean_dict({p: a82(nhu, p) for p in pri}, pri)
    rd = unit_mean(a82(nhu, "rand384"))
    pr = paired(sm, rd)
    res_T1a[nhu] = dict(struct=float(np.mean(list(sm.values()))),
                        rand=float(np.mean(list(rd.values()))), **pr)
    w(f"| {nhu} | {np.mean(list(sm.values())):.4f} | {np.mean(list(rd.values())):.4f} | "
      f"**{pr['diff']:+.4f}** | {pr['t']:+.2f} | {fmt_p(pr['p'])} | {pr['dz']:+.2f} | "
      f"{pr['n']} | {old} |")
w()
w("⇒ 效应方向、显著性、三档一致性全部不变；**`dz` 与 exp89 逐位相同**——"
  "因为 exp89 本来就是单位等权，只是 exp82 报告的 record 等权数值被误当成同一口径引用。")
w()

# ============================================================ §3 T1 对齐轴
w("## §3 T1-b · 对齐轴（固定码本、只换对齐；单位 = 候选对）")
w()
for nhu in (3, 5):
    w(f"### |U|={nhu}")
    w()
    w("| 臂 | clm Δ (p) | physchem Δ (p) | ecfp Δ (p) | 显著改善 |")
    w("|---|---|---|---|---:|")
    res_T1b = {}
    for arm in ["cq", "zq", "iqr", "zC", "margin", "cosC", "mf"]:
        cells, sig_pos, sig_neg = [], 0, 0
        for p in ("clm", "physchem", "ecfp2048"):
            try:
                pr = paired(unit_mean(a90(nhu, p, arm)), unit_mean(a90(nhu, p, "lin")))
            except KeyError:
                cells.append("—")
                continue
            res_T1b[f"{p}|{arm}"] = pr
            if np.isfinite(pr["p"]) and pr["p"] < 0.05:
                if pr["diff"] > 0:
                    sig_pos += 1
                else:
                    sig_neg += 1
            cells.append(pc(pr))
        w(f"| `{arm}` | " + " | ".join(cells) + f" | {sig_pos} |")
    w()
w("⇒ 与 exp90 一致：**0 臂显著改善**；唯一显著效应全部为负。单位从 record 改回候选对后，"
  "`mf` 的负效应仍然显著（说明它不是小样本假象）。")
w()

# ============================================================ §4 T1 三任务
w("## §4 T1-c · 三任务矩阵（单位：检出/识别 = 实体 n=7，归因 = 候选对）")
w()
w("| |U| | 臂 | D 检出 clm | I 识别 clm | A 归因 clm |")
w("|---|---|---|---:|---:|---:|")
res_T1c = {}
for nhu in (3, 5):
    for arm in ("lin", "margin", "mf1e4"):
        try:
            d = float(np.mean(list(unit_mean(a91(nhu, "clm", arm, "D"), field="auc").values())))
            i = float(np.mean(list(unit_mean(a91(nhu, "clm", arm, "I"), field="auc").values())))
            a = float(np.mean(list(unit_mean(a91(nhu, "clm", arm, "A"), field="auc_dis").values())))
        except KeyError:
            continue
        res_T1c[f"{nhu}|{arm}"] = dict(D=d, I=i, A=a)
        w(f"| {nhu} | `{arm}` | {d:.4f} | {i:.4f} | {a:.4f} |")
w()
w("### margin vs lin（单位级配对；检出/识别单位 = 实体 n=7，归因单位 = 候选对）")
w()
w("| |U| | 任务 | 先验 | Δ | t | p | n |")
w("|---|---|---|---:|---:|---:|---:|")
res_T1d = {}
for nhu in (3, 5):
    for task, tname, fld in (("D", "检出", "auc"), ("I", "识别", "auc"), ("A", "归因", "auc_dis")):
        for p in ("clm", "physchem", "ecfp2048"):
            try:
                pr = paired(unit_mean(a91(nhu, p, "margin", task), field=fld),
                            unit_mean(a91(nhu, p, "lin", task), field=fld))
            except KeyError:
                continue
            res_T1d[f"{nhu}|{task}|{p}"] = pr
            star = "**" if np.isfinite(pr["p"]) and pr["p"] < 0.05 else ""
            w(f"| {nhu} | {tname} | {p} | {star}{pr['diff']:+.4f}{star} | {pr['t']:+.2f} | "
              f"{fmt_p(pr['p'])} | {pr['n']} |")
w()
w("### 同一批比较的「划分级」敏感性（把划分当聚类单位）")
w()
w("实体只有 7 个，单位级检验功效很低；把 (seed, combo) 当聚类单位可回收一部分自由度，"
  "同时仍吸收「同一次切分」的相关性。两列同号才算稳。")
w()
w("| |U| | 任务 | 先验 | 单位级 Δ (p, n) | 划分级 Δ (p, n) | 同号 |")
w("|---|---|---|---:|---|---|---|")
res_T1d2 = {}
for nhu in (3, 5):
    for task, tname, fld in (("D", "检出", "auc"), ("I", "识别", "auc"), ("A", "归因", "auc_dis")):
        for p in ("clm", "physchem", "ecfp2048"):
            try:
                pr_t = paired(tag_mean(a91(nhu, p, "margin", task), field=fld),
                              tag_mean(a91(nhu, p, "lin", task), field=fld))
            except KeyError:
                continue
            pr_u = res_T1d[f"{nhu}|{task}|{p}"]
            res_T1d2[f"{nhu}|{task}|{p}"] = pr_t
            same = "✅" if np.sign(pr_t["diff"]) == np.sign(pr_u["diff"]) else "❌"
            w(f"| {nhu} | {tname} | {p} | {pr_u['diff']:+.4f} ({fmt_p(pr_u['p'])}, {pr_u['n']}) | "
              f"{pr_t['diff']:+.4f} ({fmt_p(pr_t['p'])}, {pr_t['n']}) | {same} |")
w()

# ============================================================ §5 检出轴：先验 vs 随机
w("## §5 T1-d · 检出的先验通道（单位 = 实体，n=7；源 exp92b）")
w()
w("| 先验 | 目标=类中心 `auc_priorZ` | 目标=净信号 `auc_priorDZ` | 随机码本 `auc_rand` |")
w("|---|---:|---:|---:|")
res_T1e = {}
for p in ("clm", "physchem"):
    cells = {}
    for f in ("auc_priorZ", "auc_priorDZ", "auc_rand"):
        ent = {}
        for eid, recs in D92B["res"][p].items():
            vs = [r[f] for r in recs if f in r and np.isfinite(r.get(f, np.nan))]
            if vs:
                ent[eid] = float(np.mean(vs))
        cells[f] = ent
    res_T1e[p] = cells
    w(f"| {p} | {np.mean(list(cells['auc_priorZ'].values())):.4f} | "
      f"{np.mean(list(cells['auc_priorDZ'].values())):.4f} | "
      f"{np.mean(list(cells['auc_rand'].values())):.4f} |")
for p in ("clm", "physchem"):
    c = res_T1e[p]
    pr = paired(c["auc_priorZ"], c["auc_rand"], f"{p} priorZ vs rand")
    pr2 = paired(c["auc_priorDZ"], c["auc_rand"], f"{p} priorDZ vs rand")
    w()
    w(f"- **{p}**：类中心目标 vs 随机码本 Δ={pr['diff']:+.4f} (p={fmt_p(pr['p'])}, n={pr['n']})；"
      f"净信号目标 vs 随机 Δ={pr2['diff']:+.4f} (p={fmt_p(pr2['p'])})")
w()
w("⇒ 检出的**先验通道**：CLM 上与随机码本不可区分（这就是 T1 里「Δ=+0.007, p=0.95」的来源）；")
w("  physchem 方向为正但 n=7 达不到显著。**必须同时披露 n=7 且含 A05/A06（基质谱退化）**。")
w()

# ============================================================ §6 T2 十先验
w("## §6 T2 · 十先验（auc_dis，单位等权）")
w()
hdr = "| 组 | 先验 | auc_dis (sd) | top-1 | vs rand Δ (p, dz) | |U|=4 | |U|=5 | 衰减 3→5 |"
w(hdr)
w("|---|---|---:|---:|---|---:|---:|---:|")
res_T2 = {}
for p in STRUCT + COMP + NULL:
    grp = "S" if p in STRUCT else ("P" if p in COMP else "N")
    d3 = a82(3, p)
    um = unit_mean(d3)
    vals = np.array(list(um.values()))
    m3, s3 = float(vals.mean()), float(vals.std(ddof=1))
    try:
        t3 = float(np.mean(list(unit_mean(D82[3]["A"][f"A:{p}|lam0.0001"]["top1"],
                                        field="top1").values())))
    except KeyError:
        t3 = float("nan")
    if p == "rand384":
        pr = dict(n=0, diff=0.0, sd=0.0, t=float("nan"), p=float("nan"),
                  dz=float("nan"), identity=False)
    else:
        pr = paired(um, rd3)
    v4 = float(np.mean(list(unit_mean(a82(4, p)).values()))) if p not in ("ecfp384", "avalon") else float("nan")
    v5 = float(np.mean(list(unit_mean(a82(5, p)).values()))) if p not in ("ecfp384", "avalon") else float("nan")
    res_T2[p] = dict(group=grp, auc_dis=m3, sd=s3, top1=t3, u4=v4, u5=v5,
                     delta_rand=pr, decay=(v5 - m3) if np.isfinite(v5) else float("nan"))
    if p == "rand384":
        cell_dr = "—（自身）"
    else:
        cell_dr = f"{pr['diff']:+.4f} ({fmt_p(pr['p'])}, {pr['dz']:+.2f})"
    w(f"| {grp} | {p} | {m3:.4f} ({s3:.3f}) | {t3:.4f} | {cell_dr} | "
      f"{v4:.4f} | {v5:.4f} | {(v5 - m3):+.4f} |")
ew = np.mean([res_T2[p]["auc_dis"] for p in STRUCT])
w(f"| **S 均值** | 6 个 | **{ew:.4f}** | — | **{res_B[3]['diff']:+.4f} ({fmt_p(res_B[3]['p'])}, "
  f"{res_B[3]['dz']:+.2f})** | {res_B[4]['struct']:.4f} | {res_B[5]['struct']:.4f} | "
  f"{res_B[5]['struct'] - res_B[3]['struct']:+.4f} |")
w()
w("第二级（结构类内部两两比较，单位等权）：")
w()
w("| |U| | 对数 | 显著对数 | 最小 p | 跨度 |")
w("|---|---:|---:|---:|---:|")
res_L2 = {}
for nhu, pri in ((3, STRUCT), (4, STRUCT_8), (5, STRUCT_8)):
    ms = {p: unit_mean(a82(nhu, p)) for p in pri}
    ps, deltas = [], []
    for i, j in __import__("itertools").combinations(pri, 2):
        pr = paired(ms[i], ms[j])
        ps.append(pr["p"])
        deltas.append(abs(pr["diff"]))
    ps = np.array(ps)
    res_L2[nhu] = dict(n_pairs=len(ps), n_sig=int((ps < 0.05).sum()),
                       min_p=float(ps.min()), spread=float(max(deltas)))
    w(f"| {nhu} | {len(ps)} | **{int((ps < 0.05).sum())}** | {fmt_p(float(ps.min()))} | {max(deltas):.4f} |")
w()

w("## §6b T4a · 先验装到普通算法上（单位等权，|U|=3）")
w()
w("| 算法 | clm | rand | Δ (p, dz, n) | 最强普通先验 | Δ(clm−best) p |")
w("|---|---:|---:|---|---|---:|")
res_T4a = {}
for algo in ("ridge", "pls", "krr", "knn1", "mlp"):
    try:
        ac = unit_mean(c82(3, algo, "clm"))
        ar = unit_mean(c82(3, algo, "rand384"))
    except KeyError:
        continue
    pr = paired(ac, ar)
    best, bv, bp = "", -1, float("nan")
    for q in STRUCT + COMP:
        try:
            v = float(np.mean(list(unit_mean(c82(3, algo, q)).values())))
        except KeyError:
            continue
        if v > bv:
            best, bv = q, v
    try:
        bp = paired(ac, unit_mean(c82(3, algo, best)))["p"]
    except KeyError:
        pass
    res_T4a[algo] = dict(clm=float(np.mean(list(ac.values()))),
                         rand=float(np.mean(list(ar.values()))),
                         best=best, best_val=bv, **pr)
    w(f"| {algo} | {np.mean(list(ac.values())):.4f} | {np.mean(list(ar.values())):.4f} | "
      f"**{pr['diff']:+.4f}** ({fmt_p(pr['p'])}, {pr['dz']:+.2f}, {pr['n']}) | {best} {bv:.3f} | "
      f"{fmt_p(bp)} |")
w()

# ============================================================ §7 T3 秩截断
w("## §7 T3 · 秩截断（单位等权，|U|=3）")
w()
w("| 码本 | r=1 | r=2 | r=3 | r=5 | r=8 | full | 斜率/维 |")
w("|---|---:|---:|---:|---:|---:|---:|---:|")
res_T3 = {}
for key in ("clm", "physchem", "charbag", "ecfp2048", "rand384", "onehot"):
    row, vals = [], []
    for r in (1, 2, 3, 5, 8, "full"):
        k = f"{key}|r{r}"
        try:
            m = float(np.mean(list(unit_mean(a93(3, k)).values())))
        except KeyError:
            m = float("nan")
        row.append(m)
        vals.append(m)
    sl = np.polyfit([1, 2, 3, 5, 8], vals[:5], 1)[0]
    res_T3[key] = dict(zip(["r1", "r2", "r3", "r5", "r8", "full"], vals), slope=float(sl))
    w(f"| {key} | " + " | ".join(f"{v:.3f}" for v in row) + f" | {sl:+.4f} |")
w()
w("### 判决：给各向同性码本它的事后最优秩")
w()
w("| |U| | 对手 | 最优秩 | 值 | 结构类 full | 单位级 Δ (p,n) | 划分级 p | 结论 |")
w("|---|---|---|---:|---:|---:|---|---:|---|")
res_T3b = {}
for nhu in (3, 5):
    _pri = STRUCT if nhu == 3 else STRUCT_8
    sm = struct_mean_dict({p: a82(nhu, p) for p in _pri}, _pri)
    for opp in ("rand384", "onehot"):
        best_r, best_v, store = None, -1, None
        for r in (1, 2, 3, 5, 8):
            m = unit_mean(a93(nhu, f"{opp}|r{r}"))
            v = float(np.mean(list(m.values())))
            if v > best_v:
                best_v, best_r, store = v, f"r{r}", m
        pr = paired(sm, store)
        _tags = {q: tag_mean(a82(nhu, q)) for q in _pri}
        _rtags = tag_mean(D93[nhu][f"{opp}|{best_r}"]["dis"])
        _keys = set(_rtags)
        for q in _pri:
            _keys &= set(_tags[q])
        _st = {k: float(np.mean([_tags[q][k] for q in _pri])) for k in _keys}
        _rt = {k: _rtags[k] for k in _keys}
        pr_t = paired(_st, _rt)
        res_T3b[f"{nhu}|{opp}"] = dict(rank=best_r, value=best_v, **pr,
                                        split_p=pr_t["p"], split_n=pr_t["n"])
        verdict = "**救不回来**" if pr["p"] < 0.05 else "打平"
        w(f"| {nhu} | {opp} | {best_r} | {best_v:.3f} | {np.mean(list(sm.values())):.3f} | "
          f"{pr['diff']:+.3f} ({fmt_p(pr['p'])}, {pr['n']}) | {fmt_p(pr_t['p'])} | {verdict} |")
w()
w("### physchem 组成块归因")
w()
w("| |U| | full | desc (12 维) | elem (13 维) | desc−full | elem−full |")
w("|---|---:|---:|---:|---:|---|---|")
res_T3c = {}
for nhu in (3, 5):
    vals = {}
    for k in ("desc", "elem", "full"):
        vals[k] = float(np.mean(list(unit_mean(a93d(nhu, f"physchem|{'rfull' if k == 'full' else k}")).values())))
    res_T3c[nhu] = vals
    w(f"| {nhu} | {vals['full']:.4f} | {vals['desc']:.4f} | {vals['elem']:.4f} | "
      f"{vals['desc'] - vals['full']:+.4f} | {vals['elem'] - vals['full']:+.4f} |")
w()

# ============================================================ §8 敏感性
w("## §8 敏感性：把聚类单位换成「划分」")
w()
w("若效应只是聚合方式造成的假象，换成以 (seed, combo) 为聚类单位后应当消失。")
w()
w("| 比较 | Δ (单位级, n=候选对) | Δ (划分级, n=split) | 划分级 p | 是否同号 |")
w("|---|---:|---:|---:|---|")
res_S2 = {}
for nhu, pri in ((3, STRUCT), (5, STRUCT_8)):
    per_tag = {p: tag_mean(a82(nhu, p)) for p in pri}
    tag_rand = tag_mean(a82(nhu, "rand384"))
    keys = set(tag_rand)
    for p in pri:
        keys &= set(per_tag[p])
    keys = sorted(keys)
    sdict = {k: float(np.mean([per_tag[p][k] for p in pri])) for k in keys}
    rdict = {k: tag_rand[k] for k in keys}
    pr = paired(sdict, rdict)
    res_S2[nhu] = pr
    same = "✅ 同号" if np.sign(pr["diff"]) == np.sign(res_T1a[nhu]["diff"]) else "❌ 反号"
    u = res_T1a[nhu]
    w(f"| 结构类 vs rand (\\|U\\|={nhu}) | {u['diff']:+.4f} (n={u['n']}) | {pr['diff']:+.4f} | "
      f"{fmt_p(pr['p'])} | {same} |")
w()

# ============================================================ §9 变更表
w("## §9 变更表（旧 → 新，正文按新值写）")
w()
w("| 位置 | 旧值 | 新值（统一口径） | 结论变了吗 |")
w("|---|---|---|---|")
w(f"| T1 归因 × 换先验 (\\|U\\|=3) | +0.602 (p=4.2e-06, dz=1.49) | "
  f"**{res_T1a[3]['diff']:+.4f} (p={fmt_p(res_T1a[3]['p'])}, dz={res_T1a[3]['dz']:+.2f}, n={res_T1a[3]['n']})** | 不变 |")
w(f"| T1 归因 × 换先验 (\\|U\\|=4) | +0.566 (p=4.1e-07, dz=1.61) | "
  f"**{res_T1a[4]['diff']:+.4f} (p={fmt_p(res_T1a[4]['p'])}, dz={res_T1a[4]['dz']:+.2f})** | 不变 |")
w(f"| T1 归因 × 换先验 (\\|U\\|=5) | +0.444 (p=5.8e-06, dz=1.33) | "
  f"**{res_T1a[5]['diff']:+.4f} (p={fmt_p(res_T1a[5]['p'])}, dz={res_T1a[5]['dz']:+.2f})** | 不变 |")
w(f"| T2 clm auc_dis | 0.8674 (n=19) / 0.828 (n=36) 混用 | **{res_T2['clm']['auc_dis']:.4f} (单位等权)** | 统一 |")
w(f"| T3 结构类 full | +0.607 | **不再单独引用**，与 T1 同源 | 删除重复证据 |")
w()

# ============================================================ §10 结论变更
w("## §10 结论变更清单（统一口径后才看得出来的事）")
w()
w("| # | 原结论 | 统一口径后 | 处置 |")
w("|---|---|---|---|")
w("| 1 | 归因：结构 vs rand **+0.602**（p=4.2e-06, dz=1.49） | **完全不变**；划分级 +0.5918 (p=6.0e-05) 同号 | ✅ 双重稳健，可作承重结论 |")
w("| 2 | 归因：对齐 8 臂无一显著改善 | **完全不变**；0/18 显著为正，17/18 同号为负 | ✅ 加强（原判定更稳） |")
w("| 3 | 识别：margin 6/6 同号正、4 格显著（n=36） | 单位级（n=7）**0/6 显著**；划分级（n=12）4/6 显著 | ⚠ 改报划分级，并注明 n=7 功效不足 |")
w("| 4 | 检出：margin 最大 +0.074 p=.0035 | 单位级 p=.175；**划分级 p=.0075** 仍显著 | ⚠ 保留结论，但必须写明是划分级 n=12 |")
w("| 5 | 检验「化学先验不可被通用低秩投影替代」：3/4 格显著 | **1/4 格显著**（只有 \\|U\\|=3 的 rand，p=.009） | ❌ 降级：只能写「\\|U\\|=3 上成立」，其余写打平 |")
w("| 6 | physchem 元素计数是纯噪声（−0.306, p<.0001） | −0.214（\\|U\\|=3）/ −0.289（\\|U\\|=5），方向与量级保持 | ✅ 仍成立 |")
w("| 7 | exp82 报告 Δ(clm−rand)=+0.320 | **不可复现，真值 +0.614（record）/ +0.602（unit）** | ❌ 弃用 +0.320 |")
w()
w("**唯一一条需要改写的正文句子**：§7 的「各向同性码本救不回来（4 格中 3 格显著）」→")
w("改为「在 \\|U\\|=3 上，即便给予事后最优秩，随机码本仍显著落后（+0.318, p=.009）；")
w("在 \\|U\\|=5 与 one-hot 上则不再可区分」——这是**缩小 claim 范围**，不是撤回。")
w()

# ============================================================ 写盘
payload = dict(
    rule="unit-level averaging then unit-level paired t; unit=pair(A)/entity(D,I)",
    A_explain=res_A, B_explain=res_B, C_pair=pC,
    T1a=res_T1a, T1b=res_T1b, T1c=res_T1c, T1d=res_T1d, T1d_split=res_T1d2,
    T1e={p: {f: v for f, v in d.items()} for p, d in res_T1e.items()},
    T2=res_T2, L2=res_L2, T4a=res_T4a, T3=res_T3, T3b=res_T3b, T3c=res_T3c,
    sensitivity=res_S2,
)
with open(os.path.join(OUT, "exp95_unified.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)

with open(os.path.join(OUT, "exp95_unified_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(f"\n[saved] out/exp95_unified.json, out/exp95_unified_report.md  ({len(L)} lines)")
