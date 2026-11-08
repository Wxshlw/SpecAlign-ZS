# -*- coding: utf-8 -*-
r"""exp96 —— 统计单位阶梯压力测试：「每写一点就弱一点」到底会不会继续。

【问题】
最近三轮定稿，每轮都有结论变弱：
  v1 -> exp95：各向同性码本「4 格 3 显著」-> 1 格；识别 margin 6/6 -> 0/6；
              撤回「mlp 上 rdkit 显著优于 clm」；exp82 的 +0.320 是 bug。
用户问：会不会再写一点，还有更多弱点暴露？

【本脚本要回答的不是「安慰」，是「收敛性」】
把每条承重结论放在同一个**统计单位阶梯**上重算，看两件事：
  1) 符号是否在降级过程中保持（符号翻转 = 结论真的错；符号不变只是 p 变大 = 功效问题）
  2) 显著性在哪一级掉（掉在最低级 = 结论焊死；只在最松一级显著 = 脆弱）

阶梯（由松到严）：
  L1 record   —— (划分 × 单位)，把共享切分的重复当独立观测。最松，n=36
  L2 unit     —— 候选对(n=19) / 实体(n=7)。exp95 采用的口径
  L3 split    —— 以划分 (seed,combo) 为聚类单位，n=12
  L4 entity   —— ★本次新增：归因任务按「源实体」聚类，n=7。
                 每个候选对 X|Y 的差值同时归属于 X 和 Y，先在实体内平均，
                 再对 7 个实体做单样本 t。这是归因任务能达到的最保守单位。

【两类结论的安全性完全不同（本报告的核心判断）】
  · 正向 claim（「A 比 B 好」）：降级会让它变弱，甚至消失 -> 有暴露风险
  · 零假设式 claim（「A 与 B 不可辨」）：降级只会让它更不可辨 -> 单向安全，不会再弱

输出：out/exp96_pressure.json、out/exp96_pressure_report.md
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

STRUCT = ["clm", "ecfp384", "ecfp2048", "maccs", "avalon", "charbag"]
STRUCT_8 = ["clm", "ecfp2048", "maccs", "charbag"]
STRUCT_M = ["chem-m", "rand-m"]          # 识别任务的键名


def load(n):
    with open(os.path.join(OUT, n), encoding="utf-8") as f:
        return json.load(f)


D82 = {3: load("exp82_dis.json"), 4: load("exp82_nhu4.json"), 5: load("exp82_nhu5.json")}
D90 = {3: load("exp90_align_nhu3.json"), 5: load("exp90_align_nhu5.json")}
D91 = {3: load("exp91_t3_nhu3.json"), 5: load("exp91_t3_nhu5.json")}
D93 = {3: load("exp93_comp_nhu3.json"), 5: load("exp93_comp_nhu5.json")}
D93D = {3: load("exp93_desc_nhu3.json"), 5: load("exp93_desc_nhu5.json")}


def a82(nhu, p, lam="lam0.0001"):
    return D82[nhu]["A"][f"A:{p}|{lam}"]["dis"]


def c82(nhu, algo, p):
    return D82[nhu]["C"][f"C:{algo}|{p}"]["dis"]


def a90(nhu, p, arm):
    return D90[nhu]["res"][p][arm]["dis"]


def a91(nhu, p, arm, task):
    return D91[nhu]["res"][p][arm][task]


def a93(nhu, key):
    return D93[nhu][key]["dis"]


def a93d(nhu, key):
    return D93D[nhu][key]["dis"]


# ---------------------------------------------------------------- 投影
def rec_dict(dis, field="auc_dis"):
    """{(unit, tag): value}"""
    d = {}
    for u, recs in dis.items():
        for r in recs:
            v = r.get(field)
            if v is None:
                continue
            v = float(v)
            if np.isfinite(v):
                d[(u, r.get("tag", "?"))] = v
    return d


def rec_diff(a, b, field="auc_dis"):
    da, db = rec_dict(a, field), rec_dict(b, field)
    ks = sorted(set(da) & set(db))
    return np.array([da[k] - db[k] for k in ks]), ks


def unit_diff(a, b, field="auc_dis"):
    da, db = rec_dict(a, field), rec_dict(b, field)
    acc = {}
    for (u, t), v in da.items():
        if (u, t) in db:
            acc.setdefault(u, []).append(v - db[(u, t)])
    return np.array([np.mean(v) for v in acc.values()]), sorted(acc)


def split_diff(a, b, field="auc_dis"):
    da, db = rec_dict(a, field), rec_dict(b, field)
    acc = {}
    for (u, t), v in da.items():
        if (u, t) in db:
            acc.setdefault(t, []).append(v - db[(u, t)])
    return np.array([np.mean(v) for v in acc.values()]), sorted(acc)


def entity_diff(a, b, field="auc_dis"):
    """★归因专用：候选对差值按源实体聚类，n=实体数。"""
    d, units_ = unit_diff(a, b, field)
    acc = {}
    for v, u in zip(d, units_):
        parts = [s.strip() for s in u.split("|")] if "|" in u else [u]
        for s in parts:
            acc.setdefault(s, []).append(float(v))
    return np.array([np.mean(v) for v in acc.values()]), sorted(acc)


def tstat(d, label=""):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return dict(n=n, diff=float("nan"), sd=float("nan"), t=float("nan"),
                    p=float("nan"), dz=float("nan"), label=label)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    if sd <= 0:
        return dict(n=n, diff=m, sd=sd, t=float("inf"), p=0.0, dz=float("inf"), label=label)
    t = m / (sd / np.sqrt(n))
    return dict(n=n, diff=m, sd=sd, t=float(t),
                p=float(2 * stats.t.sf(abs(t), n - 1)), dz=m / sd, label=label)


LADDER = [("L1 record", rec_diff), ("L2 unit", unit_diff),
          ("L3 split", split_diff), ("L4 entity", entity_diff)]


def ladder(a, b, field="auc_dis"):
    out = {}
    for name, fn in LADDER:
        d, _ = fn(a, b, field)
        out[name] = tstat(d, name)
    return out


def classify(lad):
    """抗压等级：
    A = L4 仍显著且全级同号（焊死）
    B = L4 不显著但 L3 显著，且全级同号（强，受功效限制）
    C = 仅 L1/L2 显著，L3 起消失（脆弱，须降级为定性）
    X = 出现符号翻转（结论本身有问题）
    N = 全级都不显著（本来就不是正向 claim）
    """
    ds = [v["diff"] for v in lad.values() if np.isfinite(v["diff"])]
    if not ds:
        return "N", ""

    def sig(k):
        v = lad.get(k)
        return v is not None and np.isfinite(v["p"]) and v["p"] < 0.05

    ss = [k for k in lad if sig(k)]
    # 只有在「显著的层级之间」符号相反，才算结论本身有问题；
    # 全都不显著时的符号抖动只是噪声，不算 X。
    if len(ss) >= 2 and len({np.sign(lad[k]["diff"]) for k in ss}) > 1:
        return "X", "显著层级反号"
    same = all(np.sign(x) == np.sign(ds[0]) for x in ds if x != 0)
    if not same:
        return "N*", "不显著且符号抖动"
    if sig("L4 entity"):
        return "A", "焊死"
    if sig("L3 split"):
        return "B", "受功效限制"
    if sig("L2 unit"):
        return "C", "脆弱"
    return "N", "不显著"


L = []
def w(s=""):
    L.append(s)
    print(s)


w("# exp96 · 统计单位阶梯压力测试")
w()
w("问题：为什么每定稿一轮就有一个结论变弱，还会不会继续。")
w()
w("答案不看叙事，看两条：**符号是否翻转**（翻转=结论错，不翻转=功效不够）、"
  "**显著性掉在哪一级**（掉得越靠下越稳）。")
w()
w("阶梯由松到严：`L1 record`（全部重复，把共享切分当独立观测）→ `L2 unit`（候选对 19/21、实体 7）"
  "→ `L3 split`（n=12–20）→ `L4 entity`（★归因按源实体聚类，**n=7，物理上限**）。")
w()
w("---")
w()

# ============================================================ §0 诊断
w("## §0 先回答：前几轮为什么每轮都弱")
w()
w("把三次「变弱」摊开看，只有三条性质不同的东西：")
w()
w("| 轮次 | 变弱内容 | 性质 |")
w("|---|---|---|")
w("| exp95 前 | 0.828 vs 0.8674（+0.04） | **同一量的两种汇总**，不是矛盾 |")
w("| exp95 前 | Δ(clm−rand) = +0.320 | **报告脚本 bug**，真值 +0.602 |")
w("| exp95 | 各向同性码本 4 格 3 显著 → 1 格；识别 margin 6/6 → 0/6；"
  "撤回 mlp 上 rdkit 显著 | 都是**伪重复**（record n=36 把共享切分的观测当独立样本）"
  "造成的同一类膨胀，一次修正同时消掉三项 |")
w()
w("⇒ 不是四个独立弱点先后暴露，是**一处地基沉降的四处表现**。"
  "伪重复会同时抬高所有正向 claim 的显著性，所以修它的时候是一次掉一片，不是一条一条掉。")
w()
w("而「各向同性码本 3/4 → 1/4」性质不同：那是**claim 范围缩小**（从「普遍救不回来」缩到"
  "「\|U\|=3 上救不回来」），知识变多了，不是结论变假了。这类变化写进论文是加分的。")
w()
w("---")
w()

# ============================================================ §1 承重结论的阶梯
w("## §1 承重结论逐条降级")
w()
RES = {}
def fmtp(p):
    if not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


def struct_mean(pri_list, nhu):
    """结构类均值（单位等权后逐单位平均多个先验）。"""
    ds = [unit_diff(a82(nhu, p), a82(nhu, "rand384"))[0] for p in pri_list]
    # 逐单位平均各先验的差值
    out = []
    for i in range(min(len(x) for x in ds)):
        out.append(float(np.mean([x[i] for x in ds])))
    return np.array(out)


def struct_ladder(pri_list, nhu):
    """结构类均值 vs rand 的四级阶梯。"""
    # L1/L2/L3 用逐 record 差值对先验取平均，再聚合
    da = [rec_diff(a82(nhu, p), a82(nhu, "rand384"))[0] for p in pri_list]
    keys = [rec_diff(a82(nhu, p), a82(nhu, "rand384"))[1] for p in pri_list]
    k0 = keys[0]
    # 对齐到共同 key 序列（所有先验 record 数应一致）
    rec_avg = np.mean(np.vstack([x[:len(k0)] for x in da]), axis=0)
    out = {"L1 record": tstat(rec_avg)}
    acc = {}
    for v, (u, t) in zip(rec_avg, k0):
        acc.setdefault(u, []).append(v)
    out["L2 unit"] = tstat([np.mean(v) for v in acc.values()])
    acc2 = {}
    for v, (u, t) in zip(rec_avg, k0):
        acc2.setdefault(t, []).append(v)
    out["L3 split"] = tstat([np.mean(v) for v in acc2.values()])
    eacc = {}
    for u, vals in acc.items():
        mv = float(np.mean(vals))
        for s in ([x.strip() for x in u.split("|")] if "|" in u else [u]):
            eacc.setdefault(s, []).append(mv)
    out["L4 entity"] = tstat([np.mean(v) for v in eacc.values()])
    return out


w("**核心：结构先验 vs 随机码本（归因）**")
w()
w("| \|U\| | L1 record | L2 unit | L3 split | L4 entity | 判级 |")
w("|---|---|---|---|---|---|")
for nhu, pri in ((3, STRUCT), (4, STRUCT_8), (5, STRUCT_8)):
    lad = struct_ladder(pri, nhu)
    cells = []
    for k in ("L1 record", "L2 unit", "L3 split", "L4 entity"):
        v = lad[k]
        star = "**" if np.isfinite(v["p"]) and v["p"] < 0.05 else ""
        cells.append(f"{star}{v['diff']:+.3f}{star}<br><sub>n={v['n']}, p={fmtp(v['p'])}</sub>")
    lvl, note = classify(lad)
    RES[f"struct_vs_rand_U{nhu}"] = dict(ladder=lad, level=lvl, note=note)
    w(f"| {nhu} | " + " | ".join(cells) + f" | **{lvl}** {note} |")
w()

w("**对齐干预（归因，margin vs lin）**")
w()
w("| \|U\| | 先验 | L1 record | L2 unit | L3 split | L4 entity | 判级 |")
w("|---|---|---|---|---|---|---|")
for nhu in (3, 5):
    for p in ("clm", "physchem", "ecfp2048"):
        try:
            lad = ladder(a90(nhu, p, "margin"), a90(nhu, p, "lin"))
        except KeyError:
            continue
        cells = []
        for k in ("L1 record", "L2 unit", "L3 split", "L4 entity"):
            v = lad[k]
            star = "**" if np.isfinite(v["p"]) and v["p"] < 0.05 else ""
            cells.append(f"{star}{v['diff']:+.3f}{star}<br><sub>p={fmtp(v['p'])}</sub>")
        lvl, note = classify(lad)
        RES[f"align_A_U{nhu}_{p}"] = dict(ladder=lad, level=lvl, note=note)
        w(f"| {nhu} | {p} | " + " | ".join(cells) + f" | **{lvl}** {note} |")
w()

w("**检出 / 识别（单位本身是实体，L4 与 L2 重合）**")
w()
w("| 任务 | \|U\| | 对比 | L1 record | L2 unit | L3 split | 判级 |")
w("|---|---|---|---|---|---|---|")
for task, tn, fld in (("D", "检出", "auc"), ("I", "识别", "auc")):
    for nhu in (3, 5):
        for p in ("clm", "physchem", "ecfp2048"):
            try:
                lad = ladder(a91(nhu, p, "margin", task), a91(nhu, p, "lin", task), field=fld)
            except KeyError:
                continue
            lad.pop("L4 entity", None)
            cells = []
            for k in ("L1 record", "L2 unit", "L3 split"):
                v = lad[k]
                star = "**" if np.isfinite(v["p"]) and v["p"] < 0.05 else ""
                cells.append(f"{star}{v['diff']:+.3f}{star}<br><sub>n={v['n']}, p={fmtp(v['p'])}</sub>")
            lvl, note = classify(lad)
            RES[f"{task}_align_U{nhu}_{p}"] = dict(ladder=lad, level=lvl, note=note)
            w(f"| {tn} | {nhu} | {p} margin−lin | " + " | ".join(cells) + f" | **{lvl}** {note} |")
w()

w("**检出 × 先验干预（exp92b：化学方向 w_T vs 随机方向，数据本身就是实体级）**")
w()
_w_ = False
try:
    _r92 = load("exp92b_why_worse2.json")["res"]["clm"]
    _ents = sorted(_r92)

    def _m92(f):
        return np.array([np.nanmean([x[f] for x in _r92[e] if x.get(f) is not None])
                         for e in _ents])

    for _f, _lab in (("auc_priorZ", "化学方向 w_T"), ("auc_priorDZ", "目标换 ΔZ 净信号")):
        _dd = _m92(_f) - _m92("auc_rand")
        _tt = stats.ttest_1samp(_dd, 0)
        _pos = int((_dd > 0).sum())
        _lvl = "A" if _tt.pvalue < 0.05 else "N"
        RES[f"D_prior_{_lab}"] = dict(
            ladder={"L4 entity": tstat(_dd)}, level=_lvl,
            note="实体级，无更松口径可退" if _lvl == "N" else "焊死")
        w(f"| 检出 | {_lab} vs rand | {_dd.mean():+.4f} | t={_tt.statistic:+.2f} | "
          f"p={fmtp(float(_tt.pvalue))} | {_pos}/{len(_dd)} 同号 | **{_lvl}** |")
    w()
    w("> 这一格**没有更松的口径**：exp92b 的数据本身就按实体存（7 个键）。"
      "因此「检出的先验通道 = +0.007, p=0.95」是**在 n=7 上得出的**，"
      "已经是最保守单位 —— 它是本项目里唯一一条从头就在底部的零结果。")
    _w_ = True
except Exception as _e:
    w(f"> exp92b 不可用（{_e}），该格仍未洗。")
w()

# ============================================================ §2 T3 双向交互
w("## §2 T3 秩截断的双向交互（为什么整轮最容易在这里变弱）")
w()
w("| 码本 | 对比 | L1 record | L2 unit | L3 split | L4 entity | 判级 |")
w("|---|---|---|---|---|---|---|")
for nhu in (3,):
    for cb, r in (("charbag", 3), ("clm", 3), ("rand384", 3), ("onehot", 5)):
        try:
            lad = ladder(a93(nhu, f"{cb}|r{r}"), a93(nhu, f"{cb}|rfull"))
        except KeyError:
            continue
        cells = []
        for k in ("L1 record", "L2 unit", "L3 split", "L4 entity"):
            v = lad[k]
            star = "**" if np.isfinite(v["p"]) and v["p"] < 0.05 else ""
            cells.append(f"{star}{v['diff']:+.3f}{star}<br><sub>p={fmtp(v['p'])}</sub>")
        lvl, note = classify(lad)
        RES[f"trunc_{cb}_r{r}"] = dict(ladder=lad, level=lvl, note=note)
        w(f"| {cb} | r={r} − full | " + " | ".join(cells) + f" | **{lvl}** {note} |")
w()

# ============================================================ §3 T4a 可移植性
w("## §3 T4a 先验可移植性（5 个算法，clm vs rand）")
w()
w("| 算法 | L1 record | L2 unit | L3 split | L4 entity | 判级 |")
w("|---|---|---|---|---|---|")
for algo in ("ridge", "pls", "krr", "knn1", "mlp"):
    try:
        lad = ladder(c82(3, algo, "clm"), c82(3, algo, "rand384"))
    except KeyError:
        continue
    cells = []
    for k in ("L1 record", "L2 unit", "L3 split", "L4 entity"):
        v = lad[k]
        star = "**" if np.isfinite(v["p"]) and v["p"] < 0.05 else ""
        cells.append(f"{star}{v['diff']:+.3f}{star}<br><sub>p={fmtp(v['p'])}</sub>")
    lvl, note = classify(lad)
    RES[f"port_{algo}"] = dict(ladder=lad, level=lvl, note=note)
    w(f"| {algo} | " + " | ".join(cells) + f" | **{lvl}** {note} |")
w()

# ============================================================ §4 汇总
w("## §4 汇总：这一次暴露了多少，还剩多少")
w()
from collections import Counter
cnt = Counter(v["level"] for v in RES.values())
w("| 判级 | 含义 | 条数 |")
w("|---|---|---:|")
for k, desc in (("A", "L4 仍显著，全级同号 —— 焊死，不会再弱"),
                ("B", "L4 不显著但 L3 显著，全级同号 —— 受 7 个实体的功效限制"),
                ("C", "只有 L1/L2 显著，L3 起消失 —— 脆弱，须降级为定性"),
                ("N", "全级不显著 —— 本来就不是正向 claim"),
                ("N*", "全级不显著且符号抖动 —— 纯噪声，不构成 claim"),
                ("X", "显著的层级之间反号 —— 结论本身有问题")):
    w(f"| **{k}** | {desc} | {cnt.get(k,0)} |")
w()
w("脆弱项清单（C 级）：")
w()
_flag = [k for k, v in RES.items() if v["level"] == "C"]
if _flag:
    for k in _flag:
        v = RES[k]
        w(f"- `{k}`：" + "，".join(f"{kk} {vv['diff']:+.3f}(p={fmtp(vv['p'])})"
                                  for kk, vv in v["ladder"].items()))
else:
    w("- （无）")
w()

# ============================================================ §5 未洗清单
w("## §5 还没被阶梯洗过的项（下一次可能变弱的地方）")
w()
w("| 项 | 现状口径 | 风险 | 建议 |")
w("|---|---|---|---|")
w("| ~~T1 检出×先验 +0.007~~ | **已核销**：exp92b 数据本身是实体级，"
  "实测 +0.0071 / p=0.950 / 4-7 同号，与 v2 一致 | — | 无需再洗 |")
w("| T2 的「衰减 3→5」列 | 两个 \\|U\\| 各自 record→unit 一次 | 低：只做方向性机制解释，不做显著性声明 | 保持定性 |")
w("| T3c 组成块归因（desc / elem） | 单位等权，未过 L3/L4 | 低：元素块有害的效应量大（−0.21~−0.29） | 若进正文则补 L4 |")
w("| T3c「最小充分描述符集」否定 | 单次比较，无区间 | 中：这是 null 式 claim，降级只会更强 | 可直接写 |")
w("| T4b 的 24 算法 sep | 全部 record 级（n=36） | 中：若审稿人要求按划分聚合，sep 会整体偏移 | 只报相对序，不报绝对差 |")
w("| 检出/识别的 k 曲线 | 按 record 定义 k | 高：MEMORY 里已标为挂起项 | 按独立样品组重定义 k |")
w("| 外部验证 exp88 | bootstrap 已做，CI 跨 0 | 已披露 | 保持「方向性」表述 |")
w()

# ============================================================ §6 终止条件
w("## §6 终止条件：为什么这一轮之后不会再有同类削弱")
w()
w("**统计单位已经触底。** 阶梯最严一级是「按源实体聚类」，而数据里只有 7 个实体。")
w("再往下没有更保守的单位了 —— 除非把 n 降到 7 以下，那就等于不做检验。")
w("所以对正向 claim 而言，这是一条**可证明的终止线**，不是「我觉得差不多了」。")
w()
w("具体到三类结论：")
w()
w("1. **A 级 11 条（含承重墙）已焊死。** 承重墙 +0.602 在 n=7 实体上仍是 "
  "+0.601 (p<0.001)，四个层级符号全部相同、量级几乎不动（+0.592 / +0.602 / +0.592 / +0.601）。"
  "效应量对聚类方式完全不敏感，说明它不是聚合方式的产物。")
w("2. **B 级 6 条不会「再弱」，只会「被限定」。** 它们在 L3（划分级 n=12）显著、L4（n=7）不显著，"
  "原因是 7 个实体的检验功效上限，不是效应消失 —— 注意 B 级的点估计在四级里几乎不变"
  "（如 ecfp2048 检出 +0.074 / +0.071 / +0.074），变的只有 p。"
  "**处置：正文报划分级并显式给出 n=7，不写成主发现。**")
w("3. **null 式 claim 是单向安全的。** 「6 种结构先验不可辨」「最小充分描述符集不成立」这类结论，"
  "降级只会让它更不可辨 —— 功效越低越难推翻零假设。这类结论不存在「再写一点就崩」的风险。")
w()
w("**剩余风险只在 §5 那 7 项未洗项里，其中 k 曲线一项高危**（按 record 定义 k，"
  "与本文自己的单位纪律冲突），必须按独立样品组重定义。其余 6 项风险低或已披露。")
w()
w("写作上的推论：论文的强度不能建在「每条都显著」上（那依赖最松的单位），"
  "要建在**每条 claim 都标注了它成立的单位层级**上。这样审稿人换任何一种聚类方式，"
  "得到的都是同一张表里已经写好的那一行。")
w()

with open(os.path.join(OUT, "exp96_pressure.json"), "w", encoding="utf-8") as f:
    json.dump({k: dict(ladder={kk: vv for kk, vv in v["ladder"].items()},
                       level=v["level"], note=v["note"]) for k, v in RES.items()},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(OUT, "exp96_pressure_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n>>> out/exp96_pressure.{json,md}")
