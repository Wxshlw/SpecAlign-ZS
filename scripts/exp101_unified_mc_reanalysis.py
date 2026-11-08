# -*- coding: utf-8 -*-
r"""exp101 —— 三任务在「对等 MC 随机零假设」下的统一重算

背景
----
审稿人 Major 2 指出随机码本单次抽样会让 Δ 虚高。核查发现：
  - identification (exp79) 已用 nrand 次独立抽样、逐个算 AUC 再平均 => 干净
  - attribution   (exp82) 用 out/exp81_prior_cache.npz 里**单个**固定 rand384 => 需修正
  - detection     (exp92b) 用 seed*977+31 生成的**单个**随机码本     => 需修正
exp99 / exp100 已补出这两处的对等 MC 对照，且都精确复现了原单次 draw 数值。

本脚本把三个任务放到同一口径，重算：
  - 各任务 Δ 及其 95% CI / p
  - Major 1 交互：I − D（同一批 7 个实体，配对的差值之差）
  - Major 5 梯度：A(L4 entity) − I
  - 双分离在 MC 口径下是否仍成立

输出 out/exp101_unified_mc_report.json（供正文/补充材料/response letter 引用）
"""
from __future__ import annotations
import os, json
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def load(n):
    with open(os.path.join(OUT, n), encoding="utf-8") as f:
        return json.load(f)


def one_sample(x, label):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); m = float(x.mean()); sd = float(x.std(ddof=1))
    t, p = stats.ttest_1samp(x, 0)
    se = sd / np.sqrt(n); tc = stats.t.ppf(.975, n - 1)
    return dict(label=label, n=n, d=m, sd=sd, t=float(t), p=float(p),
                ci=[float(m - tc * se), float(m + tc * se)],
                dz=float(m / sd) if sd > 0 else None)


def dod(a, b, label):
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] - b[k] for k in ks], float)
    return one_sample(x, label)


def mean_over(d):
    return {k: float(np.mean(v)) for k, v in d.items() if v}


# ---------------------------------------------------------------- detection
E100 = load("exp100_detection_matched.json")["cells"]
d_acc = defaultdict(list); d_single = defaultdict(list); d_dz = defaultdict(list)
for c in E100:
    if c["prior"] != "clm":
        continue
    e = c["entity"]
    d_acc[e].append(c["auc_struct_Z"] - float(np.mean(c["auc_rand_Z"])))
    d_single[e].append(c["auc_struct_Z"] - c["auc_rand_Z"][0])
    d_dz[e].append(c["auc_struct_dZ"] - float(np.mean(c["auc_rand_dZ"])))
D_MC = mean_over(d_acc)
D_1draw = mean_over(d_single)
D_DZ_MC = mean_over(d_dz)

# ------------------------------------------------------------ identification
E79 = load("exp79_subst.json")["M"]["0"]["old-heldout"]
_cm = {e: float(np.mean([r["auc"] for r in recs])) for e, recs in E79["chem-m"].items()}
_rm = {e: float(np.mean([r["auc"] for r in recs])) for e, recs in E79["rand-m"].items()}
I_MC = {e: _cm[e] - _rm[e] for e in _cm if e in _rm}

# --------------------------------------------------------------- attribution
E99 = load("exp99_rand_mc_matched.json")["cells"]
per_cell_mc = {}; per_cell_1draw = {}
a_acc = defaultdict(list); a_single = defaultdict(list)
for c in E99:
    sm = float(np.mean(list(c["struct"].values())))
    mc = float(np.mean(c["mc"]))
    per_cell_mc[(c["unit"], c["tag"])] = sm - mc
    per_cell_1draw[(c["unit"], c["tag"])] = sm - c["fixed_rand"]
    a_acc[c["unit"]].append(sm - mc)
    a_single[c["unit"]].append(sm - c["fixed_rand"])
A_L2_MC = mean_over(a_acc)
A_L2_1draw = mean_over(a_single)

# L4 entity: 每个候选对的差值归给它的两个源实体
eacc = defaultdict(list); eacc1 = defaultdict(list)
for u, v in A_L2_MC.items():
    for s in u.split("|"):
        eacc[s.strip()].append(v)
for u, v in A_L2_1draw.items():
    for s in u.split("|"):
        eacc1[s.strip()].append(v)
A_L4_MC = mean_over(eacc)
A_L4_1draw = mean_over(eacc1)

# ------------------------------------------------------------------- 汇总
rep = dict(
    note="MC = 独立随机码本多次抽样、逐个算 AUC 再平均；1draw = 原稿单次固定随机码本",
    detection=dict(mc=one_sample(list(D_MC.values()), "D, MC null, target=Zc"),
                   one_draw=one_sample(list(D_1draw.values()), "D, single draw, target=Zc"),
                   mc_detection_specific=one_sample(list(D_DZ_MC.values()),
                                                    "D, MC null, target=dZ")),
    identification=dict(mc=one_sample(list(I_MC.values()), "I, MC null (exp79 native)")),
    attribution=dict(mc_L2=one_sample(list(A_L2_MC.values()), "A, MC null, L2 unit n=19"),
                     mc_L4=one_sample(list(A_L4_MC.values()), "A, MC null, L4 entity n=7"),
                     one_draw_L2=one_sample(list(A_L2_1draw.values()),
                                            "A, single draw, L2 unit n=19"),
                     one_draw_L4=one_sample(list(A_L4_1draw.values()),
                                            "A, single draw, L4 entity n=7")),
    interaction=dict(mc=dod(I_MC, D_MC, "I - D (MC null, paired n=7)"),
                     one_draw=dod(I_MC, D_1draw, "I - D (single draw, paired n=7)")),
    gradient=dict(mc=dod(A_L4_MC, I_MC, "A(L4) - I (MC null, paired n=7)"),
                  one_draw=dod(A_L4_1draw, I_MC, "A(L4) - I (single draw, paired n=7)")),
    per_entity=dict(D_MC=D_MC, D_1draw=D_1draw, I=I_MC,
                    A_L4_MC=A_L4_MC, A_L4_1draw=A_L4_1draw),
)


def show(k, r):
    print("  %-44s n=%2d  d=%+.4f  p=%.4g  95%%CI=[%+.4f,%+.4f]" %
          (r["label"], r["n"], r["d"], r["p"], r["ci"][0], r["ci"][1]))


print("=== 三任务（MC 对等 null） ===")
for k in ("mc", "one_draw", "mc_detection_specific"):
    show(k, rep["detection"][k])
show("I", rep["identification"]["mc"])
for k in ("mc_L2", "mc_L4", "one_draw_L2", "one_draw_L4"):
    show(k, rep["attribution"][k])
print("\n=== Major 1 交互 / Major 5 梯度 ===")
show("mc", rep["interaction"]["mc"]); show("1draw", rep["interaction"]["one_draw"])
show("mc", rep["gradient"]["mc"]); show("1draw", rep["gradient"]["one_draw"])

with open(os.path.join(OUT, "exp101_unified_mc_report.json"), "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)
print("\n[exp101] wrote out/exp101_unified_mc_report.json")
