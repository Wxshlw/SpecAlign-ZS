"""exp02 —— 论文6 全部数据集 QC 与可行性核查。

聚焦作者 2026-10-30 的兜底：自采三套（S1 茶叶 IR / S2 蛋白粉 IR / S3 食用油 UV-Vis）
采集期有瑕疵，**仅当 QC 抓到具体质量问题时才触发排除**，否则保留。

QC 指标（统一口径）：
  * 轴：升序 & 范围合理（FTIR 400–4000 cm⁻¹；UV-Vis 195–1143 nm）
  * 完整性：NaN/Inf 计数（加载器已滤，期望 0）
  * 量纲：FTIR 吸光度应 ≈≥0、范围 0–~2.5；UV-Vis 透射应 ≥0
  * 基线：均值谱与轴的相关系数（漂移）、最大逐点差分尖峰
  * 梯度集：水平数、每水平 n、浓度响应单调性（组间方差/组内方差）
  * 自采三套：proxy_groups（无样品 ID 的代理分组），簇数 vs 谱数 → 重复扫描结构
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import p6_data as P

OUT = r"D:/2026sci论文实验设计/论文6/out"
os.makedirs(OUT, exist_ok=True)

GRADIENT = {"D7w", "D7p", "D7d", "D8", "D1", "D2", "D3", "S3"}  # 带浓度梯度
MODALITY = {"D7w": "FTIR", "D7p": "FTIR", "D7d": "FTIR", "D8": "FTIR",
            "D1": "FTIR", "D2": "FTIR", "D3": "FTIR",
            "S1": "FTIR", "S2": "FTIR", "S3": "UV-Vis"}


def snr_gradient(X, y):
    """梯度集信噪比代理：组间方差 / 组内方差（沿通道取中位）。"""
    levels = np.unique(y)
    if len(levels) < 2:
        return None
    per_level = [X[y == lv] for lv in levels]
    mean_per = np.array([b.mean(0) for b in per_level])      # (L, C)
    grand = X.mean(0)
    ss_between = np.sum([len(b) * (mean_per[i] - grand) ** 2 for i, b in enumerate(per_level)], 0)
    ss_within = np.sum([((b - mean_per[i]) ** 2).sum(0) for i, b in enumerate(per_level)], 0)
    denom = ss_within + 1e-9
    f = ss_between / denom
    return float(np.nanmedian(f))


def qc_one(sid, D):
    X = D["X"].astype(float)
    axis = np.asarray(D["axis"], float)
    y = D.get("y")
    n, C = X.shape
    mod = MODALITY[sid]
    r = dict(src=sid, modality=mod, n=n, C=C,
             axis_min=round(float(axis[0]), 2), axis_max=round(float(axis[-1]), 2),
             dnu=round(float(np.median(np.diff(axis))), 4),
             vmin=round(float(np.nanmin(X)), 3), vmax=round(float(np.nanmax(X)), 3),
             vmean=round(float(np.nanmean(X)), 3), vstd=round(float(np.nanstd(X)), 3))
    # 完整性
    r["n_nan"] = int(np.isnan(X).any(1).sum())
    r["n_inf"] = int(np.isinf(X).any(1).sum())
    # 负值占比（吸光度应≈非负；UV 透射应≥0）
    r["frac_neg"] = round(float((X < 0).mean()), 4)
    # 基线漂移：均值谱与轴的相关系数（绝对值大 = 强基线倾斜）
    mean_spec = X.mean(0)
    if np.std(mean_spec) > 0 and np.std(axis) > 0:
        r["baseline_drift_corr"] = round(float(np.corrcoef(mean_spec, axis)[0, 1]), 3)
    else:
        r["baseline_drift_corr"] = 0.0
    # 尖峰：逐点差分绝对值的 99.9 分位 / 总体 std
    d = np.abs(np.diff(X, axis=1))
    r["spike_ratio"] = round(float(np.percentile(d, 99.9) / (r["vstd"] + 1e-9)), 2)
    # 标签
    if y is not None:
        ys = np.asarray(y)
        r["y_unique_n"] = int(len(np.unique(ys)))
        r["y_unique"] = [round(float(v), 2) for v in sorted(np.unique(ys))[:10]]
    # 梯度响应
    if sid in GRADIENT and y is not None:
        r["snr_gradient"] = round(snr_gradient(X, np.asarray(y)), 3) if snr_gradient(X, np.asarray(y)) is not None else None
        r["level_counts"] = {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    # 自采三套：代理分组
    if sid in P.SELF_SETS:
        pg = P.proxy_groups(X)
        r["proxy_groups"] = {k: v for k, v in pg.items() if k.startswith("tau_")}
        r["proxy_n_spectra"] = pg["n_spectra"]
        # 簇数远小于谱数 → 存在重复扫描结构（采集可能重复同一样品）
        for k, v in r["proxy_groups"].items():
            r[f"ratio_{k}"] = round(v / n, 3)
    return r


def main():
    sets = P.load_all(verbose=True)
    rows = [qc_one(sid, D) for sid, D in sets.items()]
    # 自采三套质量判定
    issues = {}
    for r in rows:
        sid = r["src"]
        if sid not in P.SELF_SETS:
            continue
        flags = []
        if MODALITY[sid] == "FTIR":
            if r["frac_neg"] > 0.05:
                flags.append(f"负值占比高({r['frac_neg']})")
            if abs(r["baseline_drift_corr"]) > 0.5:
                flags.append(f"基线强漂移(|{r['baseline_drift_corr']}|)")
        else:  # UV-Vis
            if r["vmin"] < 0:
                flags.append(f"透射出现负值(min={r['vmin']})——量纲/采集异常")
            if r["vmax"] > 1000:
                flags.append(f"强度量级异常(max={r['vmax']})——非标准吸光度/透射")
        if sid == "S2" and r["n"] < 80:
            flags.append(f"样本数偏少(n={r['n']})")
        if "proxy_groups" in r:
            # 在较松阈值 tau=0.95 下，若簇数/谱数 < 0.6，提示大量重复扫描
            if r.get("ratio_tau_0.95", 1.0) < 0.6:
                flags.append(f"代理簇数/谱数={r['ratio_tau_0.95']}→疑似重复扫描")
        issues[sid] = flags

    # 报告
    lines = ["# exp02 数据集 QC 与可行性核查（2026-10-30）", "",
             "> 触发式兜底：自采三套（S1/S2/S3）仅在下方 QC 抓到具体质量问题时才排除，否则保留。", ""]
    lines += ["| 数据集 | 模态 | n | 通道 | 轴范围 | 量纲范围 | 负值% | 基线漂移| 尖峰比 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["src"] in P.SELF_SETS:
            tag = " **[自采]**"
        else:
            tag = ""
        lines.append(f"| {r['src']}{tag} | {r['modality']} | {r['n']} | {r['C']} | "
                     f"{r['axis_min']}–{r['axis_max']} | [{r['vmin']},{r['vmax']}] | "
                     f"{r['frac_neg']} | {r.get('baseline_drift_corr')} | {r.get('spike_ratio')} |")
    lines += ["", "## 梯度集浓度响应（SNR 代理 = 组间方差/组内方差 中位）", ""]
    lines += ["| 数据集 | 水平数 | 每水平 n | SNR_gradient |", "|---|---|---|---|"]
    for r in rows:
        if "snr_gradient" in r:
            lc = r.get("level_counts", {})
            lc_s = ",".join(f"{k}:{v}" for k, v in list(lc.items())[:8])
            lines.append(f"| {r['src']} | {r['y_unique_n']} | {lc_s} | {r['snr_gradient']} |")
    lines += ["", "## 自采三套代理分组（无样品 ID；簇数 vs 谱数）", ""]
    lines += ["| 数据集 | 谱数 | tau0.999 | tau0.995 | tau0.99 | tau0.95 | 比值(0.95) |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "proxy_groups" in r:
            pg = r["proxy_groups"]
            lines.append(f"| {r['src']} | {r['proxy_n_spectra']} | {pg.get('tau_0.999')} | "
                         f"{pg.get('tau_0.995')} | {pg.get('tau_0.99')} | {pg.get('tau_0.95')} | "
                         f"{r.get('ratio_tau_0.95')} |")
    lines += ["", "## 自采三套质量问题判定（触发式排除依据）", ""]
    any_issue = False
    for sid, fl in issues.items():
        if fl:
            any_issue = True
            lines.append(f"- **{sid}：发现问题** → " + "；".join(fl))
        else:
            lines.append(f"- **{sid}：QC 未见具体质量问题，保留**")
    lines += ["", "### 结论", ""]
    if any_issue:
        lines.append("自采三套中存在可量化的采集质量问题（见上）。按作者兜底，将在 anchor / G0 中"
                     "以『含自采』与『排除自采』两种口径各跑一遍，做敏感性对照；仅当问题导致结论反转时剔除。")
    else:
        lines.append("自采三套 QC 均未发现可量化的质量问题，按作者兜底**保留**，不触发排除。")
    lines.append("")
    # 写文件
    with open(os.path.join(OUT, "exp02_p6_qc.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    import csv
    all_keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    rows_clean = []
    for r in rows:
        rc = {}
        for k in all_keys:
            v = r.get(k, "")
            if isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False)
            rc[k] = v
        rows_clean.append(rc)
    with open(os.path.join(OUT, "exp02_p6_qc.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader(); w.writerows(rows_clean)
    print("\n".join(lines[:0]) + "\n[written] exp02_p6_qc.md / .csv")
    for sid, fl in issues.items():
        print(f"  {sid}: {'ISSUES='+str(fl) if fl else 'OK (keep)'}")


if __name__ == "__main__":
    main()
