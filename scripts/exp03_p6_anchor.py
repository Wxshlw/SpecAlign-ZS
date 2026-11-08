"""exp03 —— 论文6 甲的参考方法锚点（躲 Food Chemistry desk reject 的必须项）。

提供 Food Chemistry 硬门槛④「与既有方法对照」+③「定量指标 LOD/RMSEP/灵敏度-特异性」：

1. PLS 回归（梯度集）：cross-validated RMSEP、R²、LOD(=3.3σ/slope)、LOQ(=10σ/slope)。
2. 掺假存在性检测（梯度集，y>0 vs y==0）：PLS-DA → 灵敏度/特异性/准确率。
3. SAM/谱库匹配参考行：1-NN（余弦）有监督光谱匹配，作为「传统参考方法」对照。
4. 分类集（S1 茶配方 / S2 植物蛋白）：PLS-DA 准确率。

两种口径对照：
  * 含自采（S1+S2 经基线校正保留，S3 排除）
  * 排除全部自采（敏感性）

预处理：FTIR 统一 SG 一阶导去基线 + SNV；S3(UV 原始光强) 排除。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             recall_score, precision_score, balanced_accuracy_score)
warnings.filterwarnings("ignore")

sys.path.insert(0, r"D:/2026sci论文实验设计/论文6/scripts")
import p6_data as P

OUT = r"D:/2026sci论文实验设计/论文6/out"
SELF = set(P.SELF_SETS)
EXCLUDE_S3 = {"S3"}                      # 原始光强，格式硬伤，排除
GRADIENT = {"D7w", "D7p", "D7d", "D8", "D1", "D2", "D3"}   # 带浓度梯度
CLASSIF = {"S1", "S2"}                                       # 分类（无浓度梯度）


def preprocess_ftir(X):
    """SNV + SG 一阶导去基线。X: (n,C) 升序轴。"""
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True)
    Xz = (X - mu) / (sd + 1e-9)
    if Xz.shape[1] > 15:
        Xz = savgol_filter(Xz, window_length=15, polyorder=3, deriv=1, axis=1)
    return Xz


def pls_r_metrics(X, y, groups):
    n, c = X.shape
    ncomp = min(10, n - 1, c)
    if groups is not None and len(set(groups)) < n and len(set(groups)) > 1:
        cv = GroupKFold(n_splits=min(5, len(set(groups))))
        try:
            cv.split(X, y, groups)
        except Exception:
            cv = KFold(5, shuffle=True, random_state=0)
    else:
        cv = KFold(5, shuffle=True, random_state=0)
    yt, yp = [], []
    for tr, te in cv.split(X, y, groups):
        m = PLSRegression(n_components=ncomp)
        m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]).ravel())
    yt, yp = np.array(yt), np.array(yp)
    rmsep = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(set(yt)) > 1 else float("nan")
    slope = float(np.polyfit(yt, yp, 1)[0]) if len(set(yt)) > 1 else float("nan")
    sigma = float(np.std(yt - yp))
    lod = (3.3 * sigma / abs(slope)) if slope and abs(slope) > 1e-9 else float("inf")
    loq = (10 * sigma / abs(slope)) if slope and abs(slope) > 1e-9 else float("inf")
    return dict(rmsep=rmsep, r2=r2, lod=lod, loq=loq, slope=slope,
                ymax=float(np.max(y)), n=n, cv=str(type(cv).__name__))


def pls_da_metrics(X, y, groups, present_only=False):
    """分类或掺假存在性检测。present_only: 把 y>0 当正类。"""
    if present_only:
        yb = (np.asarray(y) > 0).astype(int)
    else:
        yb = np.asarray(y)
    n, c = X.shape
    ncomp = min(10, n - 1, c)
    if groups is not None and len(set(groups)) > 1 and len(set(groups)) < n:
        cv = GroupKFold(n_splits=min(5, len(set(groups))))
    else:
        cv = KFold(5, shuffle=True, random_state=0)
    yt, yp = [], []
    for tr, te in cv.split(X, yb, groups if groups is not None else None):
        m = PLSRegression(n_components=ncomp)
        m.fit(X[tr], yb[tr])
        sc = m.predict(X[te]).ravel()
        yp.extend((sc > 0.5).astype(int)); yt.extend(yb[te])
    yt, yp = np.array(yt), np.array(yp)
    if present_only:
        pos = (yt == 1).sum(); neg = (yt == 0).sum()
        if pos == 0 or neg == 0:
            return dict(acc=float(accuracy_score(yt, yp)), sens=float("nan"),
                        spec=float("nan"), bal=float("nan"), n=int(n))
        return dict(acc=float(accuracy_score(yt, yp)),
                    sens=float(recall_score(yt, yp, pos_label=1, zero_division=0)),
                    spec=float(recall_score(yt, yp, pos_label=0, zero_division=0)),
                    bal=float(balanced_accuracy_score(yt, yp)), n=int(n))
    # 多分类：仅报准确率与平衡准确率
    return dict(acc=float(accuracy_score(yt, yp)),
                sens=float("nan"), spec=float("nan"),
                bal=float(balanced_accuracy_score(yt, yp)), n=int(n))


def sam_1nn(X, y, groups, task):
    """1-NN 余弦（有监督参考方法行）。梯度集预测最近训练样本的y→RMSEP；分类集→准确率。"""
    n = len(X)
    Xz = X - X.mean(1, keepdims=True)
    Xn = Xz / np.clip(np.linalg.norm(Xz, axis=1, keepdims=True), 1e-12, None)
    if groups is not None and len(set(groups)) > 1 and len(set(groups)) < n:
        cv = GroupKFold(n_splits=min(5, len(set(groups))))
    else:
        cv = KFold(5, shuffle=True, random_state=0)
    from sklearn.metrics.pairwise import cosine_similarity
    yt, yp = [], []
    for tr, te in cv.split(Xn, y, groups):
        S = cosine_similarity(Xn[te], Xn[tr])
        nn = S.argmax(1)
        yt.extend(y[te]); yp.extend(y[tr][nn])
    yt, yp = np.array(yt), np.array(yp)
    if task == "gradient":
        return dict(sam_rmsep=float(np.sqrt(mean_squared_error(yt, yp))) if len(set(yt)) > 1 else float("nan"),
                    sam_type="regression")
    return dict(sam_acc=float(accuracy_score(yt, yp)), sam_type="classification")


def run(use_self: bool):
    sets = P.load_all(verbose=False)
    rows = []
    for sid, D in sets.items():
        if sid in EXCLUDE_S3:
            continue
        if sid in SELF and not use_self:
            continue
        if sid in SELF:                    # S1/S2 经基线校正保留
            X = preprocess_ftir(D["X"].astype(float))
        else:
            X = preprocess_ftir(D["X"].astype(float))   # 公共集统一同样预处理，口径一致
        y = np.asarray(D["y"], float)
        groups = D.get("groups")
        rec = dict(src=sid, modality=P.MODALITY[sid], n=len(X))
        if sid in GRADIENT:
            rec["n_pos"] = int((y > 0).sum())
            rec["n_neg"] = int((y == 0).sum())
            pls = pls_r_metrics(X, y, groups)
            pres = pls_da_metrics(X, y, groups, present_only=True)
            rec.update({f"pls_{k}": v for k, v in pls.items()})
            rec.update({f"pres_{k}": v for k, v in pres.items()})
            sam = sam_1nn(X, y, groups, "gradient")
            rec.update(sam)
            rec["task"] = "gradient"
        elif sid in CLASSIF:
            da = pls_da_metrics(X, y, groups, present_only=False)
            rec.update({f"da_{k}": v for k, v in da.items()})
            rec["task"] = "classification"
        rows.append(rec)
    return rows


def _f(v, fmt=".3f"):
    """安全格式化：float 用 fmt，nan/inf 打 —，缺键/非数打 —。"""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return format(f, fmt)


def main():
    res = run(use_self=False)   # 主口径：仅公共数据集（自采三套经 exp04 诊断触发排除）
    # 报告
    lines = ["# exp03 参考方法锚点（甲硬门槛：定量指标 + 与既有方法对照）", "",
             "> **主口径：仅公共数据集（D1–D3、D7w/D7p/D7d、D8，共 7 集）。**", "",
             "> 自采三套（S1 茶 IR / S2 蛋白粉 IR / S3 食用油 UV-Vis）经 exp04 质量诊断触发排除：",
             "> PCA 类间分散度 S1=0.007 / S2=0.039 / S3=0.000（类别无谱学信号，属采集质量问题，作者已授权排除）；",
             "> 预处理：FTIR 统一 SG 一阶导去基线 + SNV。", "",
             "## 主表（公共 7 集）", "",
              "| 集 | 任务 | n | PLS RMSEP | R² | LOD | LOQ | 存在性检测(灵敏/特异/准) | SAM(1-NN) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        if r["task"] == "gradient":
            sam = _f(r.get("sam_rmsep"))
            lines.append(f"| {r['src']} | 梯度 | {r['n']} | {_f(r.get('pls_rmsep'))} | "
                         f"{_f(r.get('pls_r2'))} | {_f(r.get('pls_lod'),'.2f')} | {_f(r.get('pls_loq'),'.2f')} | "
                         f"{_f(r.get('pres_sens'))}/{_f(r.get('pres_spec'))}/{_f(r.get('pres_acc'))} | {sam} |")
        else:
            lines.append(f"| {r['src']} | 分类 | {r['n']} | — | — | — | — | — | da_acc={_f(r.get('da_acc'))} |")
    lines += ["", "## 存在性检测补充（类不平衡口径）", "",
              "| 集 | n_pos | n_neg | 灵敏度 | 特异度 | 平衡准确率 |", "|---|---|---|---|---|---|"]
    for r in res:
        if r["task"] == "gradient":
            lines.append(f"| {r['src']} | {r.get('n_pos','—')} | {r.get('n_neg','—')} | "
                         f"{_f(r.get('pres_sens'))} | {_f(r.get('pres_spec'))} | {_f(r.get('pres_bal'))} |")
    lines += ["", "## 结论（甲硬门槛满足度）", ""]
    lines.append("- 定量指标：梯度集均报 RMSEP / R² / LOD / LOQ；存在性检测报灵敏度-特异性（类不平衡下另报平衡准确率）→ 满足 Food Chemistry ③。")
    lines.append("- 与既有方法对照：SAM(1-NN 谱库匹配) + PLS-DA 参考行 → 满足 Food Chemistry ④。")
    lines.append("- 自采三套（S1/S2/S3）经 exp04 质量诊断触发排除，主证据全部来自公共数据集（作者 2026-10-30 授权口径）。")
    with open(os.path.join(OUT, "exp03_p6_anchor.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    import csv
    keys = []
    for r in res:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "exp03_p6_anchor.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print("[written] exp03_p6_anchor.md / exp03_p6_anchor.csv")
    def _clean(r):
        out = {}
        for k, v in r.items():
            if k == "src":
                continue
            try:
                out[k] = round(float(v), 3)
            except (TypeError, ValueError):
                out[k] = v
        return out
    print("=== PUBLIC ONLY ===")
    for r in res:
        print(r["src"], _clean(r))


if __name__ == "__main__":
    main()
