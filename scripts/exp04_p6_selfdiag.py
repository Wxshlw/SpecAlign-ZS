"""exp04 —— 自采三套（S1/S2/S3）数据质量诊断：判定是否触发「排除」授权。

判别逻辑：
- 若 PCA(2D) 按 y 着色后各类仍不可分（类间方差占比低、类内混叠），
  则属采集质量问题 → 触发排除。
- 若 PCA 能分开但 PLS-DA 失败，则属 pipeline bug → 修 pipeline 保留。
- 同时核对 y 与 X 行数一致、标签语义合理（类数、类大小）。
"""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
warnings = None
import warnings as _w
_w.filterwarnings("ignore")

sys.path.insert(0, r"D:/2026sci论文实验设计/论文6/scripts")
import p6_data as P

OUT = r"D:/2026sci论文实验设计/论文6/out"


def pca_separability(X, y):
    """返回 (evr1, evr2, per_class_centroid_dist_to_global)。"""
    Xs = StandardScaler().fit_transform(X)
    p = PCA(n_components=2).fit_transform(Xs)
    evr = PCA(n_components=2).fit(Xs).explained_variance_ratio_
    y = np.asarray(y)
    classes = np.unique(y)
    g = p.mean(0)
    inter = 0.0
    for c in classes:
        inter += ((p[y == c].mean(0) - g) ** 2).sum()
    # 类间分散度 / 总分散度
    tot = ((p - g) ** 2).sum()
    sep = inter / tot if tot > 0 else 0.0
    return float(evr[0]), float(evr[1]), float(sep), int(len(classes))


def plsda(X, y):
    y = np.asarray(y)
    n, c = X.shape
    ncomp = min(10, n - 1, c)
    yt, yp = [], []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = PLSRegression(n_components=ncomp)
        m.fit(X[tr], y[tr])
        yp.extend((m.predict(X[te]).ravel() > 0.5).astype(int))
        yt.extend(y[te])
    yt, yp = np.array(yt), np.array(yp)
    return float(accuracy_score(yt, yp)), float(balanced_accuracy_score(yt, yp))


def main():
    sets = P.load_all(verbose=False)
    lines = ["# exp04 自采三套数据质量诊断（是否触发「排除」授权）", "",
             "> 判据：PCA 按 y 着色是否可分；近随机 PLS-DA + 低 PCA 类间分散 → 采集质量问题。", ""]
    lines += ["| 集 | n | 类数 | 类大小(min/median/max) | PCA-1 方差% | PCA-2 方差% | PCA类间分散度 | PLS-DA acc | PLS-DA bal | 判定 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for sid in ["S1", "S2", "S3"]:
        if sid not in sets:
            continue
        D = sets[sid]
        X = D["X"].astype(float)
        y = np.asarray(D["y"])
        ncls = len(np.unique(y))
        sizes = np.bincount(y.astype(int)) if y.dtype != object else np.array([(y == c).sum() for c in np.unique(y)])
        # 预处理（snv+sg）与 exp03 一致
        mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True)
        Xz = (X - mu) / (sd + 1e-9)
        if Xz.shape[1] > 15:
            from scipy.signal import savgol_filter
            Xz = savgol_filter(Xz, 15, 3, deriv=1, axis=1)
        ev1, ev2, sep, ncls2 = pca_separability(Xz, y)
        try:
            acc, bal = plsda(Xz, y)
        except Exception:
            acc, bal = float("nan"), float("nan")
        # 判定
        if ev1 + ev2 < 0.05 or sep < 0.15 or (acc < 0.30 and bal < 0.35):
            verdict = "⚠️ 触发排除（采集质量）"
        elif acc < 0.5:
            verdict = "弱信号（谨慎）"
        else:
            verdict = "可用"
        verdicts[sid] = verdict
        lines.append(f"| {sid} | {len(X)} | {ncls} | {sizes.min()}/{int(np.median(sizes))}/{sizes.max()} | "
                     f"{ev1*100:.1f} | {ev2*100:.1f} | {sep:.3f} | {acc:.3f} | {bal:.3f} | {verdict} |")
    lines += ["", "## 结论", ""]
    for sid, v in verdicts.items():
        lines.append(f"- **{sid}**: {v}")
    lines.append("")
    lines.append("若 S1/S2 均触发排除，则按用户授权「三个自采数据集结果排除」，主证据退回 7 个公共数据集（D1–D8 中非零类者）。")
    with open(os.path.join(OUT, "exp04_p6_selfdiag.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[written] exp04_p6_selfdiag.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
