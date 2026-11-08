r"""exp88d —— 外部数据集上的零样本掺杂物归因：8 先验 × 3 算法族。

数据：Mendeley 10.17632/ctgg7k4m5g（CC BY 4.0），ATR-FTIR，402-3738 cm-1，402 点取 1
  · 356 条纯油谱，18 类（EVOO / Almond / Canola / Corn / Grapeseed / Sunflower …）
  · 120 条 EVOO 掺混谱，掺杂物 ∈ {Almond, Canola, Corn}，浓度 5-90%，24 个配方

零样本构造
----------
  · 候选库 U = {Almond, Canola, Corn}   —— 这 3 类的**任何谱（纯油或掺混）都不参与训练**
  · 老类   = 其余 14 种纯油
  · 训练谱 = 老类纯油谱 × EVOO 纯谱 的线性合成混合（10 个浓度 0.1-0.9）
  · 测试谱 = 真实的 EVOO 掺混谱

为什么要合成混合（关键）
  第一版直接用"老类纯油谱"训练、用"掺混谱"测试，出现严重域漂移：
  所有先验 auc≈0.24（劣于随机），一个先验也没起作用——这不是"先验无效"，
  是训练/测试不同分布，实验本身失效。
  ATR-FTIR 近似满足 Beer-Lambert 线性叠加，用老类纯谱 + 基质纯谱合成混合物，
  可以在**完全不看候选类**的前提下让训练与测试同构。

三个算法族
  ridge : 本文头。S_o → Z_old 岭回归（对偶），s_T = z·W_T
  rf    : 随机森林多输出回归 S_o → Z_old（同一候选打分）
  svm   : 线性 SVR 多输出 S_o → Z_old

指标
  auc_dis : s_T − s_D 在 T/D 两类样本上的方向正确率（T、D 均未见）
  top1    : 3 候选 argmax 正确率
  浓度分层：三类浓度范围不重叠（Almond 60-90 / Canola 10-90 / Corn 5-60），必须分层
"""
from __future__ import annotations
import os, sys, json, itertools, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")

import exp88b_prior_rsa as A  # noqa: E402
from exp81a_prior_zoo import unit  # noqa: E402

CAND = [6, 11, 12]      # Corn, Almond, Canola
MATRIX = 1              # EVOO
LAMS = [1e-4, 1e-2, 1.0]
NPCA = 40
CONC_BANDS = [(0, 20, "low"), (20, 60, "mid"), (60, 101, "high")]
SYNTH_CONC = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def prep_spectra(X):
    from scipy.signal import savgol_filter
    X = np.asarray(X, float)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-9
    return savgol_filter((X - mu) / sd, 15, 3, deriv=1, axis=1)


def synth_mixtures(Xo, Xm, rng, concs=SYNTH_CONC):
    """老类纯谱 + 基质纯谱 线性混合。只用老类与基质，不看候选。"""
    out = []
    for c in concs:
        io = rng.integers(0, len(Xo), size=len(Xo))
        im = rng.integers(0, len(Xm), size=len(Xo))
        out.append(c * Xo[io] + (1 - c) * Xm[im])
    return np.concatenate(out, 0)


def head_ridge_R(S, Z, lam):
    """岭回归 S→Z，返回**预测函数**：给定候选化学向量，预测其类中心嵌入。"""
    K = S @ S.T
    A_ = np.linalg.solve(K + lam * np.eye(len(S)), Z)
    def predict(Sq):
        return (Sq @ S.T) @ A_
    return predict


def make_heads():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    def rf_head(S, Z, lam):
        m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        m.fit(S, Z)
        return lambda Sq: m.predict(Sq)

    def svm_head(S, Z, lam):
        from sklearn.multioutput import MultiOutputRegressor
        m = MultiOutputRegressor(SVR(kernel="linear", C=1.0), n_jobs=-1)
        m.fit(S, Z)
        return lambda Sq: m.predict(Sq)

    return {"ridge": head_ridge_R, "rf": rf_head, "svm": svm_head}


def unitr(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12
    return X / n


def auc2(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) < 1 or len(neg) < 1:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    d = np.load(os.path.join(OUT, "exp88_oil.npz"), allow_pickle=True)
    Xp, yc = d["X_pure"], d["yc"]
    Xa, ya_lab, conc = d["X_adul"], d["y"], d["conc"]
    names = list(d["names"])
    all_classes = sorted(set(yc.tolist()))
    others = [c for c in all_classes if c not in CAND and c != MATRIX]
    print(f"[exp88d] 纯油 {Xp.shape}  掺混 {Xa.shape}  候选 {CAND}  老类 {len(others)}")

    # ---- 合成混合（原始谱上做，之后统一预处理）----
    rng = np.random.default_rng(20261104)
    Xm_raw = Xp[yc == MATRIX]
    synth_raw, synth_cls = [], []
    for c in others:
        Xs = synth_mixtures(Xp[yc == c], Xm_raw, rng)
        synth_raw.append(Xs)
        synth_cls.append(np.full(len(Xs), c))
    synth_raw = np.concatenate(synth_raw, 0)
    synth_cls = np.concatenate(synth_cls, 0)
    print(f"  合成训练谱 {synth_raw.shape}  ({len(others)} 老类 × {len(SYNTH_CONC)} 浓度)")

    Zs = prep_spectra(synth_raw)
    Za = prep_spectra(Xa)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=NPCA, random_state=0).fit(Zs)
    synth_z = pca.transform(Zs)
    Z_old = np.stack([synth_z[synth_cls == c].mean(0) for c in others])

    Zt, conc_t = {}, {}
    for c in CAND:
        m = ya_lab == names[CAND.index(c)]
        Zt[c] = pca.transform(Za[m])
        conc_t[c] = conc[m]

    # 上界参照：用候选自身真实谱中心做分类（判断任务本身是否可分）
    oracle = {}
    for jT, jD in itertools.combinations(range(len(CAND)), 2):
        c1, c2 = CAND[jT], CAND[jD]
        oracle[f"{names[CAND.index(c1)]}|{names[CAND.index(c2)]}"] = auc2(
            Zt[c1].mean(0) @ Zt[c1].T, Zt[c2].mean(0) @ Zt[c2].T)
    print("  [oracle 上界] auc_dis:", {k: round(v, 3) for k, v in oracle.items()})

    T, native = A.build_priors(all_classes)
    cidx = {c: i for i, c in enumerate(all_classes)}
    rows_c = others                      # 训练行只有老类（基质已隐含在合成谱里）
    heads = make_heads()

    res = {}
    for hname, hfun in heads.items():
        for pname, V in T.items():
            S = V[[cidx[c] for c in rows_c]]
            Sc = V[[cidx[c] for c in CAND]]
            for lam in (LAMS if hname == "ridge" else [1e-4]):
                pred_fn = hfun(S, Z_old, lam)
                Zhat = unitr(pred_fn(Sc - S.mean(0, keepdims=True)))
                sc = {c: unitr(Zt[c]) @ Zhat[j] for j, c in enumerate(CAND)}
                key = (hname, pname)
                res.setdefault(key, {"auc": [], "top1": [], "band": {}})
                for jT, jD in itertools.combinations(range(len(CAND)), 2):
                    c1, c2 = CAND[jT], CAND[jD]
                    res[key]["auc"].append(dict(v=auc2(sc[c1], sc[c2]), lam=lam))
                    for lo, hi, nm in CONC_BANDS:
                        i1 = np.where((conc_t[c1] >= lo) & (conc_t[c1] < hi))[0]
                        i2 = np.where((conc_t[c2] >= lo) & (conc_t[c2] < hi))[0]
                        if len(i1) >= 3 and len(i2) >= 3:
                            res[key]["band"].setdefault(nm, []).append(
                                auc2(sc[c1][i1], sc[c2][i2]))
                for j, c in enumerate(CAND):
                    L = unitr(Zt[c]) @ Zhat.T
                    pred = np.array(CAND)[L.argmax(1)]
                    res[key]["top1"].append(dict(v=float(np.mean(pred == c)), lam=lam))

    out = []
    for (h, p), r in res.items():
        la = [x["v"] for x in r["auc"] if x["lam"] == 1e-4]
        lt = [x["v"] for x in r["top1"] if x["lam"] == 1e-4]
        out.append(dict(head=h, prior=p,
                        auc=float(np.nanmean(la)), top1=float(np.nanmean(lt)),
                        auc_bestlam=float(np.nanmean([x["v"] for x in r["auc"]])),
                        **{f"auc_{k}": float(np.nanmean(v)) if v else float("nan")
                           for k, v in r["band"].items()}))

    for h in ("ridge", "rf", "svm"):
        sub = sorted([r for r in out if r["head"] == h], key=lambda x: -x["auc"])
        print(f"\n==== head = {h}  (auc_dis 固定 lam=1e-4) ====")
        print(f"{'prior':10s} {'auc':>8s} {'top1':>8s} {'low':>7s} {'mid':>7s} {'high':>7s}")
        for r in sub:
            print(f"{r['prior']:10s} {r['auc']:8.4f} {r['top1']:8.4f} "
                  f"{r['auc_low']:7.3f} {r['auc_mid']:7.3f} {r['auc_high']:7.3f}")

    json.dump(dict(rows=out, oracle=oracle, cand=CAND, others=others),
              open(os.path.join(OUT, "exp88d_zs.json"), "w"), indent=1)
    print("\n[exp88d] saved out/exp88d_zs.json")


if __name__ == "__main__":
    main()
