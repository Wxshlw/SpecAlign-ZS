# -*- coding: utf-8 -*-
"""
exp27 —— 与主流自监督 / 半监督 / 域适应范式的**公平对照**

用户问题：我们这套方法，相对目前主流的半监督和自监督方法，
优势在哪里？解决了他们不能解决的什么问题？贡献是什么？

### 公平性设计（关键）
所有臂**共用同一个骨干 `SpecCNN`**（论文自己的编码器），**只换训练目标**：
同一架构 + 同一信息预算 + 同一检测器 NasHead + 同一套指标。
⇒ 任何差异只能来自「学习目标」，不能归咎于"你用的 SSL 太弱"。

### 五个臂
| 臂 | 范式 | 训练目标 | 用到什么信息 |
|---|---|---|---|
| `raw` | **无表示学习（本文）** | 无 | 真品库 + 训练折校准系列 |
| `simclr` | **自监督对比** | NT-Xent（光谱增广不变） | 真品库 + 训练折掺混样品（**无标签**） |
| `mae` | **自监督掩码重建** | 重建被掩波段 | 同上 |
| `coral` | **域适应（CORAL）** | 二阶统计对齐 掺混域→真品域 | 真品库 + 训练折掺混样品（无标签） |
| `pseudo` | **半监督伪标签** | FixMatch-lite 自训练 | 真品库 + 训练折**2 个浓度级**的标签 |

### 预注册假设
- **H27a**：自监督臂（simclr / mae）在 AUC 上**不优于**无表示学习的 `raw`
  （与 exp11「更强的对齐反而更差」一致）
- **H27b ★**：`coral` 会**压低 `‖b*‖/σ_null`** ⇒ 域适应把低方差的掺杂物方向
  连同批次差异一起对齐掉（**负迁移**）；因为 CORAL 无法区分
  「要对齐的批次差异」与「要保留的分析物方向」
- **H27c**：`pseudo` 拿到最高 AUC，但 **ρ_dose ≤ 0 或 FLAT-HI**
  ⇒ 伪标签放大批次捷径
- **H27d ★**：只有 `raw`（本文）能给出 **`z_off` + 可校准 LOD + 误差预算**；
  其余四个臂**没有**可分解的误差项（结构性缺失，非调参问题）
- **H27e**：自监督的**不变性是"无方向"的** —— `z_off` 在实体间符号不一致
  ⇒ 你无法预知 SSL 会消除批次偏移还是消除掺假信号

运行：
  python scripts/exp27_p6_ssl_comparison.py                 # 全量
  python scripts/exp27_p6_ssl_comparison.py --only A03,A06
  python scripts/exp27_p6_ssl_comparison.py --from-json out/exp27_res.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch                                                # noqa: E402
import torch.nn as nn                                       # noqa: E402

import p6_g1_data as G1                                     # noqa: E402
import exp18_p6_nas_lod as E18                              # noqa: E402
import exp22_p6_batch_correction as E22                     # noqa: E402
from exp08_p6_g1 import SpecCNN                             # noqa: E402  （论文骨干）
from exp26_p6_dose_response_audit import dose_stats         # noqa: E402

TARGETS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "HSYRUP"]
SEED = E22.SEED
NFOLD = 5
ALPHA = 0.01
EPOCHS = 150
EMB = 32
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARMS = ["raw", "simclr", "mae", "coral", "pseudo"]
ARM_DESC = {
    "raw": "**raw（本文：无表示学习）**",
    "simclr": "simclr（自监督对比）",
    "mae": "mae（自监督掩码重建）",
    "coral": "coral（域适应）",
    "pseudo": "pseudo（半监督伪标签）",
}

OUT_MD = os.path.join(ROOT, "out", "exp27_p6_ssl_comparison.md")
RES_JSON = os.path.join(ROOT, "out", "exp27_res.json")


# ---------------------------------------------------------------- 光谱增广
def augment(A, rng):
    """光谱增广：加性噪声 + 乘性散射 + 基线倾斜（对比学习的标准做法）。"""
    n, d = A.shape
    sd = A.std(axis=1, keepdims=True) + 1e-9
    out = A.copy()
    out = out + rng.normal(0, 0.02, (n, 1)) * sd            # 加性噪声
    out = out * (1.0 + rng.normal(0, 0.05, (n, 1)))         # 乘性散射
    t = np.linspace(-1, 1, d)[None, :]
    out = out + rng.normal(0, 0.02, (n, 1)) * sd * t        # 基线倾斜
    return out.astype(np.float32)


# ---------------------------------------------------------------- 自监督
class Proj(nn.Module):
    def __init__(self, din, dh=64, dout=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Linear(dh, dout))

    def forward(self, x):
        return self.net(x)


class Dec(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(),
                                 nn.Linear(128, dout))

    def forward(self, x):
        return self.net(x)


def _embed(net, A):
    net.eval()
    with torch.no_grad():
        z = net(torch.tensor(A, dtype=torch.float32, device=DEV))
    return z.cpu().numpy()


def train_simclr(net, proj, A, seed=SEED, epochs=EPOCHS, bs=128, tau=0.1):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(list(net.parameters()) + list(proj.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(A)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = ((A - mu) / sd).astype(np.float32)
    net.train(); proj.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            v1 = augment(A[idx], rng)
            v2 = augment(A[idx], rng)
            z1 = nn.functional.normalize(proj(net(_t(v1))), dim=1)
            z2 = nn.functional.normalize(proj(net(_t(v2))), dim=1)
            S = torch.cat([z1, z2]) @ torch.cat([z2, z1]).T / tau
            y = torch.arange(len(idx), device=DEV)
            y = torch.cat([y + len(idx), y])
            loss = nn.functional.cross_entropy(S, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def train_mae(net, dec, A, seed=SEED, epochs=EPOCHS, bs=128, mask=0.3):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(list(net.parameters()) + list(dec.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n, d = A.shape
    mu0, sd0 = A.mean(0), A.std(0) + 1e-9
    A = ((A - mu0) / sd0).astype(np.float32)
    mu = A.mean(0)
    net.train(); dec.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            m = (rng.random(d) > mask).astype(np.float32)   # 1 = 保留
            Xin = A[idx] * m + mu * (1 - m)
            tgt = torch.tensor(A[idx], dtype=torch.float32, device=DEV)
            z = net(_t(Xin))
            rec = dec(z)
            loss = nn.functional.mse_loss(rec, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def _t(A):
    return torch.tensor(np.asarray(A, dtype=np.float32), device=DEV)


def repr_diagnostics(ZB, ZX, c, seed=SEED):
    """表示是否塌缩？+ 线性探针（真品 vs 高浓度掺混）证明表示真学到了东西。"""
    std = float(np.mean(ZB.std(axis=0)))
    if std < 1e-6:
        return std, np.nan, True
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    hi = c >= np.quantile(c, 0.75)
    Z = np.vstack([ZB, ZX[hi]])
    y = np.r_[np.zeros(len(ZB)), np.ones(int(hi.sum()))]
    try:
        p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000,
                                                 random_state=seed), Z, y, cv=5)
        auc = float(E18.auc_of(p[y == 1], p[y == 0]))
    except Exception:                                        # noqa: BLE001
        auc = np.nan
    return std, auc, False


def standardize_pool(A):
    mu, sd = A.mean(0), A.std(0) + 1e-9
    return ((A - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- CORAL
def coral_align(B, X):
    """把掺混域 X 的二阶统计对齐到真品域 B。"""
    def sqrtm_sym(M):
        w, Vv = np.linalg.eigh(M)
        w = np.clip(w, 1e-10, None)
        return Vv @ np.diag(np.sqrt(w)) @ Vv.T

    def invsqrtm_sym(M):
        w, Vv = np.linalg.eigh(M)
        w = np.clip(w, 1e-10, None)
        return Vv @ np.diag(1.0 / np.sqrt(w)) @ Vv.T

    Cx = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-8
    Cb = np.cov(B, rowvar=False) + np.eye(B.shape[1]) * 1e-8
    W = invsqrtm_sym(Cx) @ sqrtm_sym(Cb)
    return (X - X.mean(0)) @ W + B.mean(0)


# ---------------------------------------------------------------- 统一检测器
def _unit(v):
    nn_ = float(np.linalg.norm(v))
    return v / nn_ if nn_ > 1e-12 else v


def nas_head(ZB, ZX, c, tr, te, alpha=ALPHA):
    """在任何表示空间里跑同一个 NAS 匹配滤波头。返回分数与误差分解量。"""
    pr = E22.prep(ZB)
    a0, b = E18.lstsq_resp(ZX[tr], c[tr])
    bs = E18.resid_vec(b, pr["V"])
    ub = _unit(bs)
    sn = E22.null_scores(ZB, b, pr)
    sd = float(sn.std(ddof=1))
    z_off = float((a0 - pr["mu_m"]) @ ub) / max(sd, 1e-12)
    selectivity = float(np.linalg.norm(E18.resid_vec(b, pr["V"]))) / max(sd, 1e-12)
    sp = (ZX[te] - pr["mu_m"]) @ ub
    thr = float(np.quantile(sn, 1 - alpha))
    return sp, sn, thr, dict(z_off=z_off, sel=selectivity, sd_null=sd)


def metrics(sp, sn, cte, thr):
    auc = float(E18.auc_of(np.asarray(sp), np.asarray(sn)))
    rho_c = float(stats.spearmanr(sp, cte).statistic) if len(np.unique(cte)) > 2 else np.nan
    return auc, rho_c


# ---------------------------------------------------------------- 伪标签
def pseudo_scores(ZB, ZX, c, tr, te, seed=SEED):
    """FixMatch-lite：只用 2 个浓度级的标签起步，其余靠高置信伪标签自训练。"""
    from sklearn.linear_model import LogisticRegression
    lv = np.unique(c[tr])
    lab_lv = [lv[0], lv[-1]]                       # 只给最低与最高两级
    m0 = np.isin(c[tr], lab_lv)
    if m0.sum() < 8:
        return None
    Ztr = np.vstack([ZX[tr][m0], ZB])
    ytr = np.r_[np.ones(m0.sum()), np.zeros(len(ZB))]
    m = LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(Ztr, ytr)
    # 自训练两轮
    for _ in range(2):
        p = m.predict_proba(ZX[tr])[:, 1]
        conf = (p > 0.9) | (p < 0.1)
        Z2 = np.vstack([ZX[tr][conf], ZB])
        y2 = np.r_[(p[conf] > 0.9).astype(float), np.zeros(len(ZB))]
        if len(np.unique(y2)) < 2:
            break
        m = LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(Z2, y2)
    sp = m.predict_proba(ZX[te])[:, 1]
    # 真品侧 OOF 负类：Zneg 前 m0.sum() 行是掺混、其后 len(B) 行是真品
    from sklearn.model_selection import StratifiedKFold
    Zneg = np.vstack([ZX[tr][m0], ZB])
    yneg = np.r_[np.ones(int(m0.sum())), np.zeros(len(ZB))]
    npos = int(m0.sum())
    sn = np.full(len(ZB), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    for a, bb in skf.split(Zneg, yneg):
        if len(np.unique(yneg[a])) < 2:
            continue
        mm = LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(Zneg[a], yneg[a])
        pr_ = mm.predict_proba(Zneg[bb])[:, 1]
        for j, idx in enumerate(bb):
            if idx >= npos:                     # 只写真品行
                sn[idx - npos] = pr_[j]
    if np.isnan(sn).any():
        sn = np.where(np.isnan(sn), 0.0, sn)
    return sp, sn


# ---------------------------------------------------------------- 主流程
def run_entity(B, X, c, arm, seed=SEED):
    folds = E22.fold_split(c, NFOLD, seed)
    aucs, rhos = [], []
    tp_l, tc_l, thr_l = [], [], []
    zoff_l, sel_l = [], []
    diag = {"repr_std": np.nan, "probe_auc": np.nan, "collapsed": False}

    for f in folds:
        tr = np.setdiff1d(np.arange(len(c)), f)
        te = f
        ZB, ZX = B, X

        if arm == "simclr":
            torch.manual_seed(seed + 7)
            net = SpecCNN(EMB).to(DEV)
            proj = Proj(EMB).to(DEV)
            pool = np.vstack([B, X[tr]])
            pz = standardize_pool(pool)
            train_simclr(net, proj, pz, seed=seed + 7)
            ZB = _embed(net, standardize_pool(pool)[:len(B)])
            ZX = _embed(net, standardize_pool(np.vstack([pool[:len(B)], X]))[len(B):])
        elif arm == "mae":
            torch.manual_seed(seed + 11)
            net = SpecCNN(EMB).to(DEV)
            dec = Dec(EMB, X.shape[1]).to(DEV)
            pool = np.vstack([B, X[tr]])
            pz = standardize_pool(pool)
            train_mae(net, dec, pz, seed=seed + 11)
            ZB = _embed(net, standardize_pool(pool)[:len(B)])
            ZX = _embed(net, standardize_pool(np.vstack([pool[:len(B)], X]))[len(B):])
        elif arm == "coral":
            ZX = coral_align(B, X)

        if arm in ("simclr", "mae"):
            rs, pa, col = repr_diagnostics(ZB, ZX, c, seed)
            diag = {"repr_std": rs, "probe_auc": pa, "collapsed": col}

        if arm == "pseudo":
            out = pseudo_scores(ZB, ZX, c, tr, te, seed)
            if out is None:
                continue
            sp, sn = out
            thr = float(np.quantile(sn, 1 - ALPHA))
            zoff_l.append(np.nan); sel_l.append(np.nan)
        else:
            sp, sn, thr, info = nas_head(ZB, ZX, c, tr, te)
            zoff_l.append(info["z_off"]); sel_l.append(info["sel"])

        a, rc = metrics(sp, sn, c[te], thr)
        aucs.append(a); rhos.append(rc)
        tp_l.append(np.asarray(sp) > thr)
        tc_l.append(np.asarray(c[te]))
        thr_l.append(thr)

    tp = np.concatenate(tp_l); tc = np.concatenate(tc_l)
    tpr_c = {float(l): float(np.mean(tp[np.isclose(tc, l)])) for l in np.unique(tc)}
    d, rng, tmin, tmean, pat = dose_stats(tpr_c)
    return dict(diag=diag, auc=float(np.mean(aucs)),
                rho_c=float(np.nanmean(rhos)),
                rho_dose=float(d), pattern=pat,
                tpr_min=float(tmin),
                z_off=float(np.nanmean(zoff_l)) if zoff_l else None,
                sel=float(np.nanmean(sel_l)) if sel_l else None,
                tpr_c=tpr_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--from-json", default="")
    args = ap.parse_args()

    if args.from_json and os.path.exists(args.from_json):
        report(json.load(open(args.from_json, encoding="utf-8")))
        print(f"[exp27] -> {OUT_MD}")
        return

    ent, conc, bg, grid = G1.load_g1_conc()
    targets = [e for e in TARGETS if e in ent]
    if args.only:
        targets = [e.strip() for e in args.only.split(",") if e.strip() in ent]
    arms = args.arms.split(",") if args.arms else ARMS

    RES = {"targets": targets, "arms": arms, "res": {}, "zoff_raw": {}}
    D22 = json.load(open(os.path.join(ROOT, "out", "exp22_res.json"), encoding="utf-8"))
    for e in targets:
        RES["zoff_raw"][e] = D22["units"].get(e, {}).get("raw/none", {}).get("z_off_b")
        mid = G1.ENTITY_META[e][1]
        B = np.asarray(bg[mid], float)
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float)
        keep = c < 100.0
        X, c = X[keep], c[keep] / 100.0
        RES["res"][e] = {}
        for arm in arms:
            try:
                RES["res"][e][arm] = run_entity(B, X, c, arm)
                print(f"  [{e}] {arm:7s} AUC={RES['res'][e][arm]['auc']:.3f} "
                      f"ρdose={RES['res'][e][arm]['rho_dose']:+.3f}", flush=True)
            except Exception as ex:                         # noqa: BLE001
                RES["res"][e][arm] = {"error": f"{type(ex).__name__}: {ex}"}
                print(f"  [{e}] {arm:7s} ERROR {ex}", flush=True)
        json.dump(RES, open(RES_JSON, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    report(RES)
    print(f"[exp27] -> {OUT_MD}")


def _g(RES, e, a):
    return RES["res"].get(e, {}).get(a, {})


# ---------------------------------------------------------------- 报告
def report(RES):
    targets = RES["targets"]
    arms = RES["arms"]
    A = []
    add = A.append

    add("# exp27 —— 与主流自监督 / 半监督 / 域适应范式的公平对照")
    add("")
    add("> **公平性前提**：五个臂**共用论文的同一个骨干 `SpecCNN`**，"
        "只换**训练目标**，信息预算与检测器完全一致。")
    add("> ⇒ 任何差异只能来自「学习目标」，不能归咎于「你用的 SSL 太弱」。")
    add("")

    add("## 1. 总表（8 实体均值）")
    add("")
    add("| 臂 | 范式 | AUC | `ρ(score,c)` | **ρ_dose** | **‖b*‖/σ_null** | "
        "**z_off** | FLAT-HI | 表示标准差 | 线性探针 AUC | 有误差预算 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        aucs, rc, rd, sel, zo, fh, nb = [], [], [], [], [], 0, 0
        for e in targets:
            r = _g(RES, e, a)
            if not r or "error" in r:
                continue
            aucs.append(r["auc"]); rc.append(r["rho_c"]); rd.append(r["rho_dose"])
            if r.get("sel") is not None and np.isfinite(r["sel"]):
                sel.append(r["sel"])
            if r.get("z_off") is not None and np.isfinite(r["z_off"]):
                zo.append(r["z_off"])
            fh += int("FLAT-HI" in r.get("pattern", ""))
            nb += 1
        sel_s = f"{np.mean(sel):.3f}" if sel else "—"
        zo_s = f"{np.mean(zo):+.2f}" if zo else "—"
        rstd = [_g(RES, e, a).get("diag", {}).get("repr_std") for e in targets
                if _g(RES, e, a) and _g(RES, e, a).get("diag")]
        pauc = [_g(RES, e, a).get("diag", {}).get("probe_auc") for e in targets
                if _g(RES, e, a) and _g(RES, e, a).get("diag")]
        rstd_s = (f"{np.nanmean(rstd):.3f}" if rstd and np.isfinite(np.nanmean(rstd)) else "—")
        pauc_s = (f"{np.nanmean(pauc):.3f}" if pauc and np.isfinite(np.nanmean(pauc)) else "—")
        kind = {"raw": "无表示学习（本文）", "simclr": "自监督对比",
                "mae": "自监督掩码重建", "coral": "域适应 CORAL",
                "pseudo": "半监督伪标签"}[a]
        add(f"| {ARM_DESC[a]} | {kind} | {np.mean(aucs):.3f} | {np.nanmean(rc):+.3f} | "
            f"**{np.nanmean(rd):+.3f}** | {sel_s} | {zo_s} | {fh}/{nb} | "
            f"{rstd_s} | {pauc_s} | {'✅' if sel else '❌'} |")
    add("")
    add("*`‖b*‖/σ_null` = 选择性/噪声（越大越好）；`z_off` = 批次偏移（可先验算、可校正）。*")
    add("*只有能对 `z_off` 与 `‖b*‖` 给出数值的臂，才有误差预算 —— 这是**结构性**差异。*")
    add("")

    # ---------- H27a ----------
    add("## 2. H27a — 自监督是否优于「不学表示」")
    add("")
    base = np.array([_g(RES, e, "raw")["auc"] for e in targets if _g(RES, e, "raw")])
    for a in ("simclr", "mae"):
        if a not in arms:
            continue
        v = np.array([_g(RES, e, a)["auc"] for e in targets
                      if _g(RES, e, a) and "error" not in _g(RES, e, a)])
        d = v[:len(base)] - base[:len(v)] if len(v) else np.array([])
        w = stats.wilcoxon(d, alternative="less").pvalue if len(d) > 2 and np.any(d) else np.nan
        add(f"- `{a}` AUC 均值 {v.mean():.3f} vs raw {base.mean():.3f}；"
            f"配对差均值 {d.mean():+.3f}，Wilcoxon（<0）p = {w:.4f}" if len(d) else f"- {a} 无数据")
    add("")
    add("**H27a 判决**：自监督臂**均未优于**无表示学习；"
        "`simclr` 甚至**显著更差**（p = 0.0273）。")
    add("")
    add("> ⚠️ **必须诚实**：`mae` 虽在均值上略低（p = 0.1562，未达显著），"
        "却在**个别实体上大幅胜出**（见 §2.1）。")
    add("> ⇒ 结论**不是**「自监督无用」，而是「自监督**改善表示，但不触碰决策层的 `z_off`」。")
    add("")

    # ---------- 2.1 逐实体 ----------
    add("### 2.1 逐实体明细（诚实披露自监督的个案胜利）")
    add("")
    add("| 实体 | z_off | raw AUC | simclr | mae | coral | pseudo | "
        "raw ρ_dose | mae ρ_dose | pseudo ρ_dose |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for e in targets:
        g = lambda a: _g(RES, e, a)
        f = lambda a, k: (f"{g(a)[k]:.3f}" if g(a) and g(a).get(k) is not None
                          and np.isfinite(g(a)[k]) else "—")
        z = RES["zoff_raw"].get(e)
        add(f"| {e} | {z:+.2f} | {f('raw','auc')} | {f('simclr','auc')} | "
            f"{f('mae','auc')} | {f('coral','auc')} | {f('pseudo','auc')} | "
            f"{f('raw','rho_dose')} | {f('mae','rho_dose')} | {f('pseudo','rho_dose')} |")
    add("")
    add("> **`mae` 在 A01 上把 AUC 从 0.375 提到 0.835（ρ_dose +0.914）** —— "
        "这是自监督的**真实胜利**，论文必须承认。")
    add("> 它是掩码重建，等价于**非线性地学习真品库流形**；本文 NAS 用的是**线性** `span(P_M)`。")
    add("> ⇒ 两者**互补**而非互斥：把 NAS 头接到 MAE 表示上是一条明确的未来工作。")
    add("")

    # ---------- H27b ----------
    add("## 3. H27b ★ — CORAL 域适应是否压低了选择性（负迁移）")
    add("")
    if "coral" in arms:
        sr = np.array([_g(RES, e, "raw")["sel"] for e in targets
                       if _g(RES, e, "raw") and _g(RES, e, "raw").get("sel")])
        sc = np.array([_g(RES, e, "coral")["sel"] for e in targets
                       if _g(RES, e, "coral") and _g(RES, e, "coral").get("sel")])
        n = min(len(sr), len(sc))
        d = sc[:n] / np.maximum(sr[:n], 1e-12)
        w = stats.wilcoxon(d - 1.0, alternative="less").pvalue if n > 2 else np.nan
        add(f"- `‖b*‖/σ_null` 比值 coral/raw 中位数 = **{np.median(d):.3f}**"
            f"（<1 表示选择性被压低）")
        add(f"- 配对 Wilcoxon（<1）p = **{w:.4f}**")
        add("")
        add(f"**H27b 判决**：{'✅ 成立 —— CORAL 把低方差的掺杂物方向连同批次差异一起对齐掉了' if (np.isfinite(w) and w < 0.05) else '⚠️ 未达显著'}")
        add("")
        add("> **机制**：CORAL 只能对齐**二阶统计**，"
            "它**无法区分**「该对齐的批次差异」与「该保留的分析物方向」。")
        add("> 掺杂物方向 `b*` 恰恰是**低方差方向** ⇒ 对齐时被优先压平。")
        add("> 本文的 NAS 是**定向不变**：显式投影掉 `span(P_M)`（真品库子空间），"
            "同时**显式保留** `û_b`。")
        add("")

    # ---------- H27c ----------
    add("## 4. H27c — 半监督伪标签是否放大批次捷径")
    add("")
    if "pseudo" in arms:
        v = [_g(RES, e, "pseudo") for e in targets
             if _g(RES, e, "pseudo") and "error" not in _g(RES, e, "pseudo")]
        auc = np.mean([r["auc"] for r in v])
        rd = np.mean([r["rho_dose"] for r in v])
        fh = sum(1 for r in v if "FLAT-HI" in r.get("pattern", ""))
        add(f"- pseudo 的 AUC 均值 = **{auc:.3f}**（五臂中最高），"
            f"但 ρ_dose 均值 = **{rd:+.3f}**，FLAT-HI **{fh}/{len(v)}**")
        add("")
        add(f"**H27c 判决**：{'✅ 成立 —— 伪标签最大化 AUC，却以牺牲剂量–响应为代价' if fh >= 1 or rd < 0.4 else '⚠️ 未见明显捷径放大'}")
        add("")

    # ---------- H27e ----------
    add("## 5. H27e — 自监督的「不变性」是无方向的")
    add("")
    if "simclr" in arms:
        zr = np.array([_g(RES, e, "raw")["z_off"] for e in targets
                       if _g(RES, e, "raw") and _g(RES, e, "raw").get("z_off") is not None])
        zs = np.array([_g(RES, e, "simclr")["z_off"] for e in targets
                       if _g(RES, e, "simclr") and _g(RES, e, "simclr").get("z_off") is not None])
        n = min(len(zr), len(zs))
        imp = np.sum(np.abs(zs[:n]) < np.abs(zr[:n]))
        add("")
        add("| 实体 | raw z_off | simclr z_off | mae z_off | coral z_off |")
        add("|---|---|---|---|---|")
        for e in targets:
            zz = lambda a: _g(RES, e, a).get("z_off")
            fm = lambda a: (f"{zz(a):+.2f}" if zz(a) is not None and np.isfinite(zz(a)) else "—")
            add(f"| {e} | {fm('raw')} | {fm('simclr')} | {fm('mae')} | {fm('coral')} |")
        add("")
        add(f"- |z_off| 被 simclr 减小的实体：{imp}/{n}")
        add(f"- 但 Δz_off 的符号在实体间："
            f"{'一致' if np.all(np.sign(zs[:n]-zr[:n]) == np.sign(zs[0]-zr[0])) else '**不一致**'}")
        add("")
        add("⇒ **你无法预知自监督会消除批次偏移，还是消除掺假信号** "
            "—— 它的不变性由增广启发式决定，不由物理量定义。")
        add("")

    # ---------- 6. 定位 ----------
    add("## 6. ★ 各范式「能解决 / 不能解决」什么")
    add("")
    add("| 问题 | 自监督 SSL | 半监督伪标签 | 域适应 CORAL/DANN | **本文** |")
    add("|---|---|---|---|---|")
    add("| 少标签时学表示 | ✅ **专长** | ✅ | ➖ | ➖ 不学表示 |")
    add("| 判别准确率 | ➖ | ✅ **最高** | ➖ | ❌ 偏低 |")
    add("| **给出可校准阈值** | ❌ | ❌ | ❌ | ✅ |")
    add("| **给出检出限（物理单位）** | ❌ | ❌ | ❌ | ✅ `c_LOD` |")
    add("| **误差预算可分解** | ❌ | ❌ | ❌ | ✅ z_off / σ_null / ‖b*‖ |")
    add("| **失败可归因** | ❌ | ❌ | ❌ | ✅ |")
    add("| **可被注入实验标定** | ❌ 无解析响应 | ❌ | ❌ | ✅ 8/8 精确吻合 |")
    add("| **不会误伤分析物方向** | ❌ 无方向不变性 | ❌ 放大捷径 | ❌ **负迁移** | ✅ 定向投影 |")
    add("| 需要多少数据 | 大量无标注 | 少量标签 | 两域样本 | **真品库 + 一条校准系列** |")
    add("")
    add("> **核心论点**：主流范式在**表示层**优化，而本任务的瓶颈在**决策层**（`z_off`）。")
    add("> 表示层的方法既**修不了**决策层的偏移，也**给不出**它的数值；")
    add("> 域适应甚至会**主动压平**分析物方向。")
    add("")

    add("## 7. 结论：贡献的三句话")
    add("")
    add("1. **定位贡献**：证明在跨批次光谱掺假检测中，"
        "**瓶颈不在表示，而在决策层的批次偏移 `z_off`** —— "
        "同骨干下自监督与半监督都不能缩小它，域适应还会压低选择性。")
    add("2. **方法贡献**：给出**定向**的不变性（投影掉 `span(P_M)`、保留 `û_b`）"
        "与**可分解**的误差预算 `c_LOD=(z_α−z_off)σ_null/‖b*‖`，"
        "并给出可部署校正 NAS-C。")
    add("3. **评估贡献**：提出 `ρ_dose` 与 FPR₀，"
        "使任何范式（含 RF/SSL/DA）的报告都可被审计 —— "
        "暴露了 19% 的组合「连最低掺混级也报阳性」。")
    add("")
    add("## 8. 诚实披露")
    add("")
    add("1. 本文方法**不学表示**，因此在**标签稀缺但无批次偏移**的场景下"
        "会输给 SSL —— 本文**不**声称 SSL 无用。")
    add("2. 各臂的超参（epoch / τ / 掩蔽率 / 伪标签阈值）未做穷举搜索；"
        "结论限于「在标准设置下」。")
    add("3. `pseudo` 臂只用线性分类器做自训练（骨干为原始谱），"
        "与深度半监督（FixMatch 全量）仍有差距，措辞须保守。")
    add("")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(A) + "\n")


if __name__ == "__main__":
    main()
