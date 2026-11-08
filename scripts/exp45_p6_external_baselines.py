"""exp45 —— 冷启动场景的**外部基线**对照

=============================================================================
为什么必须做（当前论文最大的硬伤）
=============================================================================
exp34–44 全部是**自家臂之间的对比**（none / center / orth / ridge / gate …）。
审稿人的第一个问题一定是：
    「和**领域里既有方法**比呢？PLS-DA 呢？谱库检索呢？」
本实验补齐这一层，且刻意选**论文所在领域的硬门槛方法**。

=============================================================================
外部方法集（按「需要多少新类信息」分层）
=============================================================================
| 方法 | 类型 | k=0 时能否建模新类 | 说明 |
|---|---|---|---|
| `plsda` | 化学计量学黄金标准 | ✗ | 领域硬门槛，Food Chem / Anal Chim Acta 必问 |
| `svm-rbf` | 通用 ML | ✗ | 非线性强基线 |
| `rf` | 通用 ML | ✗ | 树模型强基线 |
| `knn-cos` | **谱库检索** | ✗ | 传统「拿未知谱去库里比对」 |
| `centroid-cos` | **谱库检索** | ✗ | 类中心余弦，最简谱库匹配 |
| `ours-ridge` | 化学语义先验（本文） | **✓** | exp44 的主张臂，本实验重跑作锚 |

★ **核心论点（k=0 分层的关键）**
   传统方法在 k=0 时**数学上无法为新类建模**——它们需要至少一个新类样本
   才能估计该类的位置。因此 k=0 下它们的 top-1 恒为 0，这是**结构性限制**，
   不是调参问题。为了公平，k=0 时给它们换成「新类检测」口径：
        score = 1 − max_probability（或 1 − 最大余弦）
   并同样用 AUC 评估 ⇒ 这是对传统方法**有利**的降级（19 类识别 → 二分类检测）。

=============================================================================
指标（与 exp44 同口径）
=============================================================================
- `top1`：19 类 argmax 中新类召回（k=0 时传统方法恒 0）
- `auc`：新类样本 vs 老类样本在「新类得分」上的 AUC（筛查语义，主指标）

预注册判据
----------
- H45a：k=0 时 `ours-ridge` 的 top1 > 0 而所有传统方法 = 0
        ⇒ 零样本下传统方法结构性不可解（这是「不可替代」的最硬形式）
- H45b：k=0 时 `ours-ridge` 的 auc ≥ 最好的传统异常检测方法
        ⇒ 即便把任务降级为对传统方法有利的「检测」，先验仍不落后
- H45c：k∈{5,20} 时 `ours-ridge` 的 auc ≥ `plsda`
        ⇒ 少样本下不弱于领域黄金标准
- 负结果照报；不做「挑最好的外部方法」
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

OUT = r"D:/2026sci论文实验设计/论文6/out"


def auc_of(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def cos_to(A, B):
    """A 每行到 B 每行的余弦，返回 (len(A), len(B))。"""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def fit_plsda(Xtr, ytr, ncomp):
    classes = np.unique(ytr)
    Y = np.zeros((len(ytr), len(classes)))
    for i, c in enumerate(classes):
        Y[ytr == c, i] = 1.0
    sc = StandardScaler().fit(Xtr)
    pls = PLSRegression(n_components=min(ncomp, Xtr.shape[1], len(ytr) - 1))
    pls.fit(sc.transform(Xtr), Y)
    return sc, pls, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5,20")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eids", default="")
    ap.add_argument("--ncomp", default="5,10,20")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    ncomps = [int(x) for x in args.ncomp.split(",")]

    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    print(f"[exp45] 实体={eids} 基质={mids} shots={tags} seeds={args.seeds}")
    print(f"[exp45] 谱维度 = {np.asarray(ent[eids[0]]).shape}")

    res = {"acc": {t: {} for t in tags},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "methods": ["plsda", "svm-rbf", "rf", "knn-cos",
                                "centroid-cos"]}}

    rng_g = np.random.default_rng(20240101)
    for e_hold in eids:
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        names = others + [f"M_{m}" for m in mids] + [e_hold]
        NEW = 18

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new = Xe[te_ix]

                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float)); ys.append(np.full(len(bg[m]), len(others) + j))
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), 18))
                Xtr = np.concatenate(parts).astype(np.float64)
                ytr = np.concatenate(ys).astype(np.int64)

                ref = []
                for e in others:
                    Xo = np.asarray(ent[e], float)
                    ref.append(Xo[rng_g.permutation(len(Xo))[:min(200, len(Xo))]])
                for m in mids:
                    Xo = np.asarray(bg[m], float)
                    ref.append(Xo[rng_g.permutation(len(Xo))[:min(200, len(Xo))]])
                Xte_old = np.concatenate(ref).astype(np.float64)

                has_new = len(tr_e_ix) > 0
                accs = {}
                sc = StandardScaler().fit(Xtr)
                Xtr_s, Xn_s, Xo_s = sc.transform(Xtr), sc.transform(Xte_new), sc.transform(Xte_old)

                # ---------------- PLS-DA（ncomp 由内部 3 折 CV 选）----------------
                # ncomp 由训练集内部 3 折 CV 选（不碰测试集、不碰未见类）
                best_c, best_v = ncomps[0], -1.0
                idx_all = np.arange(len(Xtr_s))
                rng_c = np.random.default_rng(0)
                rng_c.shuffle(idx_all)
                folds = np.array_split(idx_all, 3)
                for c in ncomps:
                    try:
                        vs = []
                        for f in folds:
                            m = np.ones(len(Xtr_s), bool); m[f] = False
                            s2, p2, cl2 = fit_plsda(Xtr_s[m], ytr[m], c)
                            pr = cl2[p2.predict(s2.transform(Xtr_s[f])).argmax(1)]
                            vs.append(float((pr == ytr[f]).mean()))
                        v = float(np.mean(vs))
                    except Exception:
                        v = -1.0
                    if v > best_v:
                        best_v, best_c = v, c
                s_p, pls, cls = fit_plsda(Xtr_s, ytr, best_c)
                P_n = pls.predict(s_p.transform(Xn_s))
                P_o = pls.predict(s_p.transform(Xo_s))
                if has_new:
                    pred = cls[P_n.argmax(1)]
                    top1 = float((pred == NEW).mean())
                    auc = auc_of(P_n[:, list(cls).index(NEW)], P_o[:, list(cls).index(NEW)])
                else:
                    top1 = 0.0
                    auc = auc_of(-P_n.max(1), -P_o.max(1))   # 异常分：最大响应越低越异常
                accs["plsda"] = dict(top1=top1, auc=auc, ncomp=float(best_c))

                # ---------------- SVM-RBF ----------------
                sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                         random_state=0).fit(Xtr_s, ytr)
                Pn, Po = sv.predict_proba(Xn_s), sv.predict_proba(Xo_s)
                cn = list(sv.classes_)
                if has_new:
                    top1 = float((sv.predict(Xn_s) == NEW).mean())
                    auc = auc_of(Pn[:, cn.index(NEW)], Po[:, cn.index(NEW)])
                else:
                    top1 = 0.0
                    auc = auc_of(1 - Pn.max(1), 1 - Po.max(1))
                accs["svm-rbf"] = dict(top1=top1, auc=auc)

                # ---------------- Random Forest ----------------
                rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                Pn, Po = rf.predict_proba(Xn_s), rf.predict_proba(Xo_s)
                cn = list(rf.classes_)
                if has_new:
                    top1 = float((rf.predict(Xn_s) == NEW).mean())
                    auc = auc_of(Pn[:, cn.index(NEW)], Po[:, cn.index(NEW)])
                else:
                    top1 = 0.0
                    auc = auc_of(1 - Pn.max(1), 1 - Po.max(1))
                accs["rf"] = dict(top1=top1, auc=auc)

                # ---------------- 谱库检索：1-NN 余弦 / 类中心余弦 ----------------
                Cn = cos_to(Xn_s, Xtr_s); Co = cos_to(Xo_s, Xtr_s)
                lab_nn = ytr[Cn.argmax(1)]
                if has_new:
                    top1 = float((lab_nn == NEW).mean())
                    auc = auc_of(Cn[:, ytr == NEW].max(1) if (ytr == NEW).any()
                                 else Cn.max(1), Co.max(1))
                else:
                    top1 = 0.0
                    auc = auc_of(1 - Cn.max(1), 1 - Co.max(1))
                accs["knn-cos"] = dict(top1=top1, auc=auc)

                cent = np.stack([Xtr_s[ytr == j].mean(0) for j in np.unique(ytr)])
                cj = np.unique(ytr)
                Cn2 = cos_to(Xn_s, cent); Co2 = cos_to(Xo_s, cent)
                if has_new:
                    top1 = float((cj[Cn2.argmax(1)] == NEW).mean())
                    auc = auc_of(Cn2[:, list(cj).index(NEW)], Co2[:, list(cj).index(NEW)])
                else:
                    top1 = 0.0
                    auc = auc_of(1 - Cn2.max(1), 1 - Co2.max(1))
                accs["centroid-cos"] = dict(top1=top1, auc=auc)

                for m_, v in accs.items():
                    res["acc"][tag].setdefault(m_, {}).setdefault(
                        e_hold, []).append({kk: round(float(vv), 4)
                                            for kk, vv in v.items()})

        line = []
        for t in tags:
            s = " ".join(
                f"{m_}={np.mean([d['auc'] for d in res['acc'][t][m_][e_hold]]):.2f}"
                f"/{np.mean([d['top1'] for d in res['acc'][t][m_][e_hold]]):.2f}"
                for m_ in res["acc"][t] if res["acc"][t][m_].get(e_hold))
            line.append(f"{t}: {s}")
        print(f"[{e_hold:>7}] " + " | ".join(line), flush=True)

    with open(os.path.join(OUT, "exp45_baselines.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp45] 已写出 out/exp45_baselines.json")
    write_report(res, tags, eids)


def write_report(res, tags, eids):
    ms = res["meta"]["methods"]
    L = []
    w = L.append
    w("# exp45 —— 冷启动外部基线对照\n")
    w("单元格格式：**auc / top1**（auc = 筛查主指标；top1 = 19 类识别召回）。\n")
    w("k=0 时传统方法**无法为新类建模**，top1 恒 0（结构性限制）；"
      "为公平起见其 auc 改用「异常分」口径（对它们有利的降级）。\n")
    for met in ("auc", "top1"):
        w(f"\n## {met}\n")
        w("| k | " + " | ".join(f"`{m_}`" for m_ in ms) + " |")
        w("|---|" + "---|" * len(ms))
        for t in tags:
            row = []
            for m_ in ms:
                per = [np.mean([d[met] for d in res["acc"][t][m_][e]])
                       for e in eids if res["acc"][t][m_].get(e)]
                row.append(f"{np.mean(per):.3f}" if per else "—")
            w(f"| {t} | " + " | ".join(row) + " |")
    with open(os.path.join(OUT, "exp45_baselines.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[exp45] 已写出 out/exp45_baselines.md")


if __name__ == "__main__":
    main()
