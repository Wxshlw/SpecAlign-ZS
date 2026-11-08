"""exp74 —— k=5 公平性审计：RF/SVM 的高分是「记忆」还是「泛化」？

背景：group 协议（exp67/71/73）只给目标实体 T 做了按样品分组划分，
      **老类与纯基质谱仍用 rng.permutation 随机划分**（exp73 L90/L96）。
      对 KAN 三件套（50 样品 × 5 次重复扫描）这等于把同一样品的重复扫描
      同时放进训练与测试 ⇒ RF/SVM 靠记忆拿分，本文化学外推不受影响。

本实验回答两问：
  A. 把老类 + 基质也改成按样品分组划分后，RF/SVM 掉多少？（泄漏量化）
  B. k=5 时 T 只有 5 条，而 RF/SVM 用 P(T|x) 或 1−maxprob 打分，
     **稀有类偏置**是否贡献了虚高？（class_weight 平衡 / 过采样 对照）

臂（统一在 same-matrix 检出口径下评估）：
  rf           现状：k=0 用 1−maxprob；k>0 用 P(T|x)
  rf-1mp       统一 1−maxprob（消除得分口径切换）
  rf-bal       class_weight=balanced_subsample + 1−maxprob
  rf-ovs       T 类过采样到与中位类规模相当 + 1−maxprob
  svm / svm-bal 同上（SVC rbf, class_weight）
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import auc_of, tpr_at_fpr, EID2MID
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
TAU = 0.9995          # 判定「同一样品的重复扫描」的余弦阈值
GCACHE_BG = os.path.join(OUT, "exp74_bg_gid.npz")


def proxy_gid(X, tau=TAU):
    """近重复连通分量 → 代理样品 id（用于无真实 id 的纯基质谱）。"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = Xn @ Xn.T
    A = csr_matrix(G > tau)
    ncomp, lab = connected_components(A, directed=False)
    return lab, ncomp


def load_bg_gid(bg):
    """缓存版：给每个基质算代理 gid。"""
    if os.path.exists(GCACHE_BG):
        z = np.load(GCACHE_BG, allow_pickle=True)
        if set(z.files) == set(bg.keys()):
            return {m: np.asarray(z[m]) for m in bg}
    out = {}
    for m, X in bg.items():
        lab, ncomp = proxy_gid(np.asarray(X, float))
        out[m] = lab
        print(f"  [bg] {m:<10} n={len(X):<4} 代理样品组={ncomp}", flush=True)
    np.savez_compressed(GCACHE_BG, **out)
    return out


def split_group(X, gid, rng, frac=OLD_FRAC):
    """按 gid 分组划分 → (train_ix, test_ix, fallback)。

    独立样品组 < 2 时无法分组（数据退化），退回随机划分并回报 fallback。
    """
    gs = np.unique(gid)
    if len(gs) < 2:
        pp = rng.permutation(len(X))
        cut = max(1, int(round(frac * len(X))))
        return pp[:cut], pp[cut:], True
    gs = rng.permutation(gs)
    n_tr = max(1, int(round(frac * len(gs))))
    n_tr = min(n_tr, len(gs) - 1)          # 至少留一组给测试
    tr_g = list(gs[:n_tr])
    m_tr = np.isin(gid, tr_g)
    ix_tr, ix_te = np.where(m_tr)[0], np.where(~m_tr)[0]
    if len(ix_te) == 0 or len(ix_tr) == 0:
        pp = rng.permutation(len(X))
        cut = max(1, int(round(frac * len(X))))
        return pp[:cut], pp[cut:], True
    return ix_tr, ix_te, False


def metrics(sn, so):
    return dict(auc=round(float(auc_of(sn, so)), 4),
                tpr5=round(float(tpr_at_fpr(sn, so, 0.05)), 4),
                tpr20=round(float(tpr_at_fpr(sn, so, 0.20)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--protos", default="random,group")
    ap.add_argument("--out", default="exp74_fair.json")
    args = ap.parse_args()
    shots = [int(s) for s in args.shots.split(",")]
    protos = args.protos.split(",")

    t0 = time.time()
    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    BG_GID = load_bg_gid(bg)
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    print(f"[exp74] 实体={eids} protos={protos} shots={shots} "
          f"seeds={args.seeds} 载入 {time.time()-t0:.0f}s", flush=True)

    res = {"M": {}, "D": {}}
    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]

        for proto in protos:
            for k in shots:
                tag = str(k)
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed * 7919 + 13)
                    te_ix, pool_ix = split_by_group(Xe, gid_e, rng)
                    Xte_new = Xe[te_ix]
                    tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                    parts, ys = [], []
                    old_ref, mid_ref = {}, {}
                    dup_stat = {}
                    n_fb = 0
                    for j, e in enumerate(others):
                        Xo = np.asarray(ent[e], float)
                        if proto == "group" and e in GR:
                            a, b, fb = split_group(Xo, GR[e]["gid"], rng)
                            n_fb += int(fb)
                        else:
                            pp = rng.permutation(len(Xo))
                            cut = max(2, int(round(OLD_FRAC * len(Xo))))
                            a, b = pp[:cut], pp[cut:]
                        old_ref[e] = Xo[b]
                        parts.append(Xo[a]); ys.append(np.full(len(a), j))
                    for j, m in enumerate(mids):
                        Xo = np.asarray(bg[m], float)
                        gg = BG_GID[m]
                        if proto == "group":
                            a, b, fb = split_group(Xo, gg, rng)
                            n_fb += int(fb)
                        else:
                            pp = rng.permutation(len(Xo))
                            cut = max(2, int(round(OLD_FRAC * len(Xo))))
                            a, b = pp[:cut], pp[cut:]
                        mid_ref[m] = Xo[b]
                        parts.append(Xo[a])
                        ys.append(np.full(len(a), len(others) + j))
                        # 诊断：训练/测试基质谱的最大余弦
                        if m == mid_h and len(a) and len(b):
                            A = Xo[a] / (np.linalg.norm(Xo[a], axis=1,
                                                        keepdims=True) + 1e-12)
                            B = Xo[b] / (np.linalg.norm(Xo[b], axis=1,
                                                        keepdims=True) + 1e-12)
                            dup_stat[m] = round(float((A @ B.T).max()), 5)
                    NEW = len(others) + len(mids)
                    if len(tr_e_ix):
                        parts.append(Xe[tr_e_ix])
                        ys.append(np.full(len(tr_e_ix), NEW))
                    Xtr = np.concatenate(parts).astype(np.float32)
                    ytr = np.concatenate(ys).astype(np.int64)
                    Xte_old = np.concatenate(list(old_ref.values())
                                             + list(mid_ref.values())).astype(
                        np.float32)
                    Xn_neg = mid_ref[mid_h]
                    if len(Xn_neg) < 5:
                        continue

                    # k-shot 与测试谱的近重复
                    if len(tr_e_ix):
                        A = Xe[tr_e_ix] / (np.linalg.norm(
                            Xe[tr_e_ix], axis=1, keepdims=True) + 1e-12)
                        B = Xte_new / (np.linalg.norm(
                            Xte_new, axis=1, keepdims=True) + 1e-12)
                        nn_shot = round(float((A @ B.T).max()), 5)
                    else:
                        nn_shot = None

                    sc = StandardScaler().fit(Xtr)
                    Xtr_s = sc.transform(Xtr)
                    Xn_s = sc.transform(Xte_new)
                    Xo_s = sc.transform(Xn_neg)

                    cnt = np.bincount(ytr, minlength=NEW + 1)
                    med_n = int(np.median([cnt[j] for j in range(NEW) if cnt[j]]))
                    arms = {}
                    for bal in (False, True):
                        cw = "balanced_subsample" if bal else None
                        rf = RandomForestClassifier(
                            300, random_state=0, n_jobs=-1,
                            class_weight=cw).fit(Xtr_s, ytr)
                        Pn = rf.predict_proba(Xn_s)
                        Po = rf.predict_proba(Xo_s)
                        one_mp = (1 - Pn.max(1), 1 - Po.max(1))
                        suf = "-bal" if bal else ""
                        arms["rf-1mp" + suf] = one_mp
                        if len(tr_e_ix):
                            i_rf = list(rf.classes_).index(NEW)
                            arms["rf" + suf] = (Pn[:, i_rf], Po[:, i_rf])
                        else:
                            arms["rf" + suf] = one_mp
                    # 过采样：把 T 类补到中位规模
                    if len(tr_e_ix):
                        ixT = np.where(ytr == NEW)[0]
                        need = max(0, med_n - len(ixT))
                        add = resample(Xtr_s[ixT], n_samples=need,
                                       replace=True, random_state=0)
                        Xa = np.concatenate([Xtr_s, add])
                        ya = np.concatenate([ytr, np.full(need, NEW)])
                        rf2 = RandomForestClassifier(
                            300, random_state=0, n_jobs=-1).fit(Xa, ya)
                        Qn = rf2.predict_proba(Xn_s)
                        Qo = rf2.predict_proba(Xo_s)
                        arms["rf-ovs"] = (1 - Qn.max(1), 1 - Qo.max(1))
                        i2 = list(rf2.classes_).index(NEW)
                        arms["rf-ovs-pT"] = (Qn[:, i2], Qo[:, i2])

                    for bal in (False, True):
                        sv = SVC(kernel="rbf", C=10, gamma="scale",
                                 probability=True, random_state=0,
                                 class_weight=("balanced" if bal else None)
                                 ).fit(Xtr_s, ytr)
                        Qn = sv.predict_proba(Xn_s)
                        Qo = sv.predict_proba(Xo_s)
                        suf = "-bal" if bal else ""
                        arms["svm" + suf] = (1 - Qn.max(1), 1 - Qo.max(1))
                        if len(tr_e_ix):
                            i_sv = list(sv.classes_).index(NEW)
                            arms["svm-pT" + suf] = (Qn[:, i_sv], Qo[:, i_sv])

                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(proto, {}).setdefault(
                            tag, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(metrics(sn, so))
                    res["D"].setdefault(proto, {}).setdefault(
                        tag, {}).setdefault(e_hold, []).append(
                        dict(nn_shot=nn_shot, nn_mid=dup_stat.get(mid_h),
                             n_bg=len(bg[mid_h]),
                             n_bg_grp=int(len(set(BG_GID[mid_h].tolist()))),
                             n_tr=len(Xtr), med_cls=med_n,
                             n_fallback=n_fb))

        line = []
        for proto in protos:
            for arm in ("rf", "rf-1mp", "rf-bal"):
                d = res["M"].get(proto, {}).get("5", {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{proto[:3]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp74] 已写出 out/{args.out}")
    write_report(res, eids, protos, shots)


ARMS_SHOW = ["rf", "rf-1mp", "rf-bal", "rf-ovs", "rf-ovs-pT",
             "svm", "svm-pT", "svm-bal"]


def write_report(res, eids, protos, shots):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp74 —— k=5 公平性审计：RF/SVM 的高分是记忆还是泛化？\n")
    w("**核心修复**：exp67/71/73 的 group 协议只给目标实体 T 分组划分，"
      "**老类与纯基质谱仍是随机划分**。\n对 KAN 三件套（50 样品 × 5 次重复）"
      "这会把同一样品的重复扫描同时放进训练与测试。\n"
      "本实验把**老类 + 基质也改成按样品分组划分**，量化 RF/SVM 的掉落。\n")
    w("评估口径 = `same-matrix` 检出（T 的掺混谱 vs 同基质纯谱）。\n")

    def agg(proto, tag, arm, met="auc", pool=None):
        per = {}
        for e in (pool if pool is not None else eids):
            vs = [x[met] for x in res["M"].get(proto, {}).get(
                tag, {}).get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    # ---- 数据重复结构 ----
    w("\n## 〇、纯基质谱的重复结构（τ=0.9995 近重复归并）\n")
    w("| 实体 T | 基质 | 谱数 | 独立样品组 | 重复倍数 | 状态 |")
    w("|---|---|---|---|---|---|")
    clean = []
    for e in eids:
        ds = res["D"].get(protos[0], {}).get(str(shots[0]), {}).get(e, [])
        if not ds:
            continue
        d = ds[0]
        rep = d["n_bg"] / max(1, d["n_bg_grp"])
        ok = d["n_bg_grp"] >= 10
        if ok:
            clean.append(e)
        w(f"| {e} | {EID2MID[e]} | {d['n_bg']} | {d['n_bg_grp']} | {rep:.1f}× "
          f"| {'可用' if ok else '**退化（检出数字不可信）**'} |")
    w(f"\n可用实体（基质独立组 ≥10）= **{clean}**；"
      f"退化实体 = **{[e for e in eids if e not in clean]}**\n")

    w("\n## 一、近重复诊断（训练 ↔ 测试 最大余弦）\n")
    w("| 实体 T | 协议 | k-shot↔测试 | 训练基质↔测试基质 |")
    w("|---|---|---|---|")
    for e in eids:
        for proto in protos:
            ds = res["D"].get(proto, {}).get(str(shots[-1]), {}).get(e, [])
            if not ds:
                continue
            nn = [d["nn_shot"] for d in ds if d["nn_shot"] is not None]
            mm = [d["nn_mid"] for d in ds if d["nn_mid"] is not None]
            w(f"| {e} | {proto} | "
              f"{np.mean(nn):.4f} | {np.mean(mm):.4f} |"
              if nn and mm else f"| {e} | {proto} | — | — |")

    # ---- 主表 ----
    for tag in [str(s) for s in shots]:
        w(f"\n## 二、same-matrix 检出 · auc（k={tag}）\n")
        w("| 协议 | " + " | ".join(f"`{a}`" for a in ARMS_SHOW) + " |")
        w("|---|" + "---|" * len(ARMS_SHOW))
        for proto in protos:
            row = []
            for a in ARMS_SHOW:
                d = agg(proto, tag, a)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w(f"| {proto} | " + " | ".join(row) + " |")
        w(f"\n## 二b、same-matrix 检出 · tpr5（k={tag}）\n")
        w("| 协议 | " + " | ".join(f"`{a}`" for a in ARMS_SHOW) + " |")
        w("|---|" + "---|" * len(ARMS_SHOW))
        for proto in protos:
            row = []
            for a in ARMS_SHOW:
                d = agg(proto, tag, a, "tpr5")
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w(f"| {proto} | " + " | ".join(row) + " |")

    # ---- 判据 ----
    w("\n## 三、判据\n")
    if len(protos) >= 2:
        p0, p1 = protos[0], protos[1]
        for tag in [str(s) for s in shots]:
            for a in ("rf", "rf-1mp", "rf-bal", "svm"):
                x = agg(p0, tag, a); y = agg(p1, tag, a)
                com = sorted(set(x) & set(y))
                if len(com) < 3:
                    continue
                xa = np.array([x[e] for e in com])
                ya = np.array([y[e] for e in com])
                w(f"- **H74a** k={tag} `{a}` auc：{p0} {xa.mean():.3f} → "
                  f"{p1} {ya.mean():.3f} (Δ={np.mean(ya-xa):+.3f}, "
                  f"p={ttest_rel(ya, xa).pvalue:.4g}, n={len(com)}) ⇒ "
                  f"{'泄漏显著' if ttest_rel(ya, xa).pvalue < 0.05 else '不显著'}")
    w("")
    for proto in protos:
        for tag in [str(s) for s in shots]:
            base = agg(proto, tag, "rf")
            for a in ("rf-1mp", "rf-bal", "rf-ovs", "rf-ovs-pT"):
                y = agg(proto, tag, a)
                com = sorted(set(base) & set(y))
                if len(com) < 3:
                    continue
                xb = np.array([base[e] for e in com])
                yb = np.array([y[e] for e in com])
                w(f"- **H74b** k={tag} {proto}：rf {xb.mean():.3f} → `{a}` "
                  f"{yb.mean():.3f} (Δ={np.mean(yb-xb):+.3f}, "
                  f"p={ttest_rel(yb, xb).pvalue:.4g}, n={len(com)})")

    # ---- 仅可用实体 ----
    w("\n## 四、仅基质未退化的实体\n")
    for tag in [str(s) for s in shots]:
        w(f"\n### k={tag} · auc\n")
        w("| 协议 | " + " | ".join(f"`{a}`" for a in ARMS_SHOW) + " |")
        w("|---|" + "---|" * len(ARMS_SHOW))
        for proto in protos:
            row = []
            for a in ARMS_SHOW:
                d = agg(proto, tag, a, "auc", clean)
                row.append(f"{np.mean(list(d.values())):.3f}(n={len(d)})"
                           if d else "—")
            w(f"| {proto} | " + " | ".join(row) + " |")
        w(f"\n### k={tag} · tpr5\n")
        w("| 协议 | " + " | ".join(f"`{a}`" for a in ARMS_SHOW) + " |")
        w("|---|" + "---|" * len(ARMS_SHOW))
        for proto in protos:
            row = []
            for a in ARMS_SHOW:
                d = agg(proto, tag, a, "tpr5", clean)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w(f"| {proto} | " + " | ".join(row) + " |")

    with open(os.path.join(OUT, "exp74_p6_fairness_audit.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp74] 已写出 out/exp74_p6_fairness_audit.md")


if __name__ == "__main__":
    main()
