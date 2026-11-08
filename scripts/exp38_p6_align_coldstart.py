"""exp38 · 冷启动设定下的「对齐方式」消融：零样本期，对齐是唯一可变的量。

为什么必须换到冷启动设定
------------------------
exp34/exp37 已证明：闭集设定下本数据集**无法回答**先验问题 ——
13 个实体跨基质，训练 600 步后无先验基线已达 0.93+，Δ 被天花板压到 ±0.03，
连 CKA 分箱都测不出差异（exp37：Pearson r = −0.034, p = 0.82）。
⇒ 层1（闭集）不再追（用户 2026-11-02 拍板），重心转到**冷启动 + 双分离**。

冷启动设定下对齐的特殊地位
--------------------------
k = 0（无该掺杂物任何样本、也无纯品参考谱）时：
  - 可学习分类头**根本没有该类的列** ⇒ None / Init / Anchor / AlignCE 全部不可定义
  - 唯一可行的做法是**冻结一个化学码本，让谱编码器去对齐它**
  ⇒ **此时「怎么对齐」是唯一可变的自由度**，也正是本实验的核心。
k > 0 时才有可学习臂的对照（None / Init / Anchor / AlignCE）。

对齐策略（唯一变量 = 分类头 W 与化学嵌入 T 的关系）
--------------------------------------------------
  Rand        冻结随机高斯（零化学信息对照；k=0 时实测恒为 0）
  Raw         冻结 T（原始，未校正各向异性）
  Center      冻结 T − mean(T)（All-but-the-Top 首步）
  Whiten      冻结 阻尼 ZCA 白化（去均值 + 各方向等方差）
  ABTT2       冻结 去均值 − top-2 主成分（All-but-the-Top 完整版）
  Init        以 Center 初始化后微调（仅 k>0）
  Anchor(λ)   可学习头 + λ·‖Ŵ − T̂‖²（先验当正则，强度旋钮；仅 k>0）
  AlignCE(μ)  可学习头 + μ·CE(z, T̂)（把对齐写成显式对比损失；仅 k>0）

预注册判据
----------
  H38a（主）：k=0 时，校正类对齐（Center / Whiten / ABTT2 中最佳者）显著优于 Raw
  H38b（主）：k=0 时，存在化学对齐臂显著优于随机码本 Rand
             （exp35 已证 Rand≡0，此条用于确认化学信息的不可替代性并给出幅度）
  H38c（次）：k>0 时，软锚定 Anchor 的中间 λ 优于「全冻结」与「全自由」两个极端
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp08_p6_g1 as E8

OUT = r"D:/2026sci论文实验设计/论文6/out"
TEMP = 0.07
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 对齐变换
def center(T):
    T = np.asarray(T, dtype=np.float32)
    return (T - T.mean(0, keepdims=True)).astype(np.float32)


def whiten(T, eps=1e-2):
    Tc = np.asarray(center(T), dtype=np.float64)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    return (U * (S / np.sqrt(S ** 2 + eps)) @ Vt).astype(np.float32)


def abtt(T, D=2):
    Tc = np.asarray(center(T), dtype=np.float64)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    D = min(D, len(S))
    return (Tc - (U[:, :D] * S[:D]) @ Vt[:D]).astype(np.float32)


def align_diag(W):
    W = np.asarray(W, dtype=np.float64)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    off = (Wn @ Wn.T)[~np.eye(len(W), dtype=bool)]
    Wc = W - W.mean(0, keepdims=True)
    s = np.linalg.svd(Wc, compute_uv=False)
    er = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.90) + 1) if s.sum() > 0 else 0
    return {"cos_off": float(off.mean()), "eff_rank90": er}


# ---------------------------------------------------------------- 训练
def train_arm(Xtr, ytr, W0, kind, dim, nrows, steps, seed, lam=0.0, mu=0.0):
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    Tt = None if W0 is None else torch.tensor(np.asarray(W0, dtype=np.float32), device=DEV)

    if kind in ("learn", "init", "anchor", "alignce"):
        head = nn.Linear(dim, nrows).to(DEV)
        if kind != "learn":
            with torch.no_grad():
                head.weight.copy_(Tt[:nrows]); head.bias.zero_()
        W = head.weight
        params = list(net.parameters()) + list(head.parameters())
        Tsub = None if Tt is None else Tt[:nrows]
    else:  # frozen
        W = Tt[:nrows]
        params = list(net.parameters())
        head, Tsub = None, None

    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)
    net.train()
    done = 0
    while done < steps:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 256):
            if done >= steps:
                break
            idx = perm[i:i + 256]
            opt.zero_grad()
            z = F.normalize(net(Xt[idx]), dim=1)
            loss = F.cross_entropy(z @ F.normalize(W, dim=1).T / TEMP, yt[idx])
            if kind == "anchor" and lam > 0:
                loss = loss + lam * F.mse_loss(F.normalize(W, dim=1), F.normalize(Tsub, dim=1))
            elif kind == "alignce" and mu > 0:
                loss = loss + mu * F.cross_entropy(z @ F.normalize(Tsub, dim=1).T / TEMP, yt[idx])
            loss.backward(); opt.step()
            done += 1

    if kind == "frozen":
        return net, Tt                       # 评估用完整 19 行
    if nrows < 19:
        # 可学习臂在 k=0 时没有新类别的行；本实验 k=0 不跑这些臂，此处仅兜底
        pad = torch.zeros(19 - nrows, dim, device=DEV)
        W = torch.cat([W.detach(), pad], dim=0)
    return net, W


@torch.no_grad()
def top1(net, W, Xte, cand_rows, true_row):
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    logits = (z @ F.normalize(W, dim=1).T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    return float((pred == cand_rows.index(true_row)).mean())


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,20,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.01,0.1,1")
    ap.add_argument("--mus", default="0.5")
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()

    shots = []
    for s in args.shots.split(","):
        shots.append(None if s.strip() == "full" else int(s))
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]
    mus = [float(x) for x in args.mus.split(",")]

    print(f"[exp38] device = {DEV}  repr = {args.repr}", flush=True)
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys()); mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    key_index = {k: i for i, k in enumerate(keys)}

    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    assert [str(x) for x in zc["keys"]] == keys
    T_all = np.asarray(zc[args.repr], dtype=np.float32)
    dim = T_all.shape[1]

    def rand_W(n, d, seed):
        return np.random.default_rng(5000 + seed).normal(size=(n, d)).astype(np.float32)

    def build_W(arm, e_hold, seed):
        others = [e for e in eids if e != e_hold]
        rows = ([T_all[key_index[e]] for e in others]
                + [T_all[key_index[f"M_{m}"]] for m in mids]
                + [T_all[key_index[e_hold]]])
        W = np.stack(rows).astype(np.float32)
        if arm == "Rand":
            return rand_W(19, dim, seed)
        if arm == "Raw":
            return W
        if arm == "Center":
            return center(W)
        if arm == "Whiten":
            return whiten(W)
        if arm == "ABTT2":
            return abtt(W, 2)
        return center(W)          # Init / Anchor / AlignCE 的起点

    # (名字, kind, 是否需要该类的训练样本, lam, mu)
    arms = [("Rand", "frozen", False, 0.0, 0.0),
            ("Raw", "frozen", False, 0.0, 0.0),
            ("Center", "frozen", False, 0.0, 0.0),
            ("Whiten", "frozen", False, 0.0, 0.0),
            ("ABTT2", "frozen", False, 0.0, 0.0),
            ("None", "learn", True, 0.0, 0.0),
            ("Init", "init", True, 0.0, 0.0)]
    for l in lams:
        arms.append((f"Anchor{l:g}", "anchor", True, l, 0.0))
    for m in mus:
        arms.append((f"AlignCE{m:g}", "alignce", True, 0.0, m))
    arm_names = [a[0] for a in arms]

    res = {"acc": {t: {a: {} for a in arm_names} for t in tags}, "nte": {},
           "diag": {}, "meta": {"repr": args.repr, "shots": tags, "seeds": args.seeds,
                                "steps": args.steps, "eids": eids, "dim": dim,
                                "lams": lams, "mus": mus, "test_frac": 0.3}}

    for e_hold in eids:
        others = [e for e in eids if e != e_hold]
        cand_rows = list(range(len(others))) + [18]
        Xe = np.asarray(ent[e_hold], float)
        # 留一个代表性的 W 做对齐诊断（各臂几何不同，逐个记）
        for name, kind, need, lam, mu in arms:
            W0 = build_W(name, e_hold, 0)
            res["diag"].setdefault(name, {})[e_hold] = align_diag(W0)
        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix = perm[:n_te]
                pool_ix = perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                res["nte"].setdefault(e_hold, len(te_ix))

                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float))
                    ys.append(np.full(len(bg[m]), len(others) + j))
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), 18))
                    nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte = Xe[te_ix]

                for name, kind, need, lam, mu in arms:
                    if need and len(tr_e_ix) == 0:
                        continue          # k=0 时可学习臂不可定义
                    W0 = None if kind == "learn" else build_W(name, e_hold, seed)
                    net, W = train_arm(Xtr, ytr, W0, kind, dim, nrows, args.steps,
                                       seed * 1000 + eids.index(e_hold), lam=lam, mu=mu)
                    res["acc"][tag][name].setdefault(e_hold, []).append(
                        float(top1(net, W, Xte, cand_rows, 18)))
        line = []
        for t in tags:
            s = " ".join(f"{a}={np.mean([np.mean(v) for v in res['acc'][t][a].values()]):.3f}"
                         for a in arm_names if res["acc"][t][a])
            line.append(f"{t}: {s}")
        print(f"[{e_hold:>7}] " + " | ".join(line), flush=True)

    write_report(res, arm_names, tags, args)
    with open(os.path.join(OUT, f"exp38_res_{args.repr}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[exp38] 已写出 out/exp38_p6_align_coldstart_{args.repr}.md", flush=True)


# ---------------------------------------------------------------- 报告
def write_report(res, arm_names, tags, args):
    from scipy.stats import ttest_rel
    A = []
    w = lambda s: A.append(s)
    eids_all = res["meta"]["eids"]
    eids = [e for e in eids_all if any(e in res["acc"][t][a] for t in tags for a in arm_names
                                       if res["acc"][t][a])]

    def series(t, a):
        if not res["acc"][t][a]:
            return None, None
        M = np.array([res["acc"][t][a][e] for e in eids])       # (nE, seeds)
        wts = np.array([res["nte"].get(e, 1) for e in eids], float)
        micro = (M * wts[:, None]).sum(0) / wts.sum()
        return M.mean(0), micro

    w(f"# exp38 · 冷启动设定下的「对齐方式」消融（{args.repr.upper()}）")
    w("")
    w("> **k=0 时分类头只能冻结** ⇒ 「怎么对齐」是唯一可变的自由度。")
    w("> 闭集设定已弃用（exp37：任务天花板 0.93+，Δ 被压到 ±0.03，测不出任何调节效应）。")
    w("")
    w(f"- LOO 13 实体，k ∈ {{{', '.join(tags)}}}，seeds {args.seeds}，{args.steps} 步")
    w("- **固定 30% 测试集**（所有 k 档共用），micro = 按测试集大小加权，macro = 逐实体等权")
    w("")
    w("## 对齐质量诊断（分类头的静态几何，跨实体均值）")
    w("")
    w("| 臂 | 行间余弦均值 cos_off ↓ | 有效秩(90%能量) / 19 ↑ |")
    w("|---|---|---|")
    for a in arm_names:
        if a not in res["diag"]:
            continue
        co = np.mean([res["diag"][a][e]["cos_off"] for e in res["diag"][a]])
        er = np.mean([res["diag"][a][e]["eff_rank90"] for e in res["diag"][a]])
        w(f"| {a} | {co:+.3f} | {er:.1f} / 19 |")
    w("")
    for scope, agg in [("micro（按测试集大小加权）", lambda t, a: series(t, a)[1]),
                       ("macro（逐实体等权）", lambda t, a: series(t, a)[0])]:
        w(f"## 主结果 · {scope}")
        w("")
        w("| k | " + " | ".join(arm_names) + " |")
        w("|---|" + "---|" * len(arm_names))
        for t in tags:
            cells = []
            for a in arm_names:
                v = agg(t, a)
                cells.append("—" if v is None else f"{np.mean(v):.4f}")
            w(f"| {t} | " + " | ".join(cells) + " |")
        w("")

    w("## 预注册判据裁定")
    w("")
    if "0" in tags:
        t0 = "0"
        m_raw = series(t0, "Raw")[0]
        m_rand = series(t0, "Rand")[0]
        best, bname = None, ""
        for a in ["Center", "Whiten", "ABTT2"]:
            v = series(t0, a)[0]
            if v is None:
                continue
            if best is None or v.mean() > best.mean():
                best, bname = v, a
        if best is not None:
            d = best - m_raw
            ts, pv = ttest_rel(best, m_raw)
            w(f"- **H38a**（k=0，校正对齐优于 Raw）：{bname} − Raw = {d.mean():+.4f}，"
              f"配对 t p = {pv:.4f}（n={len(d)} 实体）⇒ {'✅ 成立' if (d.mean() > 0 and pv < 0.05) else '❌ 不成立'}")
            d2 = best - m_rand
            ts2, pv2 = ttest_rel(best, m_rand)
            w(f"- **H38b**（k=0，化学对齐优于随机码本）：{bname} − Rand = {d2.mean():+.4f}，"
              f"配对 t p = {pv2:.4f} ⇒ {'✅ 成立' if (d2.mean() > 0 and pv2 < 0.05) else '❌ 不成立'}")
            w("")
            w("逐实体（k=0）：")
            w("")
            w("| 实体 | Rand | Raw | Center | Whiten | ABTT2 |")
            w("|---|---|---|---|---|---|")
            for e in eids:
                w(f"| {e} | " + " | ".join(
                    f"{np.mean(res['acc'][t0][a][e]):.3f}"
                    for a in ["Rand", "Raw", "Center", "Whiten", "ABTT2"]) + " |")
            w("")
    anch = [a for a in arm_names if a.startswith("Anchor")]
    for t in tags:
        if t == "0" or len(anch) < 3:
            continue
        vals = []
        for a in anch:
            v = series(t, a)[0]
            if v is not None:
                vals.append((float(a.replace("Anchor", "")), v.mean()))
        if not vals:
            continue
        s = "，".join(f"λ={l:g}:{v:.4f}" for l, v in vals)
        best_l = max(vals, key=lambda x: x[1])[0]
        inner = min(l for l, _ in vals) < best_l < max(l for l, _ in vals)
        w(f"- **H38c**（k={t}，软锚定存在中间最优 λ）：{s} ⇒ 最优 λ={best_l:g} "
          f"（{'✅ 中间最优' if inner else '❌ 在端点'}）")
    w("")
    w("## 解读")
    w("")
    w("1. k=0 时 None/Init/Anchor 全部**不可定义**（分类头没有该类的列），")
    w("   化学结构是无样本时唯一能给出该类方向的信息源 —— 这是「必须加先验」的数学理由。")
    w("2. 但「有先验」不等于「有用」：**未校正的 Raw 因各向异性严重退化**，")
    w("   去均值 / 白化 / 去顶主成分这三种校正的差距，就是「怎么对齐」的价值幅度。")
    w("3. k>0 后先验的角色应从「冻结码本」转为「软正则」：")
    w("   exp35 已观察到冻结化学头在 k>0 时反而低于无先验基线。")
    w("")

    txt = "\n".join(A)
    with open(os.path.join(OUT, f"exp38_p6_align_coldstart_{args.repr}.md"),
              "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
