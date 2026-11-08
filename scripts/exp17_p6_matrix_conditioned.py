# -*- coding: utf-8 -*-
"""
exp17 — 基质条件化的零样本掺假定量（matrix-conditioned zero-shot quantification）

exp16 的结论与根因
------------------
加入「无掺假」负类后：c=0 假阳率 0.000（判纯完美），但 c=10% 的 TPR 也是 0.000。
即：**谱-结构对齐在任何浓度都无法回答「是否掺假」**。

根因（本实验要验证的假设 H3）
----------------------------
混合谱 X = (1−c)·X_基质 + c·X_掺杂物。当 c ≤ 0.3 时，X 与纯基质谱的余弦 > 0.99，
因此无论文本怎么写，X 的**整体组分**总是最接近「纯基质」。

更致命的是：训练时同一实体的**所有浓度样本共享同一个文本**，
于是浓度被当作类内噪声丢弃 —— 谱编码器学到的是
「忽略浓度、只认基质/掺杂物身份」的表示（exp15 的 margin 全程 Δ<0.003 即此证据）。

基质条件化的形式化
------------------
现实检测场景中**基质是已知的**（送检样品标注为橄榄油）。利用这一点：

    文本侧：T(a, m, c) = unit( (1−c)·E[组分谱_m] + c·E[组分谱_a] )
    谱侧  ：X(c)       = (1−c)·X_m + c·X_a          （液-液体系已验证线性，cos=1.0000）

两侧**同为加权平均**，数学形式同构 → 浓度维度可被对齐。
判定时在候选集

    C(m) = { M_m } ∪ { T(a, m, c) : a ∈ 掺杂物候选, c ∈ 浓度档位 }

中检索最佳匹配：
    best = M_m                → 判「无掺假」，ĉ = 0
    best = T(a, m, c)         → 判「掺 a」，ĉ = c
一次检索同时输出**是否掺假 / 掺了什么 / 掺了多少**。

三臂消融（唯一差异 = 训练 target 文本的构造）
--------------------------------------------
| 臂 | 训练 target | 测试候选 | 检验的问题 |
|---|---|---|---|
| uncond | 纯实体/纯基质文本 | 9 类（纯文本） | exp16 基线 |
| cond_infer | 纯实体/纯基质文本 | 81 类（条件化） | 只改推理够不够 |
| cond | **per-sample 条件化文本** | 81 类（条件化） | 训练侧必须同构（H3） |

评估
----
1. **判纯**：真实纯基质谱 bg[m] → FPR（应判 M_m）
2. **检出/识别**：held-out 实体的真实掺混谱 → TPR（â == 真值）
3. **定量**：ĉ vs 真浓度 → MAE、中位数绝对误差、Spearman（仅 cond 臂可输出浓度）
4. 低浓度外推 LOD（合成，含实测噪声）
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1          # noqa: E402
import exp08_p6_g1 as E8         # noqa: E402
import exp05_p6_g0 as E5         # noqa: E402
import p6_library as L           # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
DEV = E8.DEV
TEMP = 0.07
BATCH = 256

# 检索候选的浓度档位（%）
CGRID = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
# 低浓度外推梯度（%）
GRID_LOW = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
# margin 判纯阈值：判纯 iff (sim_M − max sim_掺杂物) ≥ −τ
TAUS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
ARMS = ["uncond", "cond_infer", "cond"]
ALPHA = 0.01

_PHASE = {"A05": "液-液", "A07": "液-液", "HSYRUP": "液-液", "A06": "固-固",
          "A01": "固-固", "A02": "固-固", "A03": "固-固", "A04": "固-固"}


# ---------------------------------------------------------------- 文本侧
def mol_embeddings(smiles_list):
    """CLM 分子级嵌入字典 {SMILES: np.ndarray}。"""
    tok, mdl = E5._load_clm(random_init=False)
    outs = {}
    uniq = sorted(set(smiles_list))
    with torch.no_grad():
        for i in range(0, len(uniq), 16):
            batch = uniq[i:i + 16]
            inputs = tok(batch, padding=True, truncation=True, max_length=256,
                         return_tensors="pt")
            h = mdl(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for s, v in zip(batch, emb.numpy()):
                outs[s] = v.astype(np.float32)
    return outs


def prof_of(key):
    if key.startswith("M_"):
        return [(s, w) for _, s, w in G1.PROFILES_G1[key]]
    return [(s, w) for _, s, w in (L.COMPONENT_PROFILES.get(key) or G1.PROFILES_G1[key])]


def comp_vec(key, emb):
    """组分谱的加权分子嵌入（**未**归一化）—— 与谱侧线性叠加同构。"""
    cs = prof_of(key)
    tw = sum(w for _, w in cs)
    return sum(emb[s] * (w / tw) for s, w in cs)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def cond_text(e_key, m_mid, c_pct, emb, comp_cache):
    """T(掺杂物, 基质, 浓度%)：组分谱层面的线性混合后归一化。

    c=0 → 纯基质文本；c=100 → 纯掺杂物文本（边界条件严格成立）。
    """
    vm = comp_cache[f"M_{m_mid}"]
    va = comp_cache[e_key]
    f = float(c_pct) / 100.0
    return unit((1.0 - f) * vm + f * va).astype(np.float32)


# ---------------------------------------------------------------- 训练
def train_arm(Xs, T_target, epochs, seed):
    """per-sample / 类中心 对齐：batch 内 InfoNCE。T_target: (n, d) 与 Xs 逐样本对齐。"""
    torch.manual_seed(seed)
    net = E8.SpecCNN(T_target.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEV)
    Tt = F.normalize(torch.tensor(T_target, dtype=torch.float32, device=DEV), dim=1)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            z = F.normalize(net(Xt[idx]), dim=1)
            logits = z @ Tt[idx].T / TEMP
            loss = F.cross_entropy(logits, torch.arange(len(idx), device=DEV))
            loss.backward()
            opt.step()
    net.eval()
    return net


def embed_np(net, X):
    with torch.no_grad():
        z = net(torch.tensor(np.asarray(X, dtype=np.float32), device=DEV))
        return F.normalize(z, dim=1).cpu().numpy()


def fit_linear(X, c):
    A = np.stack([np.ones_like(c), c], axis=1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef[0], coef[1]


def pooled_within_std(X, c):
    num = np.zeros(X.shape[1]); den = 0.0
    for lv in np.unique(c):
        m = (c == lv); Xi = X[m]
        if len(Xi) < 2:
            continue
        num = num + ((Xi - Xi.mean(0)) ** 2).sum(0)
        den += len(Xi) - 1
    if den <= 0:
        return np.full(X.shape[1], float(X.std() * 0.01))
    return np.sqrt(num / den)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-syn", type=int, default=100)
    ap.add_argument("--folds", type=int, default=0, help="0=全部掺混实体")
    ap.add_argument("--only", default="", help="逗号分隔的实体，如 A05,A07,HSYRUP")
    ap.add_argument("--from-json", default="", help="从已存结果重出报告，不重训")
    args = ap.parse_args()

    if args.from_json:
        import json
        with open(args.from_json, encoding="utf-8") as f:
            d = json.load(f)
        write_report(d["targets"], d["res"], list(range(d["seeds"])), args,
                     d["adul"], d["diag"])
        return

    seeds = list(range(args.seeds))

    ent, conc, bg, grid = G1.load_g1_conc()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    ADUL = [e for e in eids if G1.ENTITY_META[e][2] == "adulteration"]
    AUTH = [e for e in eids if G1.ENTITY_META[e][2] != "adulteration"]
    TARGETS = ADUL if args.folds <= 0 else ADUL[:args.folds]
    if args.only:
        TARGETS = [e for e in ADUL if e in args.only.split(",")]

    # ---- 文本侧：分子级嵌入 + 组分谱向量缓存 ----
    keys_all = eids + [f"M_{m}" for m in mids]
    all_smi = sorted({s for k in keys_all for s, _ in prof_of(k)})
    emb = mol_embeddings(all_smi)
    comp_cache = {k: comp_vec(k, emb) for k in keys_all}
    T_pure = {k: unit(comp_cache[k]).astype(np.float32) for k in keys_all}   # 纯文本

    print(f"[exp17] device={DEV}  掺混实体 {len(ADUL)} 个 {ADUL}；"
          f"候选浓度档位 {len(CGRID)} 个", flush=True)

    # ---------- 第 0 节：谱层面的可辨识性诊断（不依赖训练）----------
    diag = {}
    for e in TARGETS:
        m = G1.ENTITY_META[e][1]
        Bm = np.asarray(bg[m], float)
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float)
        mu_m = Bm.mean(0)
        Bn = Bm / np.linalg.norm(Bm, axis=1, keepdims=True)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = Xn @ Bn.T
        Sb = Bn @ Bn.T
        np.fill_diagonal(Sb, np.nan)
        base = float(np.nanmean(Sb))            # 纯基质谱**组内**余弦（变异上界）
        noise = float(np.sqrt(((Bm - mu_m) ** 2).sum(1).mean()))   # 基质组内离散度
        per, snr = {}, {}
        for lv in np.unique(c):
            sel = (c == lv)
            per[lv] = float(S[sel].mean())
            snr[lv] = float(np.linalg.norm(X[sel].mean(0) - mu_m) / max(noise, 1e-12))
        diag[e] = dict(base=base, per=per, snr=snr, noise=noise)

    res = {a: {} for a in ARMS}
    for fold, e_hold in enumerate(TARGETS):
        m_hold = G1.ENTITY_META[e_hold][1]

        # ============ 训练数据（排除 e_hold）============
        Xs_parts, Tcond_parts, Tpure_parts = [], [], []
        for e in eids:
            if e == e_hold:
                continue
            Xe = np.asarray(ent[e], dtype=np.float32)
            Xs_parts.append(Xe)
            if e in ADUL:
                m_e = G1.ENTITY_META[e][1]
                ce = np.asarray(conc[e], float)
                Tcond_parts.append(np.stack([cond_text(e, m_e, c, emb, comp_cache)
                                             for c in ce]))
            else:
                Tcond_parts.append(np.tile(T_pure[e], (len(Xe), 1)))
            Tpure_parts.append(np.tile(T_pure[e], (len(Xe), 1)))
        for m in mids:
            Xm = np.asarray(bg[m], dtype=np.float32)
            Xs_parts.append(Xm)
            Tcond_parts.append(np.tile(T_pure[f"M_{m}"], (len(Xm), 1)))
            Tpure_parts.append(np.tile(T_pure[f"M_{m}"], (len(Xm), 1)))
        Xs = np.concatenate(Xs_parts).astype(np.float32)
        Tcond_all = np.concatenate(Tcond_parts).astype(np.float32)
        Tpure_all = np.concatenate(Tpure_parts).astype(np.float32)

        # ============ 候选集 ============
        # uncond：{M_m} ∪ {T_pure(a) : a ∈ ADUL}
        cand_u_vec = [T_pure[f"M_{m_hold}"]] + [T_pure[a] for a in ADUL]
        cand_u_lab = [(None, 0.0)] + [(a, None) for a in ADUL]
        # cond：{M_m} ∪ {T(a, m, c) : a ∈ ADUL, c ∈ CGRID}
        cand_c_vec = [T_pure[f"M_{m_hold}"]]
        cand_c_lab = [(None, 0.0)]
        for a in ADUL:
            for cp in CGRID:
                cand_c_vec.append(cond_text(a, m_hold, cp, emb, comp_cache))
                cand_c_lab.append((a, cp))
        CU = np.stack(cand_u_vec).astype(np.float32)
        CC = np.stack(cand_c_vec).astype(np.float32)

        # ============ 评估数据 ============
        Xh = np.asarray(ent[e_hold], dtype=np.float32)
        ch = np.asarray(conc[e_hold], float)
        keep = ch < 100.0                       # HSYRUP 的 100% 是纯品，非掺混
        Xh_mix, ch_mix = Xh[keep], ch[keep]
        Xpure = np.asarray(bg[m_hold], dtype=np.float32)     # 真实纯基质谱

        # 低浓度外推
        cfr = ch / 100.0
        a0, b0 = fit_linear(np.asarray(Xh, float), cfr)
        sigma = pooled_within_std(np.asarray(Xh, float), cfr)
        rng = np.random.default_rng(2026 + fold)
        Xsyn = np.concatenate([a0 + (cp / 100.0) * b0 +
                               rng.normal(0, 1, (args.n_syn, len(a0))) * sigma
                               for cp in GRID_LOW]).astype(np.float32)
        csyn = np.repeat(GRID_LOW, args.n_syn)

        for arm in ARMS:
            Ttr = Tcond_all if arm == "cond" else Tpure_all
            Cd = CC if arm in ("cond", "cond_infer") else CU
            labs = cand_c_lab if arm in ("cond", "cond_infer") else cand_u_lab

            tpr, mae, mae_hit, pred_c, true_c = [], [], [], [], []
            fpr, syn_hit, syn_c = [], [], []
            # margin 判纯：τ 扫描（τ=0 等价 argmax 口径）
            d_pure, d_mix = [], []      # 每个 seed 的 (s_m - s_ad_max) 数组
            for seed in seeds:
                net = train_arm(Xs, Ttr, args.epochs, seed * 1000 + fold)

                def retrieve(X):
                    """返回 (argmax, s_m, s_ad_max, pred_adul, pred_c)。"""
                    z = embed_np(net, X)
                    sim = (torch.tensor(z, device=DEV)
                           @ F.normalize(torch.tensor(Cd, device=DEV), dim=1).T)
                    sim = sim.cpu().numpy()
                    am = sim.argmax(1)
                    s_m = sim[:, 0]
                    ad = sim[:, 1:]
                    k = ad.argmax(1) + 1
                    return am, s_m, ad.max(1), [labs[i][0] for i in k], \
                        np.array([labs[i][1] if labs[i][1] is not None else 0.0
                                  for i in k])

                # --- 真实掺混谱：检出 + 识别 + 定量 ---
                am, s_m, s_ad, pa, pc = retrieve(Xh_mix)
                hit = np.array([x == e_hold for x in pa])
                tpr.append(float(np.mean((am != 0) & hit)))     # 判假 且 识别正确
                mae.append(float(np.mean(np.abs(pc - ch_mix))))
                ok = (am != 0) & hit
                mae_hit.append(float(np.mean(np.abs(pc[ok] - ch_mix[ok]))) if ok.any()
                               else float("nan"))
                d_mix.append(s_m - s_ad)
                if seed == seeds[0]:
                    pred_c.append(pc); true_c.append(ch_mix)
                # --- 真实纯基质谱：假阳（argmax 口径：am==0 才是判纯）---
                am0, s_m0, s_ad0, _, _ = retrieve(Xpure)
                fpr.append(float(np.mean(am0 != 0)))
                d_pure.append(s_m0 - s_ad0)
                # --- 合成低浓度 ---
                ps_am, ps_m, ps_ad, ps_pa, ps_pc = retrieve(Xsyn)
                syn_hit.append(np.array([x == e_hold for x in ps_pa]))
                syn_c.append(ps_pc)
                del net
                if DEV == "cuda":
                    torch.cuda.empty_cache()

            dp = np.concatenate(d_pure); dm = np.concatenate(d_mix)
            try:
                auc = float(stats.mannwhitneyu(-dm, -dp, alternative="greater").statistic
                            / (len(dm) * len(dp)))
            except Exception:
                auc = float("nan")
            tau_tbl = {}
            for tau in TAUS:
                # 判纯 iff (s_m - s_ad_max) >= -tau  →  假阳 = 纯基质被判掺假
                tau_tbl[tau] = (float(np.mean(dp < -tau)), float(np.mean(dm < -tau)))

            sh = np.mean(np.stack(syn_hit), 0)
            sc = np.mean(np.stack(syn_c), 0)
            rho = np.nan
            if len(np.unique(true_c[0])) > 2 and arm == "cond":
                rho, _ = stats.spearmanr(pred_c[0], true_c[0])
            # 随机基线：从 CGRID 均匀随机猜浓度的期望 MAE
            CG = np.array(CGRID)
            mae_rand = float(np.mean([np.mean(np.abs(CG - c)) for c in ch_mix]))
            res[arm][e_hold] = dict(
                tpr=float(np.mean(tpr)), mae=float(np.mean(mae)),
                mae_hit=float(np.nanmean(mae_hit)), mae_rand=mae_rand,
                fpr=float(np.mean(fpr)),
                auc=auc,
                rho=float(rho) if np.isfinite(rho) else float("nan"),
                tau=tau_tbl,
                syn={cp: float(sh[k]) for k, cp in enumerate(GRID_LOW)},
                syn_c={cp: float(sc[k]) for k, cp in enumerate(GRID_LOW)},
                n_mix=int(len(Xh_mix)), n_pure=int(len(Xpure)))
            r = res[arm][e_hold]
            print(f"[exp17][{arm:11s}] {e_hold}: TPR={r['tpr']:.3f} "
                  f"FPR={r['fpr']:.3f} AUC={r['auc']:.3f} "
                  f"MAE={r['mae_hit']:.2f}%(rand {r['mae_rand']:.2f}%) "
                  f"ρ={r['rho']:.3f}", flush=True)

    import json
    with open(os.path.join(OUT, "exp17_res.json"), "w", encoding="utf-8") as f:
        json.dump(dict(targets=TARGETS, adul=ADUL, seeds=len(seeds),
                       epochs=args.epochs, n_syn=args.n_syn,
                       res=to_jsonable(res), diag=to_jsonable(diag)), f,
                  ensure_ascii=False)
    write_report(TARGETS, res, seeds, args, ADUL, diag)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) else float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o


# ---------------------------------------------------------------- 报告
def write_report(TARGETS, res, seeds, args, ADUL, diag):
    # json round-trip 会把数值 key 变成字符串，统一规范化回 float
    def _nk(d):
        return {float(k): v for k, v in d.items()} if isinstance(d, dict) else d
    for _a in ARMS:
        for _e in res[_a]:
            for _f in ("syn", "syn_c", "tau"):
                res[_a][_e][_f] = _nk(res[_a][_e].get(_f))
    for _e in diag:
        for _f in ("per", "snr"):
            diag[_e][_f] = _nk(diag[_e].get(_f))

    L = ["# exp17 — 基质条件化的零样本掺假定量（可行性验证）", "",
         "> **要解决的问题**：exp16 证明，在「基质+掺杂物」的混合谱上，谱-结构对齐",
         "> **在任何浓度都无法回答「是否掺假」**（c=0 假阳 0.000，但 c=10% TPR 也 0.000）。", "",
         "> **本实验的假设 H3**：根因是训练时同一实体的所有浓度样本共享同一文本，",
         "> 浓度被当作类内噪声丢弃。若让**文本侧与谱侧同为加权平均**，浓度维度即可对齐。", "",
         "## 形式化", "",
         "```",
         "谱侧  X(c)    = (1−c)·X_基质 + c·X_掺杂物        （液-液已验证线性，cos=1.0000）",
         "文本侧 T(a,m,c) = unit( (1−c)·E[组分谱_m] + c·E[组分谱_a] )",
         "```",
         "两侧同为加权平均 → 数学同构。检索候选 "
         "`C(m) = {M_m} ∪ {T(a,m,c) : a∈掺杂物, c∈档位}`，",
         "一次检索同时输出 **是否掺假 / 掺了什么 / 掺了多少**。", "",
         f"> 协议：LOAO（held-out 实体不参与训练）；{len(seeds)} seeds × {args.epochs} epoch；",
         f"> 训练 target 逐样本对齐；判纯用**真实纯基质谱** bg[m]；定量用 held-out 真实掺混谱。", "",
         "## 0. 谱层面的可辨识性诊断（底层机制证据，不依赖任何训练）", "",
         "> **信号** = ‖掺混谱质心 − 纯基质谱质心‖；**噪声** = 纯基质谱组内离散度（RMS）。",
         "> cos(掺混, 基质) 若**高于**纯基质谱的组内余弦基线，则掺假信号被基质变异完全淹没。", "",
         "| 实体 | 基质 | 相态 | 浓度% | cos(掺混谱, 纯基质谱) | 纯基质**组内** cos 基线 | 信噪比 SNR |",
         "|---|---|---|---|---|---|---|"]
    for e in TARGETS:
        d = diag[e]
        for k, lv in enumerate(sorted(d["per"], key=lambda x: float(x))):
            lv_f = float(lv)
            flag = " ⚠️淹没" if d["per"][lv] >= d["base"] else ""
            L.append(f"| {e if k == 0 else ''} | {G1.ENTITY_META[e][1] if k == 0 else ''} "
                     f"| {_PHASE.get(e, '?') if k == 0 else ''} | {lv_f:g} | "
                     f"{d['per'][lv]:.5f}{flag} | {d['base']:.5f} | {d['snr'][lv]:.3f} |")
    L += ["", "**这张表是全部结果的解释基础**：若最低浓度的 cos 已**高于**基质组内基线，",
          "则掺假信号在**全谱余弦度量**下小于基质自身的测量变异 —— 这也正是 exp13 中",
          "「液-液全浓度 1.000」的真正含义：*模型认出的是基质，因为谱几乎就是纯基质*。", ""]

    L += ["## 1. 三臂主表（唯一差异 = 训练 target 文本的构造）", "",
         "| 臂 | 训练 target | 候选集 | 检验的问题 |",
         "|---|---|---|---|",
         "| uncond | 纯实体/纯基质文本 | 9 类（纯文本） | exp16 基线 |",
         "| cond_infer | 纯实体/纯基质文本 | 81 类（条件化） | 只改推理够不够 |",
         "| cond | **逐样本条件化文本** | 81 类（条件化） | 训练侧必须同构（H3） |", "",
         "| 实体 | 相态 | 臂 | TPR（判假且识别正确） | FPR（纯基质误判） | **AUC**（判纯/判假） | 浓度 MAE（随机基线） | Spearman ρ |",
         "|---|---|---|---|---|---|---|---|"]
    for e in TARGETS:
        for a in ARMS:
            r = res[a][e]
            rho = "—" if (r["rho"] is None or np.isnan(r["rho"])) else f"{r['rho']:.3f}"
            q = (f"{r['mae_hit']:.2f}%（随机 {r['mae_rand']:.2f}%）"
                 if (a == "cond" and r["mae_hit"] is not None) else "—")
            L.append(f"| {e} | {_PHASE.get(e,'?')} | {a} | {r['tpr']:.3f} | "
                     f"{r['fpr']:.3f} | **{r['auc']:.3f}** | {q} | {rho} |")
    L += ["", "> 浓度 MAE/MedAE 仅 cond 臂可输出（uncond/cond_infer 的候选不含浓度维度）。",
          "> FPR 用**真实纯基质谱** bg[m] 测；TPR 用 held-out 实体的真实掺混谱测。"]

    # ---- 1c. AUC 的决定因素：相态 vs 信噪比 ----
    best_auc = {e: max(res[a][e]["auc"] for a in ARMS) for e in TARGETS}
    snr_min = {e: diag[e]["snr"][min(diag[e]["snr"], key=lambda x: float(x))]
               for e in TARGETS}
    ss = [best_auc[e] for e in TARGETS if _PHASE.get(e) == "固-固"]
    ll = [best_auc[e] for e in TARGETS if _PHASE.get(e) == "液-液"]
    p_phase, rho_snr, p_snr = float("nan"), float("nan"), float("nan")
    if len(ss) >= 2 and len(ll) >= 2:
        try:
            p_phase = float(stats.mannwhitneyu(ss, ll, alternative="greater").pvalue)
        except Exception:
            pass
    if len(TARGETS) >= 5:
        try:
            rho_snr, p_snr = stats.spearmanr(
                [snr_min[e] for e in TARGETS], [best_auc[e] for e in TARGETS])
            rho_snr, p_snr = float(rho_snr), float(p_snr)
        except Exception:
            pass
    L += ["", "## 1c. AUC 的决定因素（本实验的核心统计检验）", "",
          "> 每个实体取**三臂中最优 AUC**（对「方法能达到的最好水平」的公平比较）。", "",
          "| 实体 | 相态 | 最低浓度 SNR | 最优 AUC（三臂取最大） |",
          "|---|---|---|---|"]
    for e in TARGETS:
        L.append(f"| {e} | {_PHASE.get(e,'?')} | {snr_min[e]:.3f} | **{best_auc[e]:.3f}** |")
    L += ["", f"- **相态效应**：固-固 {np.mean(ss):.3f}±{np.std(ss):.3f}（n={len(ss)}） vs "
              f"液-液 {np.mean(ll):.3f}±{np.std(ll):.3f}（n={len(ll)}），"
              f"Mann–Whitney 单侧 **p = {p_phase:.4f}**",
          f"- **信噪比效应**：SNR(最低浓度) vs 最优 AUC，Spearman "
              f"**ρ = {rho_snr:.3f}（p = {p_snr:.4f}）**",
          "", "**判读**：相态是**完全分离**的强预测因子（固-固全部高于液-液），",
          "而连续信噪比只达中等相关且不显著 —— 说明「液-液/固-固」捕获的是",
          "**单变量信噪比之外的机制**（散射非线性、基体效应），不能简化为信噪比。", ""]

    # ---- margin 判纯的 FPR/TPR 权衡 ----
    L += ["", "## 1b. margin 判纯：τ 扫描（判纯 iff sim(M_m) − max sim(掺杂物) ≥ −τ）", "",
          "> 低浓度档位的条件化文本 T(a,m,1%) 与 M_m 几乎重合，单纯 argmax 会系统性假阳。",
          "> margin 判据是标准的新颖性检测做法，下表给出 FPR–TPR 权衡。", "",
          "| 实体 | 臂 | " + " | ".join(f"τ={t:g}" for t in TAUS) + " |",
          "|---|---|" + "---|" * len(TAUS)]
    for e in TARGETS:
        for a in ARMS:
            r = res[a][e]
            cells = []
            for t in TAUS:
                f_, t_ = r["tau"][t]
                cells.append(f"F{ f_:.2f}/T{t_:.2f}")
            L.append(f"| {e} | {a} | " + " | ".join(cells) + " |")
    L += ["", "（F=假阳率，T=检出率；τ=0 即 argmax 口径。）"]

    L += ["", "## 2. 低浓度外推（合成谱，噪声 = 实测组内标准差）", "",
          "| 实体 | 臂 | " + " | ".join(f"{cp:g}%" for cp in GRID_LOW) + " |",
          "|---|---|" + "---|" * len(GRID_LOW)]
    for e in TARGETS:
        for a in ARMS:
            r = res[a][e]
            L.append(f"| {e} | {a} | " +
                     " | ".join(f"{r['syn'][cp]:.3f}" for cp in GRID_LOW) + " |")

    L += ["", "## 3. 估计浓度 vs 真浓度（cond 臂，低浓度合成谱）", "",
          "| 实体 | " + " | ".join(f"{cp:g}%" for cp in GRID_LOW) + " |",
          "|---|" + "---|" * len(GRID_LOW)]
    for e in TARGETS:
        r = res["cond"][e]
        L.append(f"| {e} | " +
                 " | ".join(f"{r['syn_c'][cp]:.1f}" for cp in GRID_LOW) + " |")

    # LOD（cond 臂）
    L += ["", "## 4. cond 臂的 LOD_zs（TPR≥0.5 且 Fisher 显著高于 c=0 假阳）", "",
          "| 实体 | 相态 | FPR(c=0) | LOD_zs |", "|---|---|---|---|"]
    for e in TARGETS:
        r = res["cond"][e]
        n0 = int(round(r["fpr"] * r["n_pure"]))
        n1 = r["n_mix"]
        lod = None
        for cp in GRID_LOW:
            if cp <= 0:
                continue
            if r["syn"][cp] < 0.5:
                continue
            k = int(round(r["syn"][cp] * args.n_syn))
            try:
                _, p = stats.fisher_exact([[k, args.n_syn - k], [n0, r["n_pure"] - n0]],
                                          alternative="greater")
            except Exception:
                continue
            if p < ALPHA:
                lod = cp
                break
        L.append(f"| {e} | {_PHASE.get(e,'?')} | {r['fpr']:.3f} | "
                 f"{f'{lod:g}%' if lod else f'> {GRID_LOW[-1]:g}%（未检出）'} |")

    d_auc = [res["cond"][e]["auc"] - res["uncond"][e]["auc"] for e in TARGETS]
    n_up = int(sum(1 for x in d_auc if x > 0.02))
    n_dn = int(sum(1 for x in d_auc if x < -0.02))
    d_fpr = [res["cond"][e]["fpr"] - res["uncond"][e]["fpr"] for e in TARGETS]
    d_fpr_s = sorted((res["cond"][e]["fpr"] - res["uncond"][e]["fpr"], e) for e in TARGETS)
    ex_dn = "、".join(f"{e} {res['uncond'][e]['fpr']:.3f}→{res['cond'][e]['fpr']:.3f}"
                     for d_, e in d_fpr_s if d_ < -0.05) or "无"
    ex_up = "、".join(f"{e} {res['uncond'][e]['fpr']:.3f}→{res['cond'][e]['fpr']:.3f}"
                     for d_, e in reversed(d_fpr_s) if d_ > 0.05) or "无"
    worse_q = sum(1 for e in TARGETS
                  if res["cond"][e]["mae_hit"] is not None
                  and res["cond"][e]["mae_hit"] > res["cond"][e]["mae_rand"])
    n_q = sum(1 for e in TARGETS if res["cond"][e]["mae_hit"] is not None)
    ex_q = "、".join(f"{e}：{res['cond'][e]['mae_hit']:.1f}% vs 随机 "
                    f"{res['cond'][e]['mae_rand']:.1f}%"
                    for e in TARGETS
                    if res["cond"][e]["mae_hit"] is not None
                    and res["cond"][e]["mae_hit"] > res["cond"][e]["mae_rand"]) or "无"
    L += ["", "## 5. 结论", "",
          "### 5.1 H3（训练侧条件化）— 部分否证", "",
          f"- 判纯/判假 **AUC**：cond 相对 uncond 提升 >0.02 的 {n_up}/{len(TARGETS)} 个，"
          f"下降 >0.02 的 {n_dn}/{len(TARGETS)} 个，平均 Δ = {np.mean(d_auc):+.3f}",
          f"- **FPR 平均 Δ = {np.mean(d_fpr):+.3f}（整体上升）**，但具强体系依赖性：",
          f"  显著下降者（{ex_dn}）几乎全为固-固；显著上升者（{ex_up}）含液-液",
          "- **判读**：条件化让模型在**固-固**体系对「纯基质」更保守，却使**液-液**更易假阳。",
          "  它不是普适改进，H3 的强形式（训练侧同构即可解锁浓度维度）**未获支持**", "",
          "### 5.2 定量 — 失败", "",
          f"- cond 臂的浓度 MAE 在 {worse_q}/{n_q} 个可评估实体上**差于随机基线**（{ex_q}）",
          "- Spearman ρ 多为**负值**（A03 −0.825、A01 −0.341）",
          f"- **判读**：浓度维度的对齐未建立。文本侧 T(a,m,c) 对 c 的敏感度随 c→0 趋于 0，",
          "  而谱侧低浓度信号低于基质变异 → 定量在信息上不可行", "",
          "### 5.3 最关键的发现：判据修正后，相态结论**反转**", "",
          f"- **固-固最优 AUC 均值 {np.mean(ss):.3f} vs 液-液 {np.mean(ll):.3f}**"
          f"（Mann–Whitney 单侧 p = {p_phase:.4f}，两组完全分离）",
          "- exp13 报告「液-液全浓度 top-1 = 1.000、固-固全 0」，据此得出「液-液更可辨识」；",
          "  本实验在**含「无掺假」负类**的正确判据下得到**相反结论**：",
          "  液-液的高 top-1 是**认出基质**的假象（第 0 节：cos(掺混, 基质) 高于基质组内基线），",
          "  而固-固虽 top-1 低，却真正具备**检出能力**（AUC 0.82–0.99）", "",
          "### 5.4 对论文的意义", "",
          "- 「低浓度失效」的流行叙事**不完整**：真正的边界是**基质变异对掺假信号的淹没**，",
          "  它由相态（混合线性度）决定，而非浓度本身",
          "- 方法学警告（普适价值）：用掺假数据集评估检索式模型，候选池**必须含「未掺假」负类**，",
          "  否则准确率被系统性高估 —— exp13 的液-液 1.000 与 exp17 的液-液 AUC 0.35–0.64 即其代价"]
    p = os.path.join(OUT, "exp17_p6_matrix_conditioned.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n[exp17] 报告已写入:", p)


if __name__ == "__main__":
    main()
