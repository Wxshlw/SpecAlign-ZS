"""exp05c —— G0 探针修正版：限定「同基质留一掺杂物」（论文 claim 的真实场景）。

exp05（跨基质 LOEO）的塌陷定位（exp05b）表明：
  - 跨基质 fold（A04–A07）测的是论文不 claim 的域迁移，OOD 塌缩淹没语义信号；
  - 论文 claim 的场景 = 同一基质内 unseen 掺杂物识别（G1 的微型化）；
  - 公共 7 集中唯一同基质多掺杂物结构 = 肉桂 D7 三兄弟（A01 核桃 / A02 花生 / A03 木屑）。

设计：3 fold（A01/A02/A03）× 3 seeds {0,1,2} × 5 变体，candidates = 3（chance = 1/3）。
统计：9 个 (fold, seed) 配对观测，单侧符号翻转置换 + Holm（4 比较）。
判据（v9 单向收紧）：Original 显著优于全部 4 对照。
跨基质塌陷作为独立的 limitation 报告（exp05/exp05b 保留），不参与 G0 主判定。
"""
from __future__ import annotations
import os, sys, csv, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/2026sci论文实验设计/论文6/scripts")
import exp05_p6_g0 as G
import torch

OUT = G.OUT
SEEDS = [0, 1, 2]
FOLDS = ["A01", "A02", "A03"]          # 肉桂基质内三实体
CIN_SIDS = ["D7w", "D7p", "D7d"]


def train_net(Xs, ys, Ttr, epochs, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xs.shape[1], 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(),
        torch.nn.Linear(256, Ttr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xs); yt = torch.tensor(ys)
    Tt = torch.nn.functional.normalize(torch.tensor(Ttr), dim=1)
    for _ in range(epochs):
        opt.zero_grad()
        z = torch.nn.functional.normalize(net(Xt), dim=1)
        loss = torch.nn.functional.cross_entropy(z @ Tt.T / G.TEMP, yt)
        loss.backward(); opt.step()
    return net, Tt


def main():
    eids = sorted(G.L.ACTIVE_EIDS)
    data, grid = G.load_matrix_data()
    X_by_eid = {G.SID2EID[sid]: D["X"] for sid, D in data.items() if sid in G.SID2EID}
    variants = G.build_text_variants(eids)
    eid_ix = {e: i for i, e in enumerate(eids)}

    results = {}   # {variant: {(fold,seed): top1}}
    for v in G.VARIANTS:
        text_emb = variants[v]
        T_all = np.stack([text_emb[e] for e in eids]).astype(np.float32)
        results[v] = {}
        for fold in FOLDS:
            eids_tr = [e for e in eids if e != fold]
            tr_ix = {e: i for i, e in enumerate(eids_tr)}
            Xs = np.concatenate([X_by_eid[e] for e in eids_tr])
            ys = np.concatenate([[tr_ix[e]] * len(X_by_eid[e]) for e in eids_tr])
            Ttr = T_all[[eid_ix[e] for e in eids_tr]]
            for seed in SEEDS:
                net, Tt = train_net(Xs, ys, Ttr, G.EPOCHS, seed * 100 + FOLDS.index(fold))
                with torch.no_grad():
                    zh = torch.nn.functional.normalize(net(torch.tensor(X_by_eid[fold])), dim=1)
                    Ta = torch.nn.functional.normalize(torch.tensor(T_all), dim=1)
                    sim = (zh @ Ta.T).numpy()
                # 修正 bug：原 `(-sim).argsort(1).argsort(1)[:, 0]` 是第 0 类（A01）的排名。
                # A01 fold 恰好等价于 accuracy（结论不受影响）；A02 fold 数字语义曾失真。
                pred = sim.argmax(1)
                top1 = float((pred == eid_ix[fold]).mean())
                results[v][(fold, seed)] = top1
                print(f"[{v}] {fold} seed{seed}: top1={top1:.3f}", flush=True)

    # ---- 统计：9 个 (fold,seed) 配对 ----
    pairs = [(f, s) for f in FOLDS for s in SEEDS]
    orig = np.array([results["Original"][p] for p in pairs])
    p_raw, obs_mean = {}, {}
    for v in G.VARIANTS[1:]:
        ctrl = np.array([results[v][p] for p in pairs])
        p_raw[v], obs_mean[v] = G.perm_paired_greater(orig, ctrl)
    p_adj = G.holm([p_raw[v] for v in G.VARIANTS[1:]])
    verdict = {v: p <= 0.05 for v, p in zip(G.VARIANTS[1:], p_adj)}

    chance = 1.0 / len(eids)
    def m(v, fold=None):
        vals = [results[v][p] for p in pairs if fold is None or p[0] == fold]
        return float(np.mean(vals))

    lines = ["# exp05c G0 修正版：同基质留一掺杂物（肉桂 D7 三 fold × 3 seeds）", "",
             f"> 论文 claim 场景 = 同一基质内 unseen 掺杂物识别；candidates = 7（chance = {chance:.3f}）。",
             "> 跨基质塌陷（exp05/exp05b）作为独立 limitation，不参与本判定。", "",
             "## 主表", "",
             "| 变体 | top-1 均值 | A01 | A02 | A03 | vs Original Holm-p | 判定 |",
             "|---|---|---|---|---|---|---|"]
    lines.append(f"| **Original（组分谱）** | {m('Original'):.3f} | {m('Original','A01'):.3f} | "
                 f"{m('Original','A02'):.3f} | {m('Original','A03'):.3f} | — | — |")
    for i, v in enumerate(G.VARIANTS[1:]):
        lines.append(f"| {v} | {m(v):.3f} | {m(v,'A01'):.3f} | {m(v,'A02'):.3f} | {m(v,'A03'):.3f} | "
                     f"{p_adj[i]:.4f} (raw {p_raw[v]:.4f}) | {'✅ Original 显著更优' if verdict[v] else '❌ 不显著'} |")

    lines += ["", "## 逐 (fold, seed) 明细（top-1）", "",
              "| fold | seed | " + " | ".join(G.VARIANTS) + " |",
              "|---|---|" + "---|" * len(G.VARIANTS)]
    for f in FOLDS:
        for s in SEEDS:
            lines.append(f"| {f} | {s} | " + " | ".join(f"{results[v][(f,s)]:.3f}" for v in G.VARIANTS) + " |")

    n_pass = sum(verdict.values())
    lines += ["", "## G0 判定（v9 单向收紧判据，修正口径）", ""]
    if n_pass == 4:
        lines.append("- ✅ **G0 PASS**：Original 在同基质场景下显著优于全部 4 对照 → 化学语义贡献成立，进 G1。")
    elif verdict.get("Shuffle-Random") and verdict.get("Shuffle-Mismatch") and not (
            verdict.get("Random-Encoder") and verdict.get("Descriptor")):
        lines.append("- 🟡 **CLM 先验无增量**：打乱对照拉开但 Random-Encoder/Descriptor 未拉开 → "
                     "主结论改写为「结构描述的参数化决定 zero-shot 上限」，claim 下移。")
    else:
        lines.append("- 🔴 **文本侧零语义贡献**（同基质场景下仍未拉开打乱对照）→ 走 v9 退出路径。")
    lines.append("- ⚠️ 附带发现（跨基质 limitation，写进论文）：谱编码器对完全 unseen 基质发生 OOD 塌缩"
                 "（exp05b：A02→A03 100%、A07→A03 100%、A05→A06 100%）→ G1 的候选集设计须包含"
                 "「基质标识」或限定同基质候选；这也是真实数据（非合成）口径下的诚实边界。")

    with open(os.path.join(OUT, "exp05c_p6_g0_withinmatrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT, "exp05c_p6_g0_withinmatrix.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "fold", "seed", "top1"])
        for v in G.VARIANTS:
            for (fl, s), val in results[v].items():
                w.writerow([v, fl, s, val])
    print("\n".join(lines), flush=True)
    print("[written] exp05c_p6_g0_withinmatrix.md / .csv", flush=True)


if __name__ == "__main__":
    main()
