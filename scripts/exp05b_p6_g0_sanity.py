"""exp05b —— G0 探针 sanity 诊断（在 🔴 判定前排除探针自身缺陷）。

检查三件事：
  S1. 优化可行性：LOEO 训练时 seen 6 类分类准确率是否饱和（应 ≈1.0）。
  S2. 记忆上界：全部 7 实体训练后检索**训练谱自身**的 top-1（机制可达性检验，
      应 ≈1.0；若显著 <1 → 探针架构/优化有缺陷 → G0 阴性结果不可信）。
  S3. 塌陷定位：held-out fold 检索落到哪个候选（若全 fold 固定落到同一错误候选，
      说明谱编码器跨基质输出塌缩，而非文本侧无语义）。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/2026sci论文实验设计/论文6/scripts")
import exp05_p6_g0 as G

OUT = r"D:/2026sci论文实验设计/论文6/out"


def train_net(Xs, ys, Ttr, epochs, seed):
    import torch
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

    lines = ["# exp05b G0 sanity 诊断", ""]

    for vname in ["Original", "Descriptor"]:
        text_emb = variants[vname]
        T_all = np.stack([text_emb[e] for e in eids]).astype(np.float32)
        lines += [f"## 变体 {vname}", "",
                  "| fold | seen训练acc | held-out top1 | held-out 检索候选(top2) |", "|---|---|---|---|"]

        # ---- S2 记忆上界：全 7 实体训练，检索自身 ----
        Xs_all = np.concatenate([X_by_eid[e] for e in eids])
        ys_all = np.concatenate([[eid_ix[e]] * len(X_by_eid[e]) for e in eids])
        net_all, Tt_all = train_net(Xs_all, ys_all, T_all, G.EPOCHS, G.SEED)
        with torch.no_grad():
            z_all = torch.nn.functional.normalize(net_all(torch.tensor(Xs_all)), dim=1)
            sim_all = (z_all @ Tt_all.T).numpy()
        pred_all = sim_all.argmax(1)
        mem_acc = float((pred_all == ys_all).mean())
        lines.append(f"| **S2 记忆上界（7实体训练→检索自身）** | train_acc={mem_acc:.3f} | — | — |")

        # ---- LOEO with S1/S3 ----
        for fold, e_hold in enumerate(eids):
            eids_tr = [e for e in eids if e != e_hold]
            tr_ix = {e: i for i, e in enumerate(eids_tr)}
            Xs = np.concatenate([X_by_eid[e] for e in eids_tr])
            ys = np.concatenate([[tr_ix[e]] * len(X_by_eid[e]) for e in eids_tr])
            Ttr = T_all[[eid_ix[e] for e in eids_tr]]
            net, Tt = train_net(Xs, ys, Ttr, G.EPOCHS, G.SEED + fold)
            with torch.no_grad():
                # S1: seen 训练准确率
                z_seen = torch.nn.functional.normalize(net(torch.tensor(Xs)), dim=1)
                seen_acc = float(((z_seen @ Tt.T).argmax(1).numpy() == ys).mean())
                # S3: held-out 检索落点
                Xh = X_by_eid[e_hold]
                zh = torch.nn.functional.normalize(net(torch.tensor(Xh)), dim=1)
                Ta = torch.nn.functional.normalize(torch.tensor(T_all), dim=1)
                sim = (zh @ Ta.T).numpy()
            top = sim.argmax(1)
            top1 = float((top == eid_ix[e_hold]).mean())
            # top-2 候选
            uniq, cnt = np.unique(top, return_counts=True)
            top2 = [(eids[u], c / len(top)) for u, c in sorted(zip(uniq, cnt), key=lambda t: -t[1])[:2]]
            top2s = ", ".join(f"{e}:{p:.2f}" for e, p in top2)
            lines.append(f"| {e_hold} | {seen_acc:.3f} | {top1:.3f} | {top2s} |")
        lines.append("")

    with open(os.path.join(OUT, "exp05b_p6_g0_sanity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
