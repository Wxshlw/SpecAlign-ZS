"""exp00 —— 论文6 候选掺杂物库：结构表征 + 相似度矩阵 + 简并性诊断（Step 1 产出）。

输出（论文6/out/）
----------------
* exp00_p6_library.csv   实体表（含 canonical SMILES / 分子式 / 官能团命中）
* exp00_p6_fg.csv        官能团二值矩阵（RDKit SMARTS 客观匹配，不手写）
* exp00_p6_tanimoto.csv  ECFP4 / MACCS Tanimoto + 官能团 Jaccard 两两矩阵
* exp00_p6_degeneracy.json  ★ 简并性诊断：leave-one-out 下每个 target 有多少
                            distractor 与之结构近似（Tanimoto ≥ τ）
* exp00_p6_library.md    人读报告

为什么先做简并性诊断（★ 本实验的核心预警）
------------------------------------------------
v9 的可证伪假设 H 说「结构相似度 ↑ → zero-shot top-k 准确率 ↑」。但食品掺假的现实是
**掺假物常被刻意选成化学上与基质或其他候选物相近者**（低价油掺高价油、植物蛋白掺乳清蛋白）。
若候选集内多个实体共享同一代表单体（如核桃/葵花籽/玉米油都以亚油酸为代表），它们两两
Tanimoto = 1.0，检索时**理论上不可分**——此时 top-1 准确率必然塌陷，且塌陷不是模型的问题，
是**问题本身的不可辨识性**。这个诊断必须在跑 G0 之前做，否则 G0 的"不过"会被误读成
"化学语义无贡献"，从而错误触发撤稿路径。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")
os.makedirs(OUT, exist_ok=True)

from p6_library import ENTITIES, FG_SMARTS, validate  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdMolDescriptors, Descriptors  # noqa: E402
from rdkit import DataStructs  # noqa: E402


def build():
    rows = validate(verbose=False)
    mols, fg_rows = [], []
    for e in ENTITIES:
        m = Chem.MolFromSmiles(e.smiles)
        mols.append(m)
        hits = {}
        for name, sma in FG_SMARTS.items():
            patt = Chem.MolFromSmarts(sma)
            hits[name] = int(m.HasSubstructMatch(patt)) if patt is not None else 0
        hits["_eid"] = e.eid
        hits["_n_heavy_atoms"] = m.GetNumHeavyAtoms()
        hits["_molweight"] = round(Descriptors.MolWt(m), 2)
        fg_rows.append(hits)

    # ---- 指纹 ----
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ecfp = [gen.GetFingerprint(m) for m in mols]
    maccs = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    fg_mat = np.array([[r[k] for k in FG_SMARTS] for r in fg_rows], float)

    n = len(ENTITIES)
    T_ecfp = np.zeros((n, n)); T_maccs = np.zeros((n, n)); J_fg = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T_ecfp[i, j] = DataStructs.TanimotoSimilarity(ecfp[i], ecfp[j])
            T_maccs[i, j] = DataStructs.TanimotoSimilarity(maccs[i], maccs[j])
            inter = (fg_mat[i] * fg_mat[j]).sum()
            union = np.clip((fg_mat[i] + fg_mat[j]) > 0, 0, 1).sum()
            J_fg[i, j] = inter / union if union else 0.0

    ids = [e.eid for e in ENTITIES]
    labels = [f"{e.eid} {e.matrix_cn}←{e.adulterant_cn}" for e in ENTITIES]

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp00_p6_library.csv"),
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(fg_rows).to_csv(os.path.join(OUT, "exp00_p6_fg.csv"),
                                 index=False, encoding="utf-8-sig")
    tan = pd.DataFrame({
        "pair_i": [ids[i] for i in range(n) for _ in range(n)],
        "pair_j": [ids[j] for i in range(n) for j in range(n)],
        "ecfp4_tanimoto": T_ecfp.ravel(),
        "maccs_tanimoto": T_maccs.ravel(),
        "fg_jaccard": J_fg.ravel(),
    })
    tan.to_csv(os.path.join(OUT, "exp00_p6_tanimoto.csv"), index=False, encoding="utf-8-sig")

    # ---- ★ 简并性诊断：leave-one-out 下 target 有多少近似 distractor ----
    taus = [0.99, 0.9, 0.8, 0.6]
    deg = {}
    for k, tname in enumerate(["ecfp4", "maccs"]):
        M = T_ecfp if tname == "ecfp4" else T_maccs
        per = {}
        for i in range(n):
            others = np.delete(M[i], i)
            per[ids[i]] = {f"n_ge_{t}": int((others >= t).sum()) for t in taus}
        deg[tname] = dict(
            n_entities=n,
            pairs_at_1_00=int(((M >= 0.999) & ~np.eye(n, dtype=bool)).sum() // 2),
            pairs_ge_0_90=int(((M >= 0.9) & ~np.eye(n, dtype=bool)).sum() // 2),
            pairs_ge_0_80=int(((M >= 0.8) & ~np.eye(n, dtype=bool)).sum() // 2),
            mean_offdiag=float(M[~np.eye(n, dtype=bool)].mean()),
            per_target=per,
        )

    # 共享代表单体的组（canonical SMILES 完全相同）
    canon = [r["smiles"] for r in rows]
    groups = {}
    for eid, c in zip(ids, canon):
        groups.setdefault(c, []).append(eid)
    shared = {c: v for c, v in groups.items() if len(v) > 1}
    deg["identical_repr_groups"] = {f"G{i+1}": v for i, (_, v) in enumerate(shared.items())}
    deg["n_identical_groups"] = len(shared)
    deg["n_entities_in_identical_groups"] = sum(len(v) for v in shared.values())

    with open(os.path.join(OUT, "exp00_p6_degeneracy.json"), "w", encoding="utf-8") as fh:
        json.dump(deg, fh, ensure_ascii=False, indent=2)

    # ---- 人读报告 ----
    L = []
    L.append("# 论文6 Step 1｜候选掺杂物库 —— 结构表征与简并性诊断（2026-10-30）\n")
    L.append(f"真实实体 **{n}** 个，覆盖 **{len({e.matrix_cn for e in ENTITIES})}** 个食品基质、"
             f"**{len({e.source for e in ENTITIES})}** 个数据集。全部来自盘上真实数据，无合成实体。\n")
    L.append("## 1. 实体表\n")
    L.append("| ID | 基质 | 掺杂物 | 代表化合物 | 分子式 | 化学类别 | 来源 |")
    L.append("|---|---|---|---|---|---|---|")
    for r, e in zip(rows, ENTITIES):
        L.append(f"| {r['eid']} | {e.matrix_cn} | {e.adulterant_cn} | {e.repr_compound_cn} | "
                 f"{r['formula']} | {e.chem_class} | {e.source} |")
    L.append("\n> 混合物型掺杂物（核桃粉 / 木屑 / 植物蛋白 / 各种油）一律以**主要成分单体**代表，"
             "代表依据见 `p6_library.py` 的 `repr_basis`；官能团由 RDKit SMARTS 客观匹配，不手写。\n")
    L.append("## 2. ★ 简并性诊断（决定 G0 能不能跑）\n")
    L.append(f"- 共享**完全相同代表单体**的实体组：**{deg['n_identical_groups']}** 组，"
             f"涉及 **{deg['n_entities_in_identical_groups']}** 个实体")
    for g, v in deg["identical_repr_groups"].items():
        L.append(f"  - {g}: {' / '.join(v)}")
    for tname in ("ecfp4", "maccs"):
        d = deg[tname]
        L.append(f"- **{tname}**：Tanimoto=1.00 的实体对 **{d['pairs_at_1_00']}**；"
                 f"≥0.90 **{d['pairs_ge_0_90']}**；≥0.80 **{d['pairs_ge_0_80']}**；"
                 f"非对角均值 **{d['mean_offdiag']:.3f}**")
    L.append("\n### 每个 target 的近似 distractor 数（leave-one-adulterant-out，ECFP4）\n")
    L.append("| target | ≥0.99 | ≥0.90 | ≥0.80 | ≥0.60 |")
    L.append("|---|---|---|---|---|")
    for i, eid in enumerate(ids):
        p = deg["ecfp4"]["per_target"][eid]
        lbl = f"{eid} {ENTITIES[i].matrix_cn}←{ENTITIES[i].adulterant_cn}"
        L.append(f"| {lbl} | {p['n_ge_0.99']} | {p['n_ge_0.9']} | {p['n_ge_0.8']} | {p['n_ge_0.6']} |")
    L.append("\n## 3. 对 v9 假设 H 的直接后果\n")
    L.append("- 若某 target 存在 Tanimoto≈1 的 distractor，则在 leave-one-adulterant-out 下"
             "**top-1 检索在原理上不可分**；此时 top-1 准确率低**不能**解读为「化学语义无贡献」。")
    L.append("- G0 的判据必须**分层**：近似可分实体组与简并实体组分别报告；"
             "只在简并组上报准确率会误判生死。（详见 §对 G0 的修订建议）")
    with open(os.path.join(OUT, "exp00_p6_library.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print(f"[exp00] entities={n} matrices={len({e.matrix_cn for e in ENTITIES})}")
    print(f"[exp00] identical-repr groups={deg['n_identical_groups']} "
          f"(涉及 {deg['n_entities_in_identical_groups']} 实体)")
    for g, v in deg["identical_repr_groups"].items():
        print(f"         {g}: {' / '.join(v)}")
    for tname in ("ecfp4", "maccs"):
        d = deg[tname]
        print(f"[exp00] {tname}: pairs@1.00={d['pairs_at_1_00']} >=0.90={d['pairs_ge_0_90']} "
              f">=0.80={d['pairs_ge_0_80']} mean_offdiag={d['mean_offdiag']:.3f}")
    return deg, rows, T_ecfp


if __name__ == "__main__":
    build()
