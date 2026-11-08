r"""exp81a —— 先验（码本）动物园缓存：把"普通先验 / 普通语义"按**同一配方**生成。

动机（用户要求）
--------------
主表只和 RF/SVM 比、只和随机码本比，太单薄。需要回答：
    「我们的化学语义先验，是不是比**常见的普通先验**更好？」
    「这个增益是不是**换任何算法都还在**（算法无关）？」

核心公平性要求
--------------
所有先验必须走**完全相同的生成配方**，否则差异来自配方而非先验本身：
    1. 每个候选 = 一个"组分谱" profile：[(SMILES, 权重)]（与 exp34 完全一致）
    2. profile 向量 = Σ w_i · v(SMILES_i) / Σ w_i       （加权平均，同一权重）
    3. 统一映射到 **384 维**（与 CLM 同维），行 L2 归一化
这一步是关键：ECFP 2048 位 vs MACCS 167 位 vs CLM 384 维，维度不同会混淆
"先验信息量"与"岭回归容量"，因此**统一折到 384 维**做主对比。

先验清单
--------
  clm        化学语言模型语义嵌入（本文）              384 维（原生）
  ecfp384    Morgan/ECFP4 折叠指纹（经典化学指纹）     384 维（原生）
  ecfp2048   同上 2048 位 → 带符号哈希折到 384
  maccs      MACCS 骨架键（167）→ 折到 384
  avalon     Avalon 指纹（512）→ 折到 384
  rdkit      RDKit 全量描述符（217，标准化）→ 高斯投影 384
  physchem   元素计数 + MW/logP/TPSA/环数…（最"普通"的化学先验）→ 投影 384
  charbag    SMILES 字符袋（朴素**语义**先验，无化学知识）→ 折到 384
  onehot     类别索引（19 维，零化学信息，退化的"无先验"对照）
  rand384    同形随机码本（阴性对照，与 exp79 一致）

输出 out/exp81_prior_cache.npz：keys(19,), 各先验 (19,384)，另存 native_dim / 相似度诊断。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import p6_library as L

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
DIM = 384
HASH_SEED = 0


# ---------------------------------------------------------------------------
def prof_of(key):
    """与 exp34_cache_reprs 完全一致的 profile 定义。"""
    if key.startswith("M_"):
        return [(s, w) for _, s, w in G1.PROFILES_G1[key]]
    return [(s, w) for _, s, w in (L.COMPONENT_PROFILES.get(key)
                                   or G1.PROFILES_G1[key])]


def hash_fold(v, dim=DIM, seed=HASH_SEED):
    """带符号哈希折叠：把任意长度的计数/位向量线性折到 dim 维（对所有先验同一折叠）。"""
    v = np.asarray(v, float)
    ix = np.arange(len(v), dtype=np.int64)
    bucket = (ix * 2654435761 + seed * 40503) % dim
    sign = np.where(((ix * 40503 + seed * 7919) % 7) < 4, 1.0, -1.0)
    out = np.zeros(dim)
    np.add.at(out, bucket, v * sign)
    return out


def unit(v):
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else np.zeros_like(v)


# ---------------------------------------------------------------------------
def mol_vectors(all_smi):
    """每个 SMILES → 各类原生向量字典。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
    from rdkit.Chem import rdMolDescriptors as RDM
    from rdkit.Avalon import pyAvalonTools as Ava
    RDLogger.DisableLog("rdApp.*")

    gen2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    gen384 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=384)

    CHARS = sorted({c for s in all_smi for c in s})
    cidx = {c: i for i, c in enumerate(CHARS)}
    ELEMS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I", "Na", "K", "Ca"]

    raw = {k: {} for k in ("ecfp2048", "ecfp384", "maccs", "avalon",
                           "rdkit", "physchem", "charbag")}
    for s in all_smi:
        m = Chem.MolFromSmiles(s)
        assert m is not None, f"SMILES 解析失败: {s}"
        raw["ecfp2048"][s] = np.asarray(gen2048.GetFingerprint(m), float)
        raw["ecfp384"][s] = np.asarray(gen384.GetFingerprint(m), float)
        raw["maccs"][s] = np.asarray(MACCSkeys.GenMACCSKeys(m), float)[:167]
        raw["avalon"][s] = np.asarray(Ava.GetAvalonFP(m, nBits=512), float)

        d = Descriptors.CalcMolDescriptors(m)
        vv = np.array([d[k] for k in sorted(d)], float)
        vv[~np.isfinite(vv)] = 0.0
        raw["rdkit"][s] = vv

        counts = np.array([sum(1 for a in m.GetAtoms() if a.GetSymbol() == e)
                           for e in ELEMS], float)
        extra = np.array([
            Descriptors.MolWt(m), float(m.GetNumHeavyAtoms()),
            Descriptors.MolLogP(m), Descriptors.TPSA(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m), RDM.CalcNumRings(m),
            Descriptors.RingCount(m), Descriptors.FractionCSP3(m),
            RDM.CalcNumAromaticRings(m), RDM.CalcNumHeteroatoms(m),
        ], float)
        extra[~np.isfinite(extra)] = 0.0
        raw["physchem"][s] = np.concatenate([counts, extra])

        cb = np.zeros(len(CHARS))
        for c in s:
            cb[cidx[c]] += 1.0
        raw["charbag"][s] = cb
    return raw


def main():
    ent, bg, _ = G1.load_g1()
    keys = sorted(ent.keys()) + [f"M_{m}" for m in sorted(bg.keys())]
    all_smi = sorted({s for k in keys for s, _ in prof_of(k)})
    print(f"[exp81a] {len(keys)} keys, {len(all_smi)} unique SMILES", flush=True)

    # ---- CLM（本文先验，沿用 exp34 缓存，保证与所有历史实验一致）----
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    ckeys = list(z["keys"])
    assert ckeys == keys, "exp34 缓存的 key 顺序与当前不一致，禁止对齐！"
    T = {"clm": np.asarray(z["clm"], float)}

    raw = mol_vectors(all_smi)

    def prof_native(name):
        """profile 加权平均后的**原生**向量（不做投影）。"""
        out = []
        for k in keys:
            cs = prof_of(k)
            tw = sum(w for _, w in cs)
            out.append(sum(raw[name][s] * (w / tw) for s, w in cs))
        return np.stack(out)

    native = {n: prof_native(n) for n in raw}
    # RDKit 描述符量纲悬殊 → 按列标准化后再投影
    rr = native["rdkit"]
    sd = rr.std(0) + 1e-9
    native["rdkit"] = (rr - rr.mean(0)) / sd
    pp = native["physchem"]
    sd2 = pp.std(0) + 1e-9
    native["physchem"] = (pp - pp.mean(0)) / sd2

    rng = np.random.default_rng(HASH_SEED)
    proj = {}
    for n in ("rdkit", "physchem"):
        d = native[n].shape[1]
        proj[n] = rng.normal(size=(d, DIM)) / np.sqrt(DIM)   # 高斯随机投影

    for n, V in native.items():
        if n == "ecfp384":
            T["ecfp384"] = np.stack([unit(v) for v in V])
        elif n == "rdkit" or n == "physchem":
            T[n] = np.stack([unit(v @ proj[n]) for v in V])
        else:
            T[n] = np.stack([unit(hash_fold(v)) for v in V])

    # ---- 无化学信息对照 ----
    T["onehot"] = np.eye(len(keys), DIM)                     # 19 维 one-hot 补零
    R = rng.normal(size=(len(keys), DIM)).astype(float)
    T["rand384"] = np.stack([unit(v) for v in R])

    # ---- 诊断：候选间相似度（是否退化）----
    diag = {}
    for n, V in T.items():
        C = V @ V.T
        iu = np.triu_indices(len(keys), 1)
        diag[n] = dict(dim=int(V.shape[1]),
                       cos_mean=float(C[iu].mean()),
                       cos_max=float(C[iu].max()),
                       n_pairs_gt09=int((C[iu] > 0.9).sum()))
        print(f"  {n:10s} dim={V.shape[1]:5d} cos均值={C[iu].mean():+.4f} "
              f"cos最大={C[iu].max():.4f} >0.9对数={int((C[iu] > 0.9).sum())}",
              flush=True)

    path = os.path.join(OUT, "exp81_prior_cache.npz")
    np.savez(path, keys=np.array(keys),
             **{n: v.astype(np.float32) for n, v in T.items()})
    with open(os.path.join(OUT, "exp81_prior_diag.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(native_dim={n: int(native[n].shape[1])
                                   for n in native}, diag=diag),
                  f, ensure_ascii=False, indent=1)
    print(f"[exp81a] 已保存 {path}  先验数={len(T)}", flush=True)


if __name__ == "__main__":
    main()
