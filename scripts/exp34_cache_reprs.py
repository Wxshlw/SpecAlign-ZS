"""预计算并缓存 19 个候选的化学表示（CLM 语义嵌入 / ECFP 指纹）。

动机：exp08.build_text_variants_g1 每次调用都要加载一遍 CLM transformer，
且会额外计算 Shuffle-* / Random-Encoder 三个用不到的对照臂。exp34 / exp35 等
闭集实验只需 Original(CLM) 与 Descriptor(ECFP) 两臂，缓存后所有实验直接读盘。

输出：out/exp34_repr_cache.npz  → keys(19,), clm(19,d_clm), ecfp(19,d_ecfp)
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import p6_library as L
import exp05_p6_g0 as E5

OUT = r"D:/2026sci论文实验设计/论文6/out"


def prof_of(key):
    if key.startswith("M_"):
        return [(s, w) for _, s, w in G1.PROFILES_G1[key]]
    return [(s, w) for _, s, w in (L.COMPONENT_PROFILES.get(key) or G1.PROFILES_G1[key])]


def embed_batch(strs, tok, mdl):
    outs = {}
    with torch.no_grad():
        uniq = sorted(set(strs))
        for i in range(0, len(uniq), 16):
            batch = uniq[i:i + 16]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for s, vv in zip(batch, emb.numpy()):
                outs[s] = vv
    return outs


def main():
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    all_smi = sorted({s for k in keys for s, _ in prof_of(k)})
    print(f"[cache] {len(keys)} keys, {len(all_smi)} unique SMILES", flush=True)

    print("[cache] 加载 CLM ...", flush=True)
    tok, mdl = E5._load_clm(random_init=False)
    emb = embed_batch(all_smi, tok, mdl)

    def prof_embed(k):
        cs = prof_of(k); tw = sum(w for _, w in cs)
        return E5._unit(sum(emb[s] * (w / tw) for s, w in cs))

    T_clm = np.stack([prof_embed(k) for k in keys]).astype(np.float32)

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    # 两个维度都缓存：
    #   ecfp   —— 2048 位（与 exp08/exp31 一致，用于零样本检索类实验）
    #   ecfp384—— 384 位（与 CLM 嵌入同维，用于 exp34 的维度匹配公平对照）
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_of = {s: np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), float) for s in all_smi}
    gen384 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=384)
    fp384_of = {s: np.asarray(gen384.GetFingerprint(Chem.MolFromSmiles(s)), float) for s in all_smi}

    def prof_fp(k, table):
        cs = prof_of(k); tw = sum(w for _, w in cs)
        return E5._unit(sum(table[s] * (w / tw) for s, w in cs))

    T_ecfp = np.stack([prof_fp(k, fp_of) for k in keys]).astype(np.float32)
    T_ecfp384 = np.stack([prof_fp(k, fp384_of) for k in keys]).astype(np.float32)

    path = os.path.join(OUT, "exp34_repr_cache.npz")
    np.savez(path, keys=np.array(keys), clm=T_clm, ecfp=T_ecfp, ecfp384=T_ecfp384)
    print(f"[cache] 已保存 {path}  clm={T_clm.shape}  ecfp={T_ecfp.shape}  ecfp384={T_ecfp384.shape}", flush=True)


if __name__ == "__main__":
    main()
