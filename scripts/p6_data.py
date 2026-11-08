"""p6_data —— 论文6（SpecAlign-ZS）统一数据层。

整合论文6 用到的全部 10 个数据集（6 公共 + 3 自采 + 论文8 公共加载器共用缓存）：

公共（经论文8 `unmix_ftir.data`，与论文8 读同一缓存，数字天然一致）
  D7w walnut-in-cinnamon   D7p peanut-in-cinnamon   D7d wood-dust-in-cinnamon
  D1  KAN olive oil         D2  KAN coffee           D3  KAN orange juice
  D8  wheat-flour-in-turmeric

自采三套（论文5 `p5_data`，单一副本在 _shared_data/）
  S1 茶叶 IR（硫磺/配方类别）  S2 乳清蛋白粉 IR（4 植物蛋白）  S3 橄榄油 UV-Vis（4 低价油×5 水平）

纪律（继承论文5 §设计纪律）
  * 轴一律升序返回；不做插值。
  * 自采三套无样品 ID → proxy_groups（余弦单链接）代理分组。
  * 浓度梯度集（D7/D8/D1–D3/S3）报 LOD/RMSEP 用，不报浓度仅输出身份（自采三套）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

P8 = r"D:/2026sci论文实验设计/论文8"
P5 = r"D:/2026sci论文实验设计/论文5/scripts"
P6 = r"D:/2026sci论文实验设计/论文6"
for p in (P8, P5, P6):
    if p not in sys.path:
        sys.path.insert(0, p)

import unmix_ftir.data as pub          # noqa: E402
import p5_data                         # noqa: E402
import p6_library as lib               # noqa: E402

# 数据集来源 ID → 加载函数
LOADERS = {
    "D7w": lambda: pub.load_cinnamon("walnut"),
    "D7p": lambda: pub.load_cinnamon("peanut"),
    "D7d": lambda: pub.load_cinnamon("wood dust"),
    "D1":  lambda: pub.load_kan("oil"),
    "D2":  lambda: pub.load_kan("coffee"),
    "D3":  lambda: pub.load_kan("juice"),
    "D8":  lambda: pub.load_turmeric(),
    "S1":  lambda: p5_data.load_tea(verbose=False),
    "S2":  lambda: p5_data.load_protein(verbose=False),
    "S3":  lambda: p5_data.load_oil(verbose=False),
}

SELF_SETS = ["S1", "S2", "S3"]   # 自采三套（作者 2026-10-30：采集期有瑕疵，仅质量问题触发排除）

MODALITY = {"D7w": "FTIR", "D7p": "FTIR", "D7d": "FTIR", "D8": "FTIR",
            "D1": "FTIR", "D2": "FTIR", "D3": "FTIR",
            "S1": "FTIR", "S2": "FTIR", "S3": "UV-Vis"}

# 实体 eid → 数据集来源 ID（来自 p6_library.ENTITIES 的 source 字段）
ENTITY_SOURCE = {e.source if e.source in LOADERS else e.source: e.eid for e in lib.ENTITIES}


def load_all(verbose: bool = True) -> dict:
    """加载全部 10 个数据集，返回 {src_id: dict}。"""
    out = {}
    for sid, fn in LOADERS.items():
        try:
            D = fn()
            if "axis" not in D and "wvn" in D:      # 公共加载器用 wvn，自采用 axis
                D["axis"] = D["wvn"]
            out[sid] = D
            if verbose:
                print(f"[{sid}] X={D['X'].shape} axis[{D['axis'][0]:.1f}..{D['axis'][-1]:.1f}] "
                      f"val[{np.nanmin(D['X']):.2f},{np.nanmax(D['X']):.2f}] "
                      f"y={None if D.get('y') is None else sorted(set(np.asarray(D['y']).tolist()))[:6]}")
        except Exception as ex:
            import traceback
            print(f"[{sid}] LOAD FAILED: {ex}")
            traceback.print_exc()
    return out


# 实体库 source 名 → 加载器键
SRC_ALIAS = {"S1_tea_ir": "S1", "S2_protein_ir": "S2", "S3_oil_uvvis": "S3"}


def entity_to_dataset() -> dict:
    """返回 {eid: src_id}，把 16 实体映射到其数据来源（加载器键）。"""
    m = {}
    for e in lib.ENTITIES:
        m[e.eid] = SRC_ALIAS.get(e.source, e.source)
    return m


def proxy_groups(X: np.ndarray, taus=(0.999, 0.995, 0.99, 0.95)) -> dict:
    """无样品 ID 时的余弦相似度单链接代理分组（复刻 p5_data.proxy_groups）。"""
    Xz = X - X.mean(axis=1, keepdims=True)
    Xn = Xz / np.clip(np.linalg.norm(Xz, axis=1, keepdims=True), 1e-12, None)
    S = Xn @ Xn.T
    n = len(X)
    out = {}
    for tau in taus:
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] >= tau:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        out[f"tau_{tau}"] = len({find(i) for i in range(n)})
    out["n_spectra"] = n
    return out


if __name__ == "__main__":
    load_all()
    print("\nEntity → dataset:")
    for eid, src in entity_to_dataset().items():
        print(f"  {eid} -> {src}")
