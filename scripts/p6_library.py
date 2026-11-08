"""p6_library —— 论文6（SpecAlign-ZS）候选掺杂物库：真实实体 + 化学结构表征。

设计纪律（与 v9 §六 Step 1 / §五 3.4 对齐）
------------------------------------------------
* 每个掺杂物实体必须来自**盘上真实数据**的基质 × 掺杂物组合，禁止虚构实体。
* 化学结构（SMILES / 分子式 / 官能团）来源须可溯源：本表用 PubChem 标准 SMILES 与
  常用名对照；凡无法给出确定单体的**混合物**（核桃粉、木屑、植物蛋白…）一律以
  **主要成分单体**代表，并在 `repr_basis` 列写明代表依据（v9 §2.3 双通道设计）。
* 官能团**不手写**——由 RDKit SMARTS 客观匹配，避免主观臆造。
* 每个 SMILES 都必须通过 RDKit 解析；解析失败即拒绝入库（脚本会报错，不静默跳过）。

2026-10-30 初建：真实实体 16 个 / 8 基质（推翻 v9「真实数据不足、主靠合成基准」的前提）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ============================================================================
# 官能团 SMARTS（客观匹配，不手写官能团列表）
# ============================================================================
FG_SMARTS = {
    "alkyl":        "[CX4]",
    "alkene_C=C":   "[CX3]=[CX3]",
    "alkyne":       "[CX2]#C",
    "aromatic":     "a",
    "hydroxyl":     "[OX2H]",
    "carboxyl":     "[CX3](=O)[OX2H1]",
    "ester":        "[CX3](=O)[OX2][#6]",
    "amide":        "[CX3](=O)[NX3]",
    "amine":        "[NX3;H2,H1,H0;!$(N=*);!$(N[a])]",
    "aldehyde":     "[CX3H1](=O)[#6]",
    "ketone":       "[#6][CX3](=O)[#6]",
    "ether":        "[OD2]([#6])[#6]",
    "phenyl":       "c1ccccc1",
    "nitro":        "[NX3](=O)=O",
    "halide":       "[F,Cl,Br,I]",
    "sulfur":       "[#16]",
    "phosphate":    "[PX4](=O)",
}


@dataclass
class Entity:
    eid: str                 # 实体 ID
    matrix_cn: str           # 食品基质
    matrix_en: str
    adulterant_cn: str       # 掺杂物
    adulterant_en: str
    source: str              # 数据来源（数据集 ID）
    repr_compound_cn: str    # 代表化合物
    repr_compound_en: str
    smiles: str
    repr_basis: str          # 为什么用它代表（混合物 → 主要成分）
    chem_class: str          # 化学类别
    expect_formula: str = "" # 期望分子式（RDKit 校验用，留空=不校验）
    notes: str = ""


# ============================================================================
# 组分谱表示（composition-profile prior）—— 2026-10-30 作者批准升级
# 每个掺杂物用「主要成分 + 官能团分布」的多向量集合表示，以解决
# 单分子代表造成的结构简并（G1 亚油酸 / G2 油酸 / G3 麦芽糖）。
# 权重为相对含量近似值，可溯源食品组分文献（非精确定量）。
# 键 = 实体 eid；值 = [(成分中文名, SMILES, 权重), ...]，权重和≈1。
# ============================================================================
COMPONENT_PROFILES: dict[str, list[tuple[str, str, float]]] = {
    "A01": [("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.50),
            ("α-亚麻酸", "CC=CCC=CCC=CCCCCCCCCC(=O)O", 0.15),
            ("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.10),
            ("精氨酸(蛋白)", "NC(=N)NCCC[C@H](N)C(=O)O", 0.25)],
    "A02": [("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.50),
            ("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.33),
            ("精氨酸(蛋白)", "NC(=N)NCCC[C@H](N)C(=O)O", 0.17)],
    "A03": [("纤维二糖", "OC[C@H]1O[C@@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", 1.00)],
    "A04": [("麦芽糖", "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O", 0.50),
            ("葡萄糖(淀粉)", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O", 0.25),
            ("谷氨酰胺(面筋)", "NC(=O)CCC(N)C(=O)O", 0.25)],
    "A05": [("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.65),
            ("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.20),
            ("棕榈酸", "CCCCCCCCCCCCCCCC(=O)O", 0.15)],
    "A06": [("麦芽糖", "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O", 0.55),
            ("葡萄糖(麦芽糊精)", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O", 0.30),
            ("亮氨酸", "CC(C)CC(N)C(=O)O", 0.15)],
    "A07": [("苹果酸", "O=C(O)CC(O)C(=O)O", 1.00)],
    "A08": [("单质硫(S8)", "S1SSSSSSS1", 1.00)],
    "A09": [("亮氨酸", "CC(C)CC(N)C(=O)O", 0.40),
            ("赖氨酸", "NCCCC[C@H](N)C(=O)O", 0.30),
            ("谷氨酰胺", "NC(=O)CCC(N)C(=O)O", 0.30)],
    "A10": [("谷氨酰胺(面筋)", "NC(=O)CCC(N)C(=O)O", 0.60),
            ("亮氨酸", "CC(C)CC(N)C(=O)O", 0.40)],
    "A11": [("赖氨酸", "NCCCC[C@H](N)C(=O)O", 0.50),
            ("亮氨酸", "CC(C)CC(N)C(=O)O", 0.30),
            ("谷氨酰胺", "NC(=O)CCC(N)C(=O)O", 0.20)],
    "A12": [("精氨酸(花生蛋白)", "NC(=N)NCCC[C@H](N)C(=O)O", 0.60),
            ("亮氨酸", "CC(C)CC(N)C(=O)O", 0.40)],
    "A13": [("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.55),
            ("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.28),
            ("棕榈酸", "CCCCCCCCCCCCCCCC(=O)O", 0.12),
            ("硬脂酸", "CCCCCCCCCCCCCCCCCC(=O)O", 0.05)],
    "A14": [("阿魏酸(谷维素)", "COc1cc(C=CC(=O)O)ccc1O", 0.50),
            ("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.30),
            ("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.20)],
    "A15": [("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.45),
            ("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.33),
            ("花生酸(C20:0)", "CCCCCCCCCCCCCCCCCCCC(=O)O", 0.12),
            ("山嵛酸(C22:0)", "CCCCCCCCCCCCCCCCCCCCC(=O)O", 0.10)],
    "A16": [("芥酸(C22:1)", "CCCCCCCCCCCC=CCCCCCCCCC(=O)O", 0.40),
            ("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", 0.20),
            ("亚油酸", "CCCCC=CCC=CCCCCCCCCC(=O)O", 0.20),
            ("棕榈酸", "CCCCCCCCCCCCCCCC(=O)O", 0.10),
            ("硬脂酸", "CCCCCCCCCCCCCCCCCC(=O)O", 0.10)],
}


# ============================================================================
# 真实实体 16 个（全部来自 _shared_data 中已落地的数据集）
# ============================================================================
ENTITIES: list[Entity] = [
    # ---- 肉桂粉（D7，Mendeley rtbsf2r53x）----------------------------------
    Entity("A01", "肉桂粉", "cinnamon", "核桃粉", "walnut powder", "D7w",
           "亚油酸", "linoleic acid", "CCCCC=CCC=CCCCCCCCCC(=O)O",
           "核桃油脂主要脂肪酸（≈58–62% 亚油酸）", "脂肪酸/甘油三酯", "C18H32O2",
           "油脂类掺杂物；与 A05/A13 共用亚油酸代表单体"),
    Entity("A02", "肉桂粉", "cinnamon", "花生粉", "peanut powder", "D7p",
           "油酸", "oleic acid", "CCCCCCCCC=CCCCCCCCC(=O)O",
           "花生油脂主要脂肪酸（≈40–50% 油酸）", "脂肪酸/甘油三酯", "C18H34O2",
           "与 A15 共用油酸代表单体"),
    Entity("A03", "肉桂粉", "cinnamon", "木屑", "wood dust", "D7d",
           "纤维二糖", "cellobiose", "OC[C@H]1O[C@@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
           "纤维素重复单元（β-1,4-葡萄糖二聚）", "多糖/木质纤维素", "C12H22O11",
           "非食用掺杂物，化学类别与其余实体差异最大 → 预期 zero-shot 最好分"),

    # ---- 姜黄粉（D8）--------------------------------------------------------
    Entity("A04", "姜黄粉", "turmeric", "小麦粉", "wheat flour", "D8",
           "麦芽糖", "maltose", "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O",
           "淀粉酶解二糖单元（小麦粉主要成分为淀粉）", "多糖/淀粉", "C12H22O11",
           "与 A03 同为 C12H22O11 糖二聚（纤维二糖 vs 麦芽糖仅糖苷键构型不同）→ 结构近简并"),

    # ---- KAN trio（D1–D3, Bruker Tensor 27 ATR-FTIR）-----------------------
    Entity("A05", "橄榄油", "olive oil", "葵花籽油", "sunflower oil", "D1",
           "亚油酸", "linoleic acid", "CCCCC=CCC=CCCCCCCCCC(=O)O",
           "葵花籽油主要脂肪酸（≈55–70% 亚油酸）", "脂肪酸/甘油三酯", "C18H32O2",
           "★典型『化学相似物掺假』：与基质橄榄油同为甘油三酯"),
    Entity("A06", "咖啡", "roasted coffee", "麦芽粉", "malt flour", "D2",
           "麦芽糖", "maltose", "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O",
           "麦芽主要碳水化合物", "多糖/淀粉", "C12H22O11",
           "与 A04 同代表单体"),
    Entity("A07", "橙汁", "orange juice", "苹果汁", "apple juice", "D3",
           "苹果酸", "malic acid", "O=C(O)CC(O)C(=O)O",
           "苹果汁特征有机酸（区别于橙汁的柠檬酸）", "有机酸", "C4H6O5",
           "小分子有机酸；与油脂/多糖类别差异大"),

    # ---- 自采 S1 茶叶 IR ----------------------------------------------------
    Entity("A08", "茶叶", "tea", "硫磺", "sulfur", "S1_tea_ir",
           "单质硫（S8 环）", "cyclo-octasulfur", "S1SSSSSSS1",
           "非法添加物本体（熏硫/染色）", "单质/无机", "S8",
           "最简结构，几乎无有机官能团 → 检验『结构先验在无机掺杂物上是否失效』"),

    # ---- 自采 S2 乳清蛋白粉 IR（4 种植物蛋白）------------------------------
    Entity("A09", "乳清蛋白粉", "whey protein", "大豆蛋白", "soy protein isolate", "S2_protein_ir",
           "亮氨酸", "leucine", "CC(C)CC(N)C(=O)O",
           "植物分离蛋白的主要氨基酸残基代表", "蛋白/氨基酸", "C6H13NO2"),
    Entity("A10", "乳清蛋白粉", "whey protein", "小麦蛋白", "wheat gluten", "S2_protein_ir",
           "谷氨酰胺", "glutamine", "NC(=O)CCC(N)C(=O)O",
           "面筋蛋白高谷氨酰胺特征", "蛋白/氨基酸", "C5H10N2O3"),
    Entity("A11", "乳清蛋白粉", "whey protein", "白芸豆蛋白", "white kidney bean protein", "S2_protein_ir",
           "赖氨酸", "lysine", "NCCCC[C@H](N)C(=O)O",
           "豆类蛋白限制性氨基酸代表", "蛋白/氨基酸", "C6H14N2O2"),
    Entity("A12", "乳清蛋白粉", "whey protein", "花生蛋白", "peanut protein", "S2_protein_ir",
           "精氨酸", "arginine", "NC(=N)NCCC[C@H](N)C(=O)O",
           "花生蛋白高精氨酸特征", "蛋白/氨基酸", "C6H14N4O2"),

    # ---- 自采 S3 橄榄油 UV-Vis（4 种低价油）-------------------------------
    Entity("A13", "橄榄油", "olive oil", "玉米油", "corn oil", "S3_oil_uvvis",
           "亚油酸", "linoleic acid", "CCCCC=CCC=CCCCCCCCCC(=O)O",
           "玉米油主要脂肪酸（≈55–60% 亚油酸）", "脂肪酸/甘油三酯", "C18H32O2",
           "与 A01/A05 同代表单体 → 候选集内近简并"),
    Entity("A14", "橄榄油", "olive oil", "稻米油", "rice bran oil", "S3_oil_uvvis",
           "谷维素（γ-oryzanol 简化：阿魏酸）", "ferulic acid", "COc1cc(C=CC(=O)O)ccc1O",
           "稻米油特征微量成分（谷维素主体为阿魏酸酯）", "酚酸/甾醇酯", "C10H10O4",
           "以特征微量成分而非主脂肪酸代表 → 检验代表单体选择的影响"),
    Entity("A15", "橄榄油", "olive oil", "花生油", "peanut oil", "S3_oil_uvvis",
           "油酸", "oleic acid", "CCCCCCCCC=CCCCCCCCC(=O)O",
           "花生油主要脂肪酸（≈40–50% 油酸）", "脂肪酸/甘油三酯", "C18H34O2",
           "与 A02 同代表单体"),
    Entity("A16", "橄榄油", "olive oil", "菜籽油", "rapeseed oil", "S3_oil_uvvis",
           "芥酸", "erucic acid", "CCCCCCCCCCCC=CCCCCCCCCC(=O)O",
           "菜籽油特征长链脂肪酸（C22:1）", "脂肪酸/甘油三酯", "C22H42O2",
           "碳链更长 → 与 C18 系列可区分"),
]

# ============================================================================
# 实体范围（2026-10-30 作者裁定：自采三套 S1/S2/S3 经 exp04 质量诊断触发排除，
# 全部实验仅用公共数据集）。主证据实体 = A01–A07（全部来自已发表基准数据集）。
# A08–A16 化学定义与组分谱仍保留在库中（方法学完整性），但不进入主实验。
# ============================================================================
ACTIVE_EIDS = {"A01", "A02", "A03", "A04", "A05", "A06", "A07"}
SELF_SOURCES = {"S1_tea_ir", "S2_protein_ir", "S3_oil_uvvis"}


def active_entities(entities=ENTITIES):
    """返回进入主实验的实体（公共数据来源）。"""
    return [e for e in entities if e.source not in SELF_SOURCES]


def validate(entities=ENTITIES, verbose: bool = True):
    """RDKit 解析 + 分子式校验。失败直接抛错（不静默跳过）。"""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    rows, errs = [], []
    for e in entities:
        m = Chem.MolFromSmiles(e.smiles)
        if m is None:
            errs.append(f"{e.eid} {e.adulterant_cn}: SMILES 无法解析 -> {e.smiles}")
            continue
        formula = rdMolDescriptors.CalcMolFormula(m)
        ok = (not e.expect_formula) or (formula == e.expect_formula)
        if not ok:
            errs.append(f"{e.eid} {e.adulterant_cn}: 分子式 {formula} != 期望 {e.expect_formula}")
        canon = Chem.MolToSmiles(m)
        rows.append(dict(eid=e.eid, matrix=e.matrix_cn, adulterant=e.adulterant_cn,
                         repr=e.repr_compound_cn, smiles=canon, formula=formula,
                         expect=e.expect_formula, formula_ok=ok,
                         chem_class=e.chem_class, source=e.source))
        if verbose:
            flag = "" if ok else f"  ⚠️ 期望 {e.expect_formula}"
            print(f"  {e.eid} {e.matrix_cn}←{e.adulterant_cn:<8s} {formula:<10s} {canon[:52]:<52s}{flag}")
    if errs:
        raise SystemExit("SMILES 校验失败：\n  " + "\n  ".join(errs))
    return rows


if __name__ == "__main__":
    print("[p6_library] 实体数 =", len(ENTITIES))
    validate()
