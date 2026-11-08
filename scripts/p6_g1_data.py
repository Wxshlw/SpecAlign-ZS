"""p6_g1_data —— G1 候选库统一数据层（exp06 裁定：13 实体，全公开真实数据）。

实体（13）：
  A01–A07  真实掺混梯度（D7w/p/d、D8、D1、D2、D3），query = 掺混谱（y>0）
  HSYRUP   糖浆掺蜂蜜（F1 Kaggle），query = 10/20/40% 掺混谱
  MEATCH/MEATP/MEATT  鸡/猪/火鸡肉泥（qib MIRFreshMeats），query = 纯品谱
  COFA/COFR           Arabica/Robusta 速溶咖啡（quadram），query = 纯品谱

矩阵背景（6，训练时作对齐目标、评估时非候选）：
  cinnamon / turmeric / olive_oil / coffee_roasted / orange_juice / honey
  —— 每折训练都含 held-out 实体所在矩阵的纯谱背景 → 消除 exp05 的 OOD 塌陷。

公共谱窗：各源交集 ≈ [1005, 1800] cm⁻¹（指纹区），统一网格 240 点。
预处理：SNV + SG(9,2,deriv=1)。
"""
from __future__ import annotations
import os, sys, io, zipfile
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SHARED = r"D:/2026sci论文实验设计/_shared_data"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6_data as P          # D1–D8 公共集加载器

GRID_LO, GRID_HI, GRID_N = 1005.4, 1799.0, 240

# ----------------------------------------------------------------------------
# 组分谱定义（SMILES 已过 RDKit 分子式校验）
# ----------------------------------------------------------------------------
S_SUCROSE = "OC[C@H]1O[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H](O)[C@@H]1O"
S_GLUCOSE = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O"
S_FROSE = "OC[C@@]1(O)O[C@H](CO)[C@@H](O)[C@H]1O"
S_CAFF = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
S_CHLORO = "O=C(O)[C@H]1[C@H](O)[C@H](O)[C@H](OC(=O)/C=C/c2ccc(O)c(O)c2)[C@@H](O)C1"
S_CINNAMAL = "O=CC=CC1=CC=CC=C1"
S_CUMARIN = "O=C1OC2=CC=CC=C2C=C1"
S_CURCUMIN = "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O"
S_CITRIC = "OC(=O)CC(O)(CC(=O)O)C(=O)O"
S_SQUA = "CC(C)=CCCC(C)=CC=C(C)CCCC(C)=CCCC(C)=CCCC=C(C)C"
S_LEU = "CC(C)CC(N)C(=O)O"
S_ALA = "CC(N)C(=O)O"
S_GLN = "NC(=O)CCC(N)C(=O)O"
S_PALM = "CCCCCCCCCCCCCCCC(=O)O"
S_OLE = "CCCCCCCCC=CCCCCCCCC(=O)O"
S_LINO = "CCCCC=CCC=CCCCCCCCCC(=O)O"
# —— G1v2 新增：种属/品种标志物（SMILES 已过 RDKit 分子式校验，2026-10-30）——
S_CARN = "NCCC(=O)N[C@@H](Cc1c[nH]cn1)C(=O)O"      # 肌肽 carnosine  C9H14N4O3 / 226.23（猪等哺乳动物骨骼肌主二肽）
S_ANS = "Cn1cc(C[C@H](NC(=O)CCN)C(=O)O)nc1"        # 鹅肌肽 anserine C10H16N4O3 / 240.26（禽类肌肉显著高于猪）
S_TRIG = "C[n+]1cccc(c1)C(=O)[O-]"                 # 胡芦巴碱 trigonelline C7H7NO2 / 137.14（Arabica > Robusta）
S_STEAR = "CCCCCCCCCCCCCCCCCC(=O)O"                # 硬脂酸 stearic acid C18H36O2 / 284.48（猪 SFA 占比高于禽）

PROFILES_G1 = {
    # —— 13 候选实体 ——
    "HSYRUP":  [("sucrose", S_SUCROSE, .55), ("glucose", S_GLUCOSE, .25), ("fructose", S_FROSE, .20)],
    # —— G1v2（2026-10-30）：姊妹实体文本先验细化，依据是公开文献的种属/品种成分差异，
    #      非拟合测试集。定性事实：①禽类肌肉二肽以鹅肌肽为主、猪以肌肽为主；
    #      ②禽肉 PUFA(亚油酸)占比高于猪，猪 MUFA(油酸)/SFA(棕榈酸+硬脂酸)占比更高；
    #      ③Robusta 咖啡因约为 Arabica 两倍以上且绿原酸更高，Arabica 胡芦巴碱与脂类更高。
    "MEATCH":  [("leucine", S_LEU, .15), ("alanine", S_ALA, .10), ("glutamine", S_GLN, .15),
                ("anserine", S_ANS, .08), ("carnosine", S_CARN, .03),
                ("palmitic", S_PALM, .10), ("stearic", S_STEAR, .04),
                ("oleic", S_OLE, .16), ("linoleic", S_LINO, .19)],
    "MEATP":   [("leucine", S_LEU, .15), ("alanine", S_ALA, .10), ("glutamine", S_GLN, .14),
                ("anserine", S_ANS, .01), ("carnosine", S_CARN, .07),
                ("palmitic", S_PALM, .15), ("stearic", S_STEAR, .08),
                ("oleic", S_OLE, .24), ("linoleic", S_LINO, .06)],
    "MEATT":   [("leucine", S_LEU, .18), ("alanine", S_ALA, .11), ("glutamine", S_GLN, .17),
                ("anserine", S_ANS, .06), ("carnosine", S_CARN, .02),
                ("palmitic", S_PALM, .07), ("stearic", S_STEAR, .03),
                ("oleic", S_OLE, .13), ("linoleic", S_LINO, .23)],
    "COFA":    [("caffeine", S_CAFF, .07), ("trigonelline", S_TRIG, .11),
                ("chlorogenic", S_CHLORO, .18), ("sucrose", S_SUCROSE, .10),
                ("oleic", S_OLE, .20), ("linoleic", S_LINO, .17), ("palmitic", S_PALM, .09)],
    "COFR":    [("caffeine", S_CAFF, .18), ("trigonelline", S_TRIG, .05),
                ("chlorogenic", S_CHLORO, .26), ("sucrose", S_SUCROSE, .06),
                ("oleic", S_OLE, .11), ("linoleic", S_LINO, .10), ("palmitic", S_PALM, .06)],
    # —— 6 矩阵背景 ——
    "M_cinnamon":  [("cinnamaldehyde", S_CINNAMAL, .40), ("coumarin", S_CUMARIN, .20),
                    ("linoleic", S_LINO, .15), ("palmitic", S_PALM, .15)],
    "M_turmeric":  [("curcumin", S_CURCUMIN, .35), ("linoleic", S_LINO, .20), ("sucrose", S_SUCROSE, .15)],
    "M_oliveoil":  [("oleic", S_OLE, .55), ("linoleic", S_LINO, .15), ("squalene", S_SQUA, .15), ("palmitic", S_PALM, .15)],
    "M_coffeeR":   [("caffeine", S_CAFF, .15), ("chlorogenic", S_CHLORO, .20),
                    ("linoleic", S_LINO, .20), ("palmitic", S_PALM, .15)],
    "M_juice":     [("citric", S_CITRIC, .40), ("sucrose", S_SUCROSE, .25), ("fructose", S_FROSE, .20)],
    "M_honey":     [("fructose", S_FROSE, .40), ("glucose", S_GLUCOSE, .30), ("sucrose", S_SUCROSE, .15)],
}
# A01–A07 组分谱沿用 p6_library.COMPONENT_PROFILES
import p6_library as L

ENTITY_META = {  # eid -> (中文名, matrix_id, kind)
    "A01": ("核桃粉", "cinnamon", "adulteration"),
    "A02": ("花生粉", "cinnamon", "adulteration"),
    "A03": ("木屑", "cinnamon", "adulteration"),
    "A04": ("小麦粉", "turmeric", "adulteration"),
    "A05": ("葵花籽油", "oliveoil", "adulteration"),
    "A06": ("麦芽粉", "coffeeR", "adulteration"),
    "A07": ("苹果汁", "juice", "adulteration"),
    "HSYRUP": ("糖浆", "honey", "adulteration"),
    "MEATCH": ("鸡肉泥", "meat", "authentication"),
    "MEATP": ("猪肉泥", "meat", "authentication"),
    "MEATT": ("火鸡肉泥", "meat", "authentication"),
    "COFA": ("Arabica", "coffeeI", "authentication"),
    "COFR": ("Robusta", "coffeeI", "authentication"),
}
SID2EID = {"D7w": "A01", "D7p": "A02", "D7d": "A03", "D8": "A04", "D1": "A05", "D2": "A06", "D3": "A07"}
SID2MID = {"D7w": "cinnamon", "D7p": "cinnamon", "D7d": "cinnamon", "D8": "turmeric",
           "D1": "oliveoil", "D2": "coffeeR", "D3": "juice"}


def _resample(X, axis, grid):
    out = np.empty((len(X), len(grid)), np.float64)
    for i, row in enumerate(X):
        out[i] = np.interp(grid, axis, row)
    return out


def _prep(X):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True)
    Xz = (X - mu) / (sd + 1e-9)
    return savgol_filter(Xz, 9, 2, deriv=1, axis=1)


def _grid():
    return np.linspace(GRID_LO, GRID_HI, GRID_N)


# ----------------------------------------------------------------------------
# 各源加载
# ----------------------------------------------------------------------------
def _load_pub():
    """D1–D8 → 实体 query（y>0）与矩阵背景（y==0）。"""
    sets = P.load_all(verbose=False)
    ent, bg = {}, {}
    for sid, D in sets.items():
        if sid not in SID2EID:
            continue
        grid = _grid()
        Xg = _resample(np.asarray(D["X"], float), np.asarray(D["axis"], float), grid)
        Xp = _prep(Xg)
        y = np.asarray(D["y"], float)
        eid = SID2EID[sid]
        ent[eid] = Xp[y > 0]
        mid = SID2MID[sid]
        bg.setdefault(mid, []).append(Xp[y == 0])
    return ent, {k: np.concatenate(v) for k, v in bg.items()}


def _load_pub_conc():
    """同 _load_pub，额外返回 conc：{eid: 掺混浓度(%) 数组}（D1–D8 的 y 即浓度百分比）。"""
    sets = P.load_all(verbose=False)
    ent, bg, conc = {}, {}, {}
    for sid, D in sets.items():
        if sid not in SID2EID:
            continue
        grid = _grid()
        Xg = _resample(np.asarray(D["X"], float), np.asarray(D["axis"], float), grid)
        Xp = _prep(Xg)
        y = np.asarray(D["y"], float)
        eid = SID2EID[sid]
        ent[eid] = Xp[y > 0]
        conc[eid] = y[y > 0]
        mid = SID2MID[sid]
        bg.setdefault(mid, []).append(Xp[y == 0])
    return ent, {k: np.concatenate(v) for k, v in bg.items()}, conc


def load_g1_conc():
    """返回 (entities, conc, matrix_bg, grid)。

    conc[eid] = 每条 query 谱对应的掺混浓度(%)，纯品/认证型实体为 None。
    用于 exp13：zero-shot 可辨识性 vs 掺混浓度（现实检测限）。
    """
    ent1, bg1, c1 = _load_pub_conc()
    ent2, bg2, c2 = _load_honey_conc()
    ent3 = _load_meats()
    ent4 = _load_coffee()
    ent = {**ent1, **ent2, **ent3, **ent4}
    bg = {**bg1, **bg2}
    conc = {**c1, **c2}
    for k in ent3:
        conc[k] = None
    for k in ent4:
        conc[k] = None
    for k, v in ent.items():
        assert np.isfinite(v).all(), f"{k} has NaN"
    return ent, conc, bg, _grid()


def _load_honey():
    """F1 Kaggle 蜂蜜：query=糖浆掺混(10/20/40%)，背景=纯蜂蜜。"""
    z = zipfile.ZipFile(os.path.join(SHARED, "honey_ftir.zip"))
    df = pd.read_csv(io.BytesIO(z.read("ftir.csv")))
    num_cols = [c for c in df.columns[3:] if str(c).replace('.', '', 1).isdigit()]
    ax = np.array([float(c) for c in num_cols])
    order = np.argsort(ax)                      # 轴 4000→650 降序，转升序
    ax = ax[order]
    X = df[num_cols].to_numpy(float)[:, order]
    rate = df["Sugar rate"].astype(float).to_numpy()
    grid = _grid()
    Xp = _prep(_resample(X, ax, grid))
    return dict(HSYRUP=Xp[rate > 0]), dict(honey=Xp[rate == 0])


def _load_honey_conc():
    """同 _load_honey，额外返回 conc: {HSYRUP: 糖浆掺混率(%) 数组}。"""
    z = zipfile.ZipFile(os.path.join(SHARED, "honey_ftir.zip"))
    df = pd.read_csv(io.BytesIO(z.read("ftir.csv")))
    num_cols = [c for c in df.columns[3:] if str(c).replace('.', '', 1).isdigit()]
    ax = np.array([float(c) for c in num_cols])
    order = np.argsort(ax)
    ax = ax[order]
    X = df[num_cols].to_numpy(float)[:, order]
    rate = df["Sugar rate"].astype(float).to_numpy()
    grid = _grid()
    Xp = _prep(_resample(X, ax, grid))
    return dict(HSYRUP=Xp[rate > 0]), dict(honey=Xp[rate == 0]), dict(HSYRUP=rate[rate > 0])


def _load_meats():
    """qib MIRFreshMeats：3 物种纯品肉泥。"""
    df = pd.read_csv(os.path.join(SHARED, "qib/MIRFreshMeats.csv"))
    ax = df["Wavenumbers"].to_numpy(float)
    cols = [c for c in df.columns if c != "Wavenumbers"]
    grid = _grid()
    ent = {}
    for sp in ["FreshChicken", "FreshPork", "FreshTurkey"]:
        cc = [c for c in cols if c.startswith(sp)]
        X = df[cc].to_numpy(float).T
        eid = {"FreshChicken": "MEATCH", "FreshPork": "MEATP", "FreshTurkey": "MEATT"}[sp]
        ent[eid] = _prep(_resample(X, ax, grid))
    return ent


def _load_coffee():
    """quadram 速溶咖啡：Arabica/Robusta 纯品。"""
    df = pd.read_csv(os.path.join(SHARED, "quadram/FTIR_Spectra_instant_coffee.csv"), header=None)
    ax = df.iloc[3:, 0].to_numpy(float)
    sp_row = df.iloc[2, 1:].astype(str).to_numpy()
    X = df.iloc[3:, 1:].to_numpy(float).T
    grid = _grid()
    ent = {}
    for sp, eid in [("Arabica", "COFA"), ("Robusta", "COFR")]:
        Xs = _prep(_resample(X[sp_row == sp], ax, grid))
        ent[eid] = Xs
    return ent


def load_g1():
    """返回 (entities, matrix_bg, grid)。entities: {eid: X(n,B)}；matrix_bg: {mid: X}。"""
    ent1, bg1 = _load_pub()
    ent2, bg2 = _load_honey()
    ent3 = _load_meats()
    ent4 = _load_coffee()
    ent = {**ent1, **ent2, **ent3, **ent4}
    bg = {**bg1, **bg2}
    for k, v in ent.items():
        assert np.isfinite(v).all(), f"{k} has NaN"
    return ent, bg, _grid()


if __name__ == "__main__":
    ent, bg, grid = load_g1()
    print(f"grid: {grid[0]:.1f}–{grid[-1]:.1f} cm⁻¹, {len(grid)} pts")
    print(f"{'eid':8s} {'matrix':10s} {'kind':14s} n")
    for eid, X in ent.items():
        name, mid, kind = ENTITY_META[eid]
        print(f"{eid:8s} {mid:10s} {kind:14s} {len(X):5d}  ({name})")
    print("--- matrix background ---")
    for mid, X in bg.items():
        print(f"{mid:12s} n={len(X)}")
    print("total entities:", len(ent))
