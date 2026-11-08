"""exp03a —— 自采三套预处理可行性探测。

判断 S1/S2/S3 的负值与量纲问题是『格式/预处理缺失』（可救）还是『采集硬伤』（需排除）。
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, r"D:/2026sci论文实验设计/论文6/scripts")
import p6_data as P

OUT = r"D:/2026sci论文实验设计/论文6/out"


def snv(X):
    mu = X.mean(1, keepdims=True); sd = np.std(X, 1, keepdims=True)
    return (X - mu) / (sd + 1e-9)


def main():
    sets = P.load_all(verbose=False)
    rows = []
    for sid in ["S1", "S2", "S3"]:
        X = sets[sid]["X"].astype(float)
        raw_neg = float((X < 0).mean())
        raw_in01 = float(((X >= 0) & (X <= 1)).mean())
        Xs = snv(X)
        snv_neg = float((Xs < 0).mean())              # SNV 后必有一半负（均值0）→ 仅看是否合理
        # 判定：若原始值大量落在 [0, ~1.5] 且仅有少量负（基线偏移）→ 属基线问题，SNV/基线校正可救
        # 若值跨正负且分布离散（如 S3 到 5万）→ 属原始光强，需参考谱才能转吸光度
        verdict = []
        if sid == "S3":
            verdict.append("原始光强(含负、量级达万)→ 无参考谱无法转吸光度，格式硬伤")
        else:
            if raw_in01 > 0.7 and raw_neg < 0.5:
                verdict.append("多为[0,1]基线偏移 → SNV/基线校正可救，非采集硬伤")
            else:
                verdict.append(f"负值{raw_neg:.0%}偏高 → 需基线校正后重评")
        rows.append(dict(src=sid, raw_min=round(float(X.min()),3), raw_max=round(float(X.max()),3),
                         raw_neg=round(raw_neg,4), raw_in01=round(raw_in01,4),
                         snv_min=round(float(Xs.min()),3), snv_max=round(float(Xs.max()),3),
                         verdict="；".join(verdict)))
    for r in rows:
        print(r)
    with open(os.path.join(OUT, "exp03a_p6_preproc_probe.md"), "w", encoding="utf-8") as f:
        f.write("# exp03a 自采三套预处理可行性探测\n\n")
        f.write("| 集 | raw_min | raw_max | 负值% | 在[0,1]% | SNV_min | SNV_max | 判定 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['src']} | {r['raw_min']} | {r['raw_max']} | {r['raw_neg']} | "
                     f"{r['raw_in01']} | {r['snv_min']} | {r['snv_max']} | {r['verdict']} |\n")
    print("[written] exp03a_p6_preproc_probe.md")


if __name__ == "__main__":
    main()
