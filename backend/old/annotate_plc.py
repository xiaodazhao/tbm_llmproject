#annotate_plc.py
import pandas as pd
from pathlib import Path

from config import DATA_DIR, EVIDENCE_DB_PATH, RESULT_DIR

plc_path = DATA_DIR / "伯舒拉岭TBM_合并后.csv"
evidence_path = EVIDENCE_DB_PATH


def main():
    # ===== 路径 =====
    plc_path = Path(r"/Users/zhaoxiaoda/Desktop/伯舒拉岭_plc_超报数据/数据/tbm9伯舒拉岭右线/伯舒拉岭TBM_合并后.csv")
    evidence_path = DB_DIR / "evidence_db.csv"

    out_path = RESULT_DIR / "plc_annotated.csv"
    out_unique_path = RESULT_DIR / "chainage_annotated_unique.csv"

    # ===== 读取 =====
    plc = pd.read_csv(plc_path)
    evidence = pd.read_csv(evidence_path)

    print("原始PLC数据量：", len(plc))
    print("PLC列名：", plc.columns.tolist())

    # ===== 只保留掘进状态=1 =====
    plc = plc[plc["掘进状态"] == 1].copy()
    print("掘进状态=1 后数据量：", len(plc))

    # ===== 统一里程列 =====
    plc["chainage"] = pd.to_numeric(plc["导向盾首里程"], errors="coerce")
    plc = plc.dropna(subset=["chainage"]).copy()
    print("有效里程数据量：", len(plc))

    # ===== 唯一里程匹配 =====
    unique_chainage = (
        plc[["chainage"]]
        .drop_duplicates()
        .sort_values("chainage")
        .reset_index(drop=True)
    )
    print("唯一里程数：", len(unique_chainage))

    # ===== 只对唯一里程做融合 =====
    anno_unique = annotate_unique_chainage(unique_chainage, evidence)

    # 保存唯一里程结果
    anno_unique.to_csv(out_unique_path, index=False, encoding="utf-8-sig")
    print(f"唯一里程融合结果已保存: {out_unique_path}")

    # ===== 回填到原始PLC =====
    out = plc.merge(anno_unique, on="chainage", how="left")

    # 保存完整结果
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"完整PLC打标结果已保存: {out_path}")

    # ===== 简单统计 =====
    print("\n===== 覆盖统计 =====")
    print(out["coverage"].value_counts(dropna=False))

    print("\n===== 风险统计 =====")
    print(out["risk"].value_counts(dropna=False))

    print("\n===== 示例输出 =====")
    show_cols = [
        "chainage", "coverage", "risk", "hazard",
        "active_source_count", "active_sources",
        "fused_grade", "uncertainty"
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    print(out[show_cols].head(10))


if __name__ == "__main__":
    main()