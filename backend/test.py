import pandas as pd
from geology.geology_fusion_backend import load_evidence_db, attach_geology_labels, _ensure_chainage_column
from config import EVIDENCE_DB_PATH, DATA_DIR

date_str = "20230915"
plc_path = DATA_DIR / f"tbm_data_{date_str}.csv"

df_plc = pd.read_csv(plc_path)
df_plc2 = _ensure_chainage_column(df_plc)

evidence_df = load_evidence_db(EVIDENCE_DB_PATH)
df_geo = attach_geology_labels(df_plc, evidence_df)

print("PLC原始列名：", df_plc.columns.tolist())
print("PLC统一后chainage范围：", df_plc2["chainage"].min(), df_plc2["chainage"].max())
print("Evidence总范围：", evidence_df["start_num"].min(), evidence_df["end_num"].max())

tsp_df = evidence_df[evidence_df["source_type"] == "tsp"].copy()
print("\nTSP覆盖区间：")
print(
    tsp_df[["report_id", "start_num", "end_num"]]
    .drop_duplicates()
    .sort_values(["start_num", "end_num"])
    .to_string(index=False)
)

print("\n融合后前20行：")
cols = [
    "chainage", "risk", "risk_score", "hazard",
    "fused_grade", "active_source_count",
    "water_risk_score", "collapse_risk_score",
    "rockmass_risk_score", "grade_risk_score",
    "detail_score_mean", "detail_score_max"
]
cols = [c for c in cols if c in df_geo.columns]
print(df_geo[cols].head(20).to_string())