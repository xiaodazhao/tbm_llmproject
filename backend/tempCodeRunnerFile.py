# test_fusion.py
import pandas as pd
from pathlib import Path

from geology.geology_fusion_backend import load_evidence_db, attach_geology_labels
from config import EVIDENCE_DB_PATH, DATA_DIR

date_str = "20231230"
plc_path = DATA_DIR / f"tbm_data_{date_str}.csv"

df_plc = pd.read_csv(plc_path)
evidence_df = load_evidence_db(EVIDENCE_DB_PATH)

df_geo = attach_geology_labels(df_plc, evidence_df)

cols = [
    "chainage",
    "risk",
    "risk_score",
    "hazard",
    "fused_grade",
    "active_source_count",
    "water_risk_score",
    "collapse_risk_score",
    "rockmass_risk_score",
    "grade_risk_score",
    "detail_score_mean",
    "detail_score_max",
]
cols = [c for c in cols if c in df_geo.columns]

print(df_geo[cols].head(20).to_string())