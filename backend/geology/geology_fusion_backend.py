# geology_fusion_backend.py
import pandas as pd
from pathlib import Path
from geology.fusion import annotate_unique_chainage

from config import EVIDENCE_DB_PATH
DEFAULT_EVIDENCE_DB_PATH = EVIDENCE_DB_PATH

def load_evidence_db(path=DEFAULT_EVIDENCE_DB_PATH):
    """
    读取证据库
    """
    df = pd.read_csv(path)
    return df


def _ensure_chainage_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保存在统一的 chainage 字段
    优先使用：导向盾首里程，其次开累进尺
    """
    out = df.copy()

    if "chainage" in out.columns:
        out["chainage"] = pd.to_numeric(out["chainage"], errors="coerce")
        return out

    if "导向盾首里程" in out.columns:
        out["chainage"] = pd.to_numeric(out["导向盾首里程"], errors="coerce")
        return out

    if "开累进尺" in out.columns:
        out["chainage"] = pd.to_numeric(out["开累进尺"], errors="coerce")
        return out

    return out


def attach_geology_labels(df_plc: pd.DataFrame, evidence_df: pd.DataFrame):
    """
    给 PLC 数据挂接地质融合标签
    """
    df = _ensure_chainage_column(df_plc)

    if "chainage" not in df.columns:
        print("未找到可用里程字段，返回原始数据。")
        return df

    df = df.dropna(subset=["chainage"]).copy()

    unique_chainage = (
        df[["chainage"]]
        .drop_duplicates()
        .sort_values("chainage")
        .reset_index(drop=True)
    )

    anno_unique = annotate_unique_chainage(unique_chainage, evidence_df)

    df = df.merge(anno_unique, on="chainage", how="left")
    return df