# geology_fusion_backend.py
import json
import pandas as pd
from geology.fusion import annotate_unique_chainage

from config import EVIDENCE_DB_PATH

DEFAULT_EVIDENCE_DB_PATH = EVIDENCE_DB_PATH


def _safe_load_attrs(x):
    try:
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_evidence_db(path=DEFAULT_EVIDENCE_DB_PATH):
    """
    读取证据库，并展开常用 attrs_json 字段
    """
    df = pd.read_csv(path)

    if "attrs_json" in df.columns:
        df["attrs_obj"] = df["attrs_json"].apply(_safe_load_attrs)

        # 常用字段展开，便于调试或后续分析
        df["risk_level"] = df["attrs_obj"].apply(lambda x: x.get("risk_level"))
        df["water_flag"] = df["attrs_obj"].apply(lambda x: x.get("water_flag", 0))
        df["collapse_flag"] = df["attrs_obj"].apply(lambda x: x.get("collapse_flag", 0))
        df["deformation_flag"] = df["attrs_obj"].apply(lambda x: x.get("deformation_flag", 0))
        df["support_grade"] = df["attrs_obj"].apply(
            lambda x: x.get("support_grade") or x.get("rock_grade")
        )
        df["water_type"] = df["attrs_obj"].apply(lambda x: x.get("water_type"))
        df["risk_tags"] = df["attrs_obj"].apply(lambda x: x.get("risk_tags", []))

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

    if evidence_df is None or len(evidence_df) == 0:
        print("证据库为空，返回原始数据。")
        return df

    df = df.dropna(subset=["chainage"]).copy()
    if df.empty:
        print("PLC 里程为空，返回原始数据。")
        return df

    # 保留有效证据层级
    if "source_level" in evidence_df.columns:
        evidence_df = evidence_df[
            evidence_df["source_level"].isin(["segment", "report_conclusion"])
        ].copy()

    unique_chainage = (
        df[["chainage"]]
        .drop_duplicates()
        .sort_values("chainage")
        .reset_index(drop=True)
    )

    anno_unique = annotate_unique_chainage(unique_chainage, evidence_df)

    df = df.merge(anno_unique, on="chainage", how="left")
    return df