# test_fusion_effect.py
from pathlib import Path
import pandas as pd

from geology.geology_fusion_backend import (
    load_evidence_db,
    attach_geology_labels,
    _ensure_chainage_column,
)
from config import EVIDENCE_DB_PATH


def print_basic_info(df_plc: pd.DataFrame, evidence_df: pd.DataFrame, df_geo: pd.DataFrame):
    print("=" * 100)
    print("基础信息")
    print("=" * 100)

    print(f"PLC 行数: {len(df_plc)}")
    print(f"Evidence 行数: {len(evidence_df)}")
    print(f"融合后行数: {len(df_geo)}")
    print()

    if "chainage" in df_plc.columns:
        print("PLC chainage 范围:",
              df_plc["chainage"].min(), "~", df_plc["chainage"].max())

    if "start_num" in evidence_df.columns and "end_num" in evidence_df.columns:
        print("Evidence 覆盖范围:",
              evidence_df["start_num"].min(), "~", evidence_df["end_num"].max())
    print()

    if "source_type" in evidence_df.columns:
        print("Evidence source_type 分布：")
        print(evidence_df["source_type"].value_counts(dropna=False).to_string())
        print()

    if "source_level" in evidence_df.columns:
        print("Evidence source_level 分布：")
        print(evidence_df["source_level"].value_counts(dropna=False).to_string())
        print()


def print_fusion_summary(df_geo: pd.DataFrame):
    print("=" * 100)
    print("融合结果概览")
    print("=" * 100)

    if "risk" in df_geo.columns:
        print("risk 分布：")
        print(df_geo["risk"].value_counts(dropna=False).to_string())
        print()

    if "active_source_count" in df_geo.columns:
        print("active_source_count 分布：")
        print(df_geo["active_source_count"].value_counts(dropna=False).sort_index().to_string())
        print()

        hit_df = df_geo[df_geo["active_source_count"] > 0].copy()
        print(f"命中至少1个地质证据的 PLC 记录数: {len(hit_df)} / {len(df_geo)}")
        if len(df_geo) > 0:
            print(f"命中比例: {len(hit_df) / len(df_geo):.2%}")
        print()

    if "hazard" in df_geo.columns:
        print("hazard Top10：")
        print(df_geo["hazard"].value_counts(dropna=False).head(10).to_string())
        print()

    if "fused_grade" in df_geo.columns:
        print("fused_grade 分布：")
        print(df_geo["fused_grade"].value_counts(dropna=False).to_string())
        print()


def print_hit_examples(df_geo: pd.DataFrame, top_n: int = 20):
    print("=" * 100)
    print("命中地质证据的典型记录")
    print("=" * 100)

    if "active_source_count" not in df_geo.columns:
        print("没有 active_source_count 列。")
        return

    hit_df = df_geo[df_geo["active_source_count"] > 0].copy()
    if hit_df.empty:
        print("当前 PLC 数据没有命中任何地质证据。")
        return

    cols = [
        "chainage",
        "risk",
        "risk_score",
        "hazard",
        "fused_grade",
        "active_source_count",
        "active_sources",
        "active_report_ids",
        "water_risk_score",
        "collapse_risk_score",
        "rockmass_risk_score",
        "grade_risk_score",
        "detail_score_mean",
        "detail_score_max",
    ]
    cols = [c for c in cols if c in hit_df.columns]

    # 按风险和命中数排序
    sort_cols = [c for c in ["risk_score", "active_source_count", "detail_score_max"] if c in hit_df.columns]
    ascending = [False] * len(sort_cols)

    if sort_cols:
        hit_df = hit_df.sort_values(sort_cols, ascending=ascending)

    print(hit_df[cols].head(top_n).to_string(index=False))
    print()


def print_segment_like_view(df_geo: pd.DataFrame):
    """
    不依赖 segment_analysis，先做一个简化版里程聚合，方便快速看效果
    """
    print("=" * 100)
    print("按里程聚合后的典型区段（简化版）")
    print("=" * 100)

    if "chainage" not in df_geo.columns:
        print("没有 chainage 列。")
        return

    tmp = df_geo.copy()
    tmp["segment_10m"] = (tmp["chainage"] // 10 * 10).astype("Int64")

    agg_dict = {}
    for col in ["risk_score", "active_source_count", "detail_score_mean", "detail_score_max"]:
        if col in tmp.columns:
            agg_dict[col] = ["mean", "max"]

    for col in ["推进速度", "推力", "刀盘扭矩"]:
        if col in tmp.columns:
            agg_dict[col] = ["mean", "max"]

    if not agg_dict:
        print("没有足够列可聚合。")
        return

    seg = tmp.groupby("segment_10m").agg(agg_dict)
    seg.columns = ["_".join([str(x) for x in c if x]) for c in seg.columns]
    seg = seg.reset_index()

    sort_cols = [c for c in [
        "risk_score_max",
        "active_source_count_max",
        "detail_score_max_max",
        "detail_score_mean_max",
    ] if c in seg.columns]

    if sort_cols:
        seg = seg.sort_values(sort_cols, ascending=False)

    print(seg.head(20).to_string(index=False))
    print()


def print_source_hit_statistics(df_geo: pd.DataFrame):
    print("=" * 100)
    print("来源命中情况")
    print("=" * 100)

    if "active_sources" not in df_geo.columns:
        print("没有 active_sources 列。")
        return

    hit_df = df_geo[df_geo["active_source_count"] > 0].copy()
    if hit_df.empty:
        print("没有任何命中来源。")
        return

    source_counter = {}

    for x in hit_df["active_sources"].dropna():
        parts = [p.strip() for p in str(x).split(";") if p.strip()]
        for p in parts:
            source_counter[p] = source_counter.get(p, 0) + 1

    if not source_counter:
        print("active_sources 为空。")
        return

    out = pd.Series(source_counter).sort_values(ascending=False)
    print(out.to_string())
    print()


if __name__ == "__main__":
    # ===== 1. 改成你的 PLC 文件路径 =====
    plc_path = Path(
        r"C:\Users\22923\Desktop\伯舒拉岭_plc_超报数据\数据\tbm9伯舒拉岭右线\伯舒拉岭TBM_合并后.csv"
    )

    # ===== 2. 读 PLC =====
    df_plc = pd.read_csv(plc_path)

    # 统一 chainage，方便先看范围
    df_plc = _ensure_chainage_column(df_plc)

    # ===== 3. 读 evidence_db =====
    evidence_df = load_evidence_db(EVIDENCE_DB_PATH)

    # ===== 4. 地质融合 =====
    df_geo = attach_geology_labels(df_plc, evidence_df)

    # ===== 5. 输出检查 =====
    print_basic_info(df_plc, evidence_df, df_geo)
    print_fusion_summary(df_geo)
    print_source_hit_statistics(df_geo)
    print_hit_examples(df_geo, top_n=20)
    print_segment_like_view(df_geo)