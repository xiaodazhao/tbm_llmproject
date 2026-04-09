# geology_summary.py
import pandas as pd


def summarize_geology_record_level(df: pd.DataFrame):
    """
    记录级摘要：保留你原来的逻辑，但用于辅助说明
    """
    result = {}

    if "active_source_count" not in df.columns:
        return {
            "has_geology": False,
            "summary_text": "未进行地质融合分析。"
        }

    result["has_geology"] = True
    result["sample_count"] = len(df)

    coverage_counts = df["coverage"].value_counts(dropna=False).to_dict() if "coverage" in df.columns else {}
    risk_counts = df["risk"].value_counts(dropna=False).to_dict() if "risk" in df.columns else {}
    hazard_counts = df["hazard"].value_counts(dropna=False).head(5).to_dict() if "hazard" in df.columns else {}

    result["coverage_counts"] = coverage_counts
    result["risk_counts"] = risk_counts
    result["hazard_counts"] = hazard_counts

    result["max_attention"] = int(df["active_source_count"].max()) if "active_source_count" in df.columns else None
    result["mean_attention"] = float(df["active_source_count"].mean()) if "active_source_count" in df.columns else None

    high_attention_df = df[df["active_source_count"] >= 4] if "active_source_count" in df.columns else pd.DataFrame()
    result["high_attention_count"] = int(len(high_attention_df))

    lines = []
    lines.append(f"本时段共匹配到 {len(df)} 条带地质标签的PLC记录。")

    if coverage_counts:
        lines.append(f"地质覆盖情况：{', '.join([f'{k}={v}' for k, v in coverage_counts.items()])}。")

    if risk_counts:
        lines.append(f"风险分布情况：{', '.join([f'{k}={v}' for k, v in risk_counts.items()])}。")

    if result["max_attention"] is not None:
        lines.append(
            f"多源报告最大命中数为 {result['max_attention']}，平均命中数为 {result['mean_attention']:.2f}。"
        )

    if result["high_attention_count"] > 0:
        lines.append(f"active_source_count≥4 的较高关注记录共有 {result['high_attention_count']} 条。")

    if hazard_counts:
        top_hazard_text = "，".join([f"{k}({v})" for k, v in hazard_counts.items()])
        lines.append(f"主要灾害表现为：{top_hazard_text}。")

    result["summary_text"] = "\n".join(lines)
    return result


def summarize_geology_segment_level(segment_df: pd.DataFrame):
    """
    区段级摘要：这是更推荐给报告使用的版本
    """
    if segment_df is None or len(segment_df) == 0:
        return {
            "has_geology": False,
            "summary_text": "未形成区段级地质融合分析结果。"
        }

    result = {
        "has_geology": True,
        "segment_count": int(len(segment_df))
    }

    risk_dist = segment_df["risk_mode"].value_counts(dropna=False).to_dict() if "risk_mode" in segment_df.columns else {}
    interpretation_dist = (
        segment_df["interpretation"].value_counts(dropna=False).to_dict()
        if "interpretation" in segment_df.columns else {}
    )

    result["risk_dist"] = risk_dist
    result["interpretation_dist"] = interpretation_dist

    high_risk_df = segment_df[segment_df["risk_score_max"] >= 3] if "risk_score_max" in segment_df.columns else pd.DataFrame()
    multi_source_df = segment_df[segment_df["active_source_count_max"] >= 3] if "active_source_count_max" in segment_df.columns else pd.DataFrame()

    result["high_risk_segment_count"] = int(len(high_risk_df))
    result["multi_source_segment_count"] = int(len(multi_source_df))

    lines = []
    lines.append(f"本次区段级分析共识别 {len(segment_df)} 个施工区段。")

    if risk_dist:
        lines.append(
            f"区段风险分布为：{', '.join([f'{k}={v}' for k, v in risk_dist.items()])}。"
        )

    if len(high_risk_df) > 0:
        lines.append(f"其中高风险区段共 {len(high_risk_df)} 个。")

    if len(multi_source_df) > 0:
        lines.append(f"多源共同关注区段共 {len(multi_source_df)} 个。")

    if interpretation_dist:
        lines.append(
            f"施工响应判读结果：{', '.join([f'{k}={v}' for k, v in interpretation_dist.items()])}。"
        )

    # 典型区段
    if "segment" in segment_df.columns and "interpretation" in segment_df.columns:
        typical = segment_df.head(3)
        typical_lines = []
        for _, row in typical.iterrows():
            seg = row.get("segment", "")
            interp = row.get("interpretation", "")
            risk = row.get("risk_mode", "")
            typical_lines.append(f"{seg}（{risk}，{interp}）")
        if typical_lines:
            lines.append("典型区段包括：" + "；".join(typical_lines) + "。")

    result["summary_text"] = "\n".join(lines)
    return result


def geology_summary_to_text(geo_summary: dict):
    if not geo_summary or not geo_summary.get("has_geology", False):
        return "本时段未进行地质融合分析。"
    return geo_summary.get("summary_text", "本时段已完成地质融合分析。")