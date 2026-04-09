import json
import pandas as pd


RISK_SCORE_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3
}


def _safe_load_attrs(x):
    try:
        if pd.isna(x):
            return {}
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _format_dk(chainage):
    """
    将数值里程格式化为 DKxxxx+xxx
    假设 chainage 形式类似 1013.363 -> DK1013+363
    """
    if pd.isna(chainage):
        return "未知里程"

    value = float(chainage)
    km = int(value)
    m = int(round((value - km) * 1000))
    return f"DK{km}+{m:03d}"


def _normalize_hazard(attrs):
    hazards = []
    if attrs.get("water_flag"):
        hazards.append("出水")
    if attrs.get("collapse_flag"):
        hazards.append("掉块")
    if attrs.get("deformation_flag"):
        hazards.append("变形")
    return hazards


def _summarize_forward_evidence(forward_df):
    """
    对前方命中的证据做简要融合统计
    """
    if forward_df.empty:
        return {
            "forward_segment_count": 0,
            "high_risk_count": 0,
            "multi_source_count": 0,
            "risk_level_dist": {},
            "main_hazards": [],
            "source_types": [],
            "covered_start": None,
            "covered_end": None,
        }

    temp = forward_df.copy()

    # 解析 attrs_json
    temp["attrs_obj"] = temp["attrs_json"].apply(_safe_load_attrs)

    # 风险等级
    temp["risk_level"] = temp["attrs_obj"].apply(lambda x: x.get("risk_level", "low"))
    temp["risk_score"] = temp["risk_level"].map(RISK_SCORE_MAP).fillna(1)

    # hazard
    temp["hazards"] = temp["attrs_obj"].apply(_normalize_hazard)

    # 按证据段统计
    risk_level_dist = temp["risk_level"].value_counts().to_dict()
    high_risk_count = int((temp["risk_score"] >= 3).sum())

    # source_type 统计
    source_types = sorted(set(temp["source_type"].astype(str).dropna().tolist()))

    # multi-source：这里按 source_type + report_id 去重统计
    dedup = (
        temp[["source_type", "report_id"]]
        .astype(str)
        .drop_duplicates()
    )
    multi_source_count = len(dedup)

    # hazards 汇总
    hazard_counter = {}
    for hs in temp["hazards"]:
        for h in hs:
            hazard_counter[h] = hazard_counter.get(h, 0) + 1

    main_hazards = [
        k for k, _ in sorted(hazard_counter.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "forward_segment_count": int(len(temp)),
        "high_risk_count": high_risk_count,
        "multi_source_count": int(multi_source_count),
        "risk_level_dist": risk_level_dist,
        "main_hazards": main_hazards,
        "source_types": source_types,
        "covered_start": float(temp["start_num"].min()) if "start_num" in temp.columns else None,
        "covered_end": float(temp["end_num"].max()) if "end_num" in temp.columns else None,
    }


def generate_forward_risk_summary(df_plc: pd.DataFrame, evidence_df: pd.DataFrame, lookahead_m=30):
    """
    基于当天 PLC 最后位置，提取“前方窗口”的超报/地质证据，并生成结构化摘要

    参数：
    - df_plc: 当天 PLC 数据
    - evidence_df: 证据库（evidence_db.csv 读入后的 DataFrame）
    - lookahead_m: 向前看的距离，单位与 chainage 同体系

    返回 dict：
    {
        "has_forward_risk": bool,
        "current_chainage": ...,
        "current_chainage_dk": ...,
        "lookahead_m": ...,
        "forward_start": ...,
        "forward_end": ...,
        "forward_start_dk": ...,
        "forward_end_dk": ...,
        "forward_segment_count": ...,
        "high_risk_count": ...,
        "multi_source_count": ...,
        "risk_level_dist": {...},
        "main_hazards": [...],
        "source_types": [...],
        "advice_level": "...",
        "advice_text": "..."
    }
    """
    if df_plc is None or len(df_plc) == 0:
        return {
            "has_forward_risk": False,
            "message": "PLC 数据为空，无法生成前方风险提示。"
        }

    df = df_plc.copy()

    # 确定 chainage
    if "chainage" in df.columns:
        df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    elif "导向盾首里程" in df.columns:
        df["chainage"] = pd.to_numeric(df["导向盾首里程"], errors="coerce")
    elif "开累进尺" in df.columns:
        df["chainage"] = pd.to_numeric(df["开累进尺"], errors="coerce")
    else:
        return {
            "has_forward_risk": False,
            "message": "PLC 数据中缺少可用里程字段，无法生成前方风险提示。"
        }

    df = df.dropna(subset=["chainage"]).copy()
    if df.empty:
        return {
            "has_forward_risk": False,
            "message": "PLC 里程字段为空，无法生成前方风险提示。"
        }

    # 当前里程：当天最后一条 PLC 记录的里程
    current_chainage = float(df["chainage"].iloc[-1])
    forward_start = current_chainage
    forward_end = current_chainage + float(lookahead_m)

    # 提取前方证据：与 [forward_start, forward_end] 有交集的证据段
    ev = evidence_df.copy()
    ev["start_num"] = pd.to_numeric(ev["start_num"], errors="coerce")
    ev["end_num"] = pd.to_numeric(ev["end_num"], errors="coerce")
    ev = ev.dropna(subset=["start_num", "end_num"]).copy()

    forward_df = ev[
        (ev["end_num"] >= forward_start) &
        (ev["start_num"] <= forward_end)
    ].copy()

    stat = _summarize_forward_evidence(forward_df)

    # 给一个温和的建议级别
    if stat["high_risk_count"] > 0 and stat["multi_source_count"] >= 3:
        advice_level = "high"
    elif stat["high_risk_count"] > 0 or stat["multi_source_count"] >= 2:
        advice_level = "medium"
    elif stat["forward_segment_count"] > 0:
        advice_level = "low"
    else:
        advice_level = "none"

    # 生成简短建议文本
    if stat["forward_segment_count"] == 0:
        advice_text = (
            f"截至当前掘进位置 { _format_dk(current_chainage) }，前方 {lookahead_m} m 范围内"
            "未识别到明显风险提示，建议保持常规施工监测。"
        )
    else:
        hazard_text = "、".join(stat["main_hazards"][:3]) if stat["main_hazards"] else "未见明确灾害类型"
        risk_dist = stat["risk_level_dist"]

        if advice_level == "high":
            advice_text = (
                f"截至当前掘进位置 { _format_dk(current_chainage) }，前方 {lookahead_m} m 范围内"
                f"识别出 {stat['forward_segment_count']} 个风险提示段，其中高风险段 {stat['high_risk_count']} 个，"
                f"多源共同关注程度较高。主要风险表现为 {hazard_text}，"
                "建议后续施工过程中加强现场监测，并重点关注推进速度、推力及刀盘扭矩变化。"
            )
        elif advice_level == "medium":
            advice_text = (
                f"截至当前掘进位置 { _format_dk(current_chainage) }，前方 {lookahead_m} m 范围内"
                f"存在一定风险提示，共识别 {stat['forward_segment_count']} 个相关区段，"
                f"主要风险表现为 {hazard_text}。"
                "建议后续施工中持续跟踪施工参数变化，并加强对前方区段的关注。"
            )
        else:
            advice_text = (
                f"截至当前掘进位置 { _format_dk(current_chainage) }，前方 {lookahead_m} m 范围内"
                f"识别到 {stat['forward_segment_count']} 个一般风险提示段，"
                "整体风险程度相对有限，建议保持常规监测并关注后续变化。"
            )

    return {
        "has_forward_risk": True,
        "current_chainage": current_chainage,
        "current_chainage_dk": _format_dk(current_chainage),
        "lookahead_m": float(lookahead_m),
        "forward_start": forward_start,
        "forward_end": forward_end,
        "forward_start_dk": _format_dk(forward_start),
        "forward_end_dk": _format_dk(forward_end),
        "forward_segment_count": stat["forward_segment_count"],
        "high_risk_count": stat["high_risk_count"],
        "multi_source_count": stat["multi_source_count"],
        "risk_level_dist": stat["risk_level_dist"],
        "main_hazards": stat["main_hazards"],
        "source_types": stat["source_types"],
        "advice_level": advice_level,
        "advice_text": advice_text,
    }


def forward_risk_to_text(summary: dict):
    """
    结构化前方风险摘要 -> 文本
    """
    if not summary or not summary.get("has_forward_risk", False):
        return "未形成可用的前方风险提示。"

    if summary.get("forward_segment_count", 0) == 0:
        return summary.get(
            "advice_text",
            "当前前方区段未识别到明显风险提示。"
        )

    hazards = summary.get("main_hazards", [])
    hazard_text = "、".join(hazards[:3]) if hazards else "未见明确灾害类型"

    lines = [
        f"截至当前掘进位置 {summary.get('current_chainage_dk', '未知里程')}，"
        f"前方 {int(summary.get('lookahead_m', 0))} m 范围为 "
        f"{summary.get('forward_start_dk', '')} ~ {summary.get('forward_end_dk', '')}。",

        f"该范围内共识别 {summary.get('forward_segment_count', 0)} 个风险提示段，"
        f"其中高风险段 {summary.get('high_risk_count', 0)} 个，"
        f"多源共同关注程度为 {summary.get('multi_source_count', 0)}。",

        f"主要风险类型表现为：{hazard_text}。",

        summary.get("advice_text", "建议后续施工过程中保持关注。")
    ]

    return "\n".join(lines)