# segment_analysis.py
import pandas as pd
import numpy as np


def build_segments(df: pd.DataFrame, segment_length=10):
    """
    按固定里程长度划分区段
    例如 segment_length=10 表示每10m一个区段
    """
    out = df.copy()

    if "chainage" not in out.columns:
        raise ValueError("输入数据缺少 chainage 字段")

    out = out.dropna(subset=["chainage"]).copy()
    out["chainage"] = pd.to_numeric(out["chainage"], errors="coerce")
    out = out.dropna(subset=["chainage"]).copy()

    out["segment_start"] = (np.floor(out["chainage"] / segment_length) * segment_length).astype(float)
    out["segment_end"] = out["segment_start"] + segment_length
    out["segment_id"] = out["segment_start"].astype(str) + "_" + out["segment_end"].astype(str)

    return out


def _pick_existing_columns(df: pd.DataFrame, candidate_cols):
    return [c for c in candidate_cols if c in df.columns]


def aggregate_segments(df: pd.DataFrame):
    """
    对每个区段做聚合统计
    """
    if "segment_id" not in df.columns:
        raise ValueError("请先调用 build_segments()")

    out = df.copy()

    numeric_candidates = [
        "推进速度",
        "推进给定速度",
        "推力",
        "刀盘扭矩",
        "贯入度",
        "推进伸出压力",
        "推进回收压力",
        "主皮带反转压力",
        "主皮带正转压力",
        "risk_score",
        "geo_prior_score",
        "active_source_count",
        "water_flag_fused",
        "collapse_flag_fused",
        "deformation_flag_fused",
    ]

    existing_numeric = _pick_existing_columns(out, numeric_candidates)

    agg_dict = {
        "chainage": ["min", "max", "mean", "count"],
        "segment_start": "first",
        "segment_end": "first",
    }

    for col in existing_numeric:
        if col in ["risk_score", "geo_prior_score", "active_source_count",
                   "water_flag_fused", "collapse_flag_fused", "deformation_flag_fused"]:
            agg_dict[col] = ["max", "mean"]
        else:
            agg_dict[col] = ["mean", "std", "min", "max"]

    if "risk" in out.columns:
        agg_dict["risk"] = lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    if "hazard" in out.columns:
        agg_dict["hazard"] = lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    if "coverage" in out.columns:
        agg_dict["coverage"] = lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    if "uncertainty" in out.columns:
        agg_dict["uncertainty"] = lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    if "fused_grade" in out.columns:
        agg_dict["fused_grade"] = lambda x: x.mode().iloc[0] if not x.mode().empty else ""

    grouped = out.groupby("segment_id").agg(agg_dict)

    new_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)

    grouped.columns = new_cols
    grouped = grouped.reset_index()

    return grouped

def _format_dk(x):
    """
    把数值里程格式化为 DKxxx+yyy
    例如 1013.363 -> DK1013+363
    """
    x = float(x)
    km = int(x)
    meter = int(round((x - km) * 1000))
    return f"DK{km}+{meter:03d}"


def format_segment_label(df: pd.DataFrame):
    """
    增加 segment 文本标签
    """
    out = df.copy()

    out["segment"] = out.apply(
        lambda r: f"{_format_dk(r['segment_start_first'])}~{_format_dk(r['segment_end_first'])}",
        axis=1
    )

    return out


def add_efficiency_indicator(df: pd.DataFrame):
    """
    增加推进效率指标：推进速度 / 推进给定速度
    """
    out = df.copy()

    if "推进速度_mean" in out.columns and "推进给定速度_mean" in out.columns:
        denom = out["推进给定速度_mean"].replace(0, np.nan)
        out["efficiency"] = out["推进速度_mean"] / denom
    else:
        out["efficiency"] = np.nan

    return out


def analyze_segment_response(df: pd.DataFrame):
    """
    基于区段统计结果，判断地质风险与施工响应关系
    """
    out = df.copy()

    speed_col = "推进速度_mean" if "推进速度_mean" in out.columns else None
    thrust_col = "推力_mean" if "推力_mean" in out.columns else None
    torque_col = "刀盘扭矩_mean" if "刀盘扭矩_mean" in out.columns else None

    if "risk_score_max" in out.columns:
        risk_col = "risk_score_max"
    elif "geo_prior_score_max" in out.columns:
        risk_col = "geo_prior_score_max"
    elif "geo_prior_score_mean" in out.columns:
        risk_col = "geo_prior_score_mean"
    else:
        risk_col = None

    source_col = "active_source_count_max" if "active_source_count_max" in out.columns else None

    if speed_col is None or risk_col is None:
        out["interpretation"] = "当前区段级施工响应关联条件仍不充分，判读结果需谨慎解释"
        return out

    global_speed = out[speed_col].mean(skipna=True)
    global_thrust = out[thrust_col].mean(skipna=True) if thrust_col else np.nan
    global_torque = out[torque_col].mean(skipna=True) if torque_col else np.nan

    interpretations = []

    for _, row in out.iterrows():
        risk_score = row.get(risk_col, np.nan)
        speed = row.get(speed_col, np.nan)
        thrust = row.get(thrust_col, np.nan) if thrust_col else np.nan
        torque = row.get(torque_col, np.nan) if torque_col else np.nan
        source_count = row.get(source_col, np.nan) if source_col else np.nan

        speed_low = pd.notna(speed) and pd.notna(global_speed) and speed < global_speed * 0.90
        thrust_high = pd.notna(thrust) and pd.notna(global_thrust) and thrust > global_thrust
        torque_high = pd.notna(torque) and pd.notna(global_torque) and torque > global_torque
        multi_source = pd.notna(source_count) and source_count >= 3

        if risk_score >= 3:
            if speed_low and (thrust_high or torque_high):
                if multi_source:
                    interpretations.append("高风险多源关注区，且施工扰动显著")
                else:
                    interpretations.append("高风险区，施工扰动显著")
            elif speed_low:
                interpretations.append("高风险区，推进速度下降")
            else:
                interpretations.append("高风险区，但施工参数未显著恶化")

        elif risk_score >= 2:
            if speed_low and (thrust_high or torque_high):
                interpretations.append("中等风险区，对施工有一定影响")
            elif speed_low:
                interpretations.append("中等风险区，推进略受影响")
            else:
                interpretations.append("中等风险区，施工总体稳定")

        else:
            if speed_low and (thrust_high or torque_high):
                interpretations.append("低风险标签下出现施工异常，建议复核")
            else:
                interpretations.append("低风险区段，施工正常")

    out["interpretation"] = interpretations
    return out


def add_relative_change_features(df: pd.DataFrame):
    """
    增加相对全局均值的变化率，便于后续写报告
    """
    out = df.copy()

    for col in ["推进速度_mean", "推力_mean", "刀盘扭矩_mean", "efficiency"]:
        if col in out.columns:
            global_mean = out[col].mean(skipna=True)
            if pd.notna(global_mean) and global_mean != 0:
                out[f"{col}_rel_change"] = (out[col] - global_mean) / global_mean
            else:
                out[f"{col}_rel_change"] = np.nan

    return out


def run_segment_analysis(df: pd.DataFrame, segment_length=10):
    """
    一键完成区段级分析
    """
    out = build_segments(df, segment_length=segment_length)
    out = aggregate_segments(out)
    out = format_segment_label(out)
    out = add_efficiency_indicator(out)
    out = add_relative_change_features(out)
    out = analyze_segment_response(out)

    # 排序更方便看
    if "segment_start_first" in out.columns:
        out = out.sort_values("segment_start_first").reset_index(drop=True)

    return out


def build_typical_segments_table(segment_df: pd.DataFrame, top_n=20):
    """
    生成典型区段表：优先筛选高风险、多源关注、施工异常区
    """
    out = segment_df.copy()

    if "risk_score_max" not in out.columns:
        return out.head(top_n)

    score = pd.Series(0, index=out.index, dtype=float)

    score += out["risk_score_max"].fillna(0) * 3

    if "active_source_count_max" in out.columns:
        score += out["active_source_count_max"].fillna(0) * 2

    if "推进速度_mean_rel_change" in out.columns:
        # 速度下降越多，优先级越高
        score += (-out["推进速度_mean_rel_change"].fillna(0)).clip(lower=0) * 10

    if "推力_mean_rel_change" in out.columns:
        score += out["推力_mean_rel_change"].fillna(0).clip(lower=0) * 6

    if "刀盘扭矩_mean_rel_change" in out.columns:
        score += out["刀盘扭矩_mean_rel_change"].fillna(0).clip(lower=0) * 6

    out["priority_score"] = score
    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)

    return out.head(top_n)