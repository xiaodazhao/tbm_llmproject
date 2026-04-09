# geology/fusion.py
import json
import re
import pandas as pd
from config import TOLERANCE_M


RISK_SCORE_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3
}


def get_active(chainage, evidence):
    """
    获取当前里程命中的所有地质证据
    """
    return evidence[
        (evidence["start_num"] - TOLERANCE_M <= chainage) &
        (evidence["end_num"] + TOLERANCE_M >= chainage)
    ]


def normalize_report_id(report_id: str) -> str:
    """
    规范化报告ID，避免同一报告因命名差异被重复统计
    """
    if pd.isna(report_id):
        return ""

    report_id = str(report_id).strip()

    # ① 去掉 _0 _1 等编号
    report_id = re.sub(r"_[0-9]+$", "", report_id)

    # ② 统一 "-_" → "_"
    report_id = report_id.replace("-_", "_")

    # ③ 去掉连续符号
    report_id = re.sub(r"[-_]+", "_", report_id)

    # ④ 去掉末尾多余 _
    report_id = report_id.rstrip("_")

    return report_id


def _safe_load_attrs(attrs_json_str):
    try:
        data = json.loads(attrs_json_str)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def fuse(chainage, df):
    """
    对某一里程的命中证据进行融合
    """
    # ===== 无证据 =====
    if df.empty:
        return {
            "chainage": chainage,
            "coverage": "none",
            "risk": "low",
            "risk_score": 1,
            "hazard": "无明显异常",
            "water_flag_fused": 0,
            "collapse_flag_fused": 0,
            "deformation_flag_fused": 0,
            "active_source_count": 0,
            "active_sources": "",
            "active_evidence_ids": "",
            "active_report_ids": "",
            "fused_grade": "",
            "uncertainty": "high"
        }

    attrs = [_safe_load_attrs(a) for a in df["attrs_json"]]

    # ===== 风险投票 =====
    scores = []
    for a in attrs:
        r = a.get("risk_level")
        if r in RISK_SCORE_MAP:
            scores.append(RISK_SCORE_MAP[r])

    if not scores:
        risk = "low"
    else:
        avg_score = sum(scores) / len(scores)
        if avg_score >= 2.5:
            risk = "high"
        elif avg_score >= 1.8:
            risk = "medium"
        else:
            risk = "low"

    risk_score = RISK_SCORE_MAP.get(risk, 1)

    # ===== 灾害融合 =====
    hazard_set = set()
    grades = []

    water_flag_fused = 0
    collapse_flag_fused = 0
    deformation_flag_fused = 0

    for a in attrs:
        if a.get("water_flag"):
            hazard_set.add("出水")
            water_flag_fused = 1

        if a.get("collapse_flag"):
            hazard_set.add("掉块")
            collapse_flag_fused = 1

        if a.get("deformation_flag"):
            hazard_set.add("变形")
            deformation_flag_fused = 1

        rock_grade = a.get("rock_grade")
        if rock_grade:
            grades.append(str(rock_grade))

    hazard = "+".join(sorted(hazard_set)) if hazard_set else "无明显异常"

    # ===== 围岩等级融合（从严原则）=====
    fused_grade = ""
    if grades:
        order = {"Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5}
        fused_grade = sorted(
            grades,
            key=lambda x: order.get(str(x), 0),
            reverse=True
        )[0]

    # ===== 按 source_type + report_id 去重统计 =====
    unique_report_keys = set()
    unique_report_ids = set()

    for _, row in df.iterrows():
        source_type = str(row.get("source_type", "")).strip()
        report_id = normalize_report_id(row.get("report_id", ""))
        unique_report_keys.add((source_type, report_id))
        unique_report_ids.add(report_id)

    active_source_count = len(unique_report_keys)

    unique_sources = sorted(set(df["source_type"].astype(str)))

    # ===== 不确定性 =====
    if active_source_count >= 3:
        uncertainty = "low"
    elif active_source_count == 2:
        uncertainty = "medium"
    else:
        uncertainty = "high"

    return {
        "chainage": chainage,
        "coverage": "multi" if active_source_count > 1 else "single",
        "risk": risk,
        "risk_score": risk_score,
        "hazard": hazard,
        "water_flag_fused": water_flag_fused,
        "collapse_flag_fused": collapse_flag_fused,
        "deformation_flag_fused": deformation_flag_fused,
        "active_source_count": active_source_count,
        "active_sources": ";".join(unique_sources),
        "active_evidence_ids": ";".join(df["evidence_id"].astype(str).tolist()),
        "active_report_ids": ";".join(sorted([x for x in unique_report_ids if x])),
        "fused_grade": fused_grade,
        "uncertainty": uncertainty
    }


def annotate_unique_chainage(unique_chainage_df, evidence):
    """
    对唯一里程表做逐里程注记
    """
    results = []
    total = len(unique_chainage_df)

    for i, x in enumerate(unique_chainage_df["chainage"], start=1):
        if i % 5000 == 0:
            print(f"已处理唯一里程: {i}/{total}")

        hit_df = get_active(x, evidence)
        results.append(fuse(x, hit_df))

    return pd.DataFrame(results)