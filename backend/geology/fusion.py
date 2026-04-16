# geology/fusion.py
import json
import re
import pandas as pd
from config import TOLERANCE_M


GRADE_ORDER = {
    "Ⅰ": 1,
    "Ⅱ": 2,
    "Ⅲ": 3,
    "Ⅳ": 4,
    "Ⅴ": 5,
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
}


def get_active(chainage, evidence, point_buffer=5.0):
    """
    获取当前里程命中的所有地质证据
    - segment / report_conclusion: 按区间命中
    - point: 按点位缓冲命中
    """
    if evidence is None or evidence.empty:
        return evidence

    if "source_level" not in evidence.columns:
        return evidence[
            (evidence["start_num"] - TOLERANCE_M <= chainage) &
            (evidence["end_num"] + TOLERANCE_M >= chainage)
        ]

    seg_df = evidence[evidence["source_level"] != "point"].copy()
    point_df = evidence[evidence["source_level"] == "point"].copy()

    hit_seg = seg_df[
        (seg_df["start_num"] - TOLERANCE_M <= chainage) &
        (seg_df["end_num"] + TOLERANCE_M >= chainage)
    ]

    hit_point = point_df[
        (point_df["start_num"] - point_buffer <= chainage) &
        (point_df["end_num"] + point_buffer >= chainage)
    ]

    return pd.concat([hit_seg, hit_point], ignore_index=True)
def normalize_report_id(report_id: str) -> str:
    """
    规范化报告ID，避免同一报告因命名差异被重复统计
    """
    if pd.isna(report_id):
        return ""

    report_id = str(report_id).strip()
    report_id = re.sub(r"_[0-9]+$", "", report_id)
    report_id = report_id.replace("-_", "_")
    report_id = re.sub(r"[-_]+", "_", report_id)
    report_id = report_id.rstrip("_")
    return report_id


def _safe_load_attrs(attrs_json_str):
    try:
        data = json.loads(attrs_json_str)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_grade(x):
    """
    统一围岩等级字段
    """
    if x is None:
        return None

    x = str(x).strip()
    if not x:
        return None

    mapping = {
        "1": "Ⅰ",
        "2": "Ⅱ",
        "3": "Ⅲ",
        "4": "Ⅳ",
        "5": "Ⅴ",
        "I": "Ⅰ",
        "II": "Ⅱ",
        "III": "Ⅲ",
        "IV": "Ⅳ",
        "V": "Ⅴ",
        "Ⅰ": "Ⅰ",
        "Ⅱ": "Ⅱ",
        "Ⅲ": "Ⅲ",
        "Ⅳ": "Ⅳ",
        "Ⅴ": "Ⅴ",
    }
    return mapping.get(x, x)


def _record_weight(row, attrs):
    """
    这里只作为“证据强度/可信度”的权重，不再用于风险评定
    """
    weight = 1.0

    source_level = str(row.get("source_level", "")).strip()
    confidence = str(row.get("confidence", "")).strip().lower()

    if source_level == "report_conclusion":
        weight *= 1.20
    elif source_level == "segment":
        weight *= 1.00
    elif source_level == "point":
        weight *= 1.10

    if confidence == "high":
        weight *= 1.15
    elif confidence == "medium":
        weight *= 1.00
    elif confidence == "low":
        weight *= 0.90

    if attrs.get("consistency_flag") == 0:
        weight *= 0.90
    if attrs.get("grade_conflict") == 1:
        weight *= 0.95

    return weight


def _pick_mode(values):
    values = [v for v in values if v not in [None, "", [], {}]]
    if not values:
        return None
    s = pd.Series(values)
    mode = s.mode()
    if len(mode) == 0:
        return values[0]
    return mode.iloc[0]


def _pick_worst_grade(grades):
    grades = [_normalize_grade(g) for g in grades if g]
    if not grades:
        return ""
    return sorted(
        grades,
        key=lambda x: GRADE_ORDER.get(str(x), 0),
        reverse=True
    )[0]


def _dedup_preserve_order(items):
    out = []
    seen = set()
    for x in items:
        if x in [None, ""]:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _merge_hazard_tags(parsed_rows):
    hazard_set = []

    for row, attrs in parsed_rows:
        water_type = attrs.get("water_type")
        if water_type:
            hazard_set.append(str(water_type))
        elif attrs.get("water_flag"):
            hazard_set.append("出水")

        if attrs.get("collapse_flag"):
            hazard_set.append("掉块")

        if attrs.get("deformation_flag"):
            hazard_set.append("变形")

        tags = attrs.get("risk_tags", [])
        if isinstance(tags, list):
            for t in tags:
                if t:
                    hazard_set.append(str(t))

    hazard_list = _dedup_preserve_order(hazard_set)
    return hazard_list, "+".join(hazard_list) if hazard_list else "无明显异常"


def _merge_water_type(parsed_rows):
    vals = []
    for _, attrs in parsed_rows:
        x = attrs.get("water_type")
        if x:
            vals.append(str(x))
    vals = _dedup_preserve_order(vals)
    return ";".join(vals) if vals else None


def _weighted_mean(values, weights):
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    return round(sum(v * w for v, w in pairs) / total_w, 4)


def _count_true_flags(parsed_rows, key):
    vals = []
    for _, attrs in parsed_rows:
        vals.append(1 if attrs.get(key) else 0)
    return int(sum(vals))


def _merge_field_mode(parsed_rows, key):
    vals = []
    for _, attrs in parsed_rows:
        x = attrs.get(key)
        if x not in [None, ""]:
            vals.append(x)
    return _pick_mode(vals)


def fuse(chainage, df):
    """
    对某一里程的命中证据进行融合
    —— 只做“事实融合”，不做风险评定
    """
    if df.empty:
        return {
            "chainage": chainage,
            "coverage": "none",

            # 事实型输出
            "hazard": "无明显异常",
            "hazard_list": "",
            "fused_grade": "",
            "water_flag_fused": 0,
            "water_type_fused": None,
            "collapse_flag_fused": 0,
            "deformation_flag_fused": 0,
            "joint_degree_fused": None,
            "rock_mass_state_fused": None,
            "rock_uniformity_fused": None,
            "weathering_fused": None,
            "stability_fused": None,
            "lithology_fused": None,

            # 来源与命中
            "active_source_count": 0,
            "active_sources": "",
            "active_evidence_ids": "",
            "active_report_ids": "",
            "active_source_levels": "",
            "uncertainty": "high",

            # 证据统计
            "evidence_count": 0,
            "weighted_evidence_strength": 0.0,
            "grade_count": 0,
            "water_evidence_count": 0,
            "collapse_evidence_count": 0,
            "deformation_evidence_count": 0,
        }

    parsed_rows = []
    weights = []
    for _, row in df.iterrows():
        attrs = _safe_load_attrs(row.get("attrs_json", ""))
        parsed_rows.append((row, attrs))
        weights.append(_record_weight(row, attrs))

    # ===== 字段融合 =====
    grades = []
    for _, attrs in parsed_rows:
        g = _normalize_grade(attrs.get("support_grade") or attrs.get("rock_grade"))
        if g:
            grades.append(g)

    fused_grade = _pick_worst_grade(grades)

    water_flag_fused = 1 if _count_true_flags(parsed_rows, "water_flag") > 0 else 0
    collapse_flag_fused = 1 if _count_true_flags(parsed_rows, "collapse_flag") > 0 else 0
    deformation_flag_fused = 1 if _count_true_flags(parsed_rows, "deformation_flag") > 0 else 0

    water_type_fused = _merge_water_type(parsed_rows)
    joint_degree_fused = _merge_field_mode(parsed_rows, "joint_degree")
    rock_mass_state_fused = _merge_field_mode(parsed_rows, "rock_mass_state")
    rock_uniformity_fused = _merge_field_mode(parsed_rows, "rock_uniformity")
    weathering_fused = _merge_field_mode(parsed_rows, "weathering")
    stability_fused = _merge_field_mode(parsed_rows, "stability")
    lithology_fused = _merge_field_mode(parsed_rows, "lithology")

    hazard_list, hazard = _merge_hazard_tags(parsed_rows)

    # ===== 来源统计 =====
    unique_report_keys = set()
    unique_report_ids = set()
    unique_sources = set()
    unique_levels = set()

    for _, row in df.iterrows():
        source_type = str(row.get("source_type", "")).strip()
        source_level = str(row.get("source_level", "")).strip()
        report_id = normalize_report_id(row.get("report_id", ""))

        unique_report_keys.add((source_type, report_id))
        if report_id:
            unique_report_ids.add(report_id)
        if source_type:
            unique_sources.add(source_type)
        if source_level:
            unique_levels.add(source_level)

    active_source_count = len(unique_report_keys)

    if active_source_count >= 3:
        uncertainty = "low"
    elif active_source_count == 2:
        uncertainty = "medium"
    else:
        uncertainty = "high"

    # ===== 证据统计 =====
    evidence_count = len(parsed_rows)
    weighted_evidence_strength = round(sum(weights), 4)

    water_evidence_count = _count_true_flags(parsed_rows, "water_flag")
    collapse_evidence_count = _count_true_flags(parsed_rows, "collapse_flag")
    deformation_evidence_count = _count_true_flags(parsed_rows, "deformation_flag")

    return {
        "chainage": chainage,
        "coverage": "multi" if active_source_count > 1 else "single",

        # 事实型输出
        "hazard": hazard,
        "hazard_list": ";".join(hazard_list),
        "fused_grade": fused_grade,
        "water_flag_fused": water_flag_fused,
        "water_type_fused": water_type_fused,
        "collapse_flag_fused": collapse_flag_fused,
        "deformation_flag_fused": deformation_flag_fused,
        "joint_degree_fused": joint_degree_fused,
        "rock_mass_state_fused": rock_mass_state_fused,
        "rock_uniformity_fused": rock_uniformity_fused,
        "weathering_fused": weathering_fused,
        "stability_fused": stability_fused,
        "lithology_fused": lithology_fused,

        # 来源与命中
        "active_source_count": active_source_count,
        "active_sources": ";".join(sorted(unique_sources)),
        "active_evidence_ids": ";".join(df["evidence_id"].astype(str).tolist()),
        "active_report_ids": ";".join(sorted(unique_report_ids)),
        "active_source_levels": ";".join(sorted(unique_levels)),
        "uncertainty": uncertainty,

        # 证据统计
        "evidence_count": evidence_count,
        "weighted_evidence_strength": weighted_evidence_strength,
        "grade_count": len(grades),
        "water_evidence_count": water_evidence_count,
        "collapse_evidence_count": collapse_evidence_count,
        "deformation_evidence_count": deformation_evidence_count,
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