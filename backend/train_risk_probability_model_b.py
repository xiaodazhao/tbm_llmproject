# -*- coding: utf-8 -*-
"""
train_risk_probability_model_b_v3.py

升级版：地质证据 + 后续施工响应 联合定义风险标签
------------------------------------------------
增强点：
1. 保留地质融合缓存
2. 保留区段级特征
3. 增加趋势特征（前后段变化）
4. 增加风险平滑（rolling）
5. 增加风险等级划分（低/中/高）
6. 增加更适合汇报的可视化输出
7. 地质先验评分改为工程级门控逻辑
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from geology.geology_fusion_backend import load_evidence_db, attach_geology_labels


# =========================
# 基础工具
# =========================
GRADE_MAP = {
    "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5,
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
}


def normalize_grade(x):
    if pd.isna(x):
        return np.nan
    return GRADE_MAP.get(str(x).strip(), np.nan)


def ensure_time_chainage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "运行时间-time" not in out.columns and "time" in out.columns:
        out = out.rename(columns={"time": "运行时间-time"})

    if "运行时间-time" in out.columns:
        out["运行时间-time"] = pd.to_datetime(out["运行时间-time"], errors="coerce")

    if "chainage" not in out.columns:
        if "导向盾首里程" in out.columns:
            out["chainage"] = pd.to_numeric(out["导向盾首里程"], errors="coerce")
        elif "开累进尺" in out.columns:
            out["chainage"] = pd.to_numeric(out["开累进尺"], errors="coerce")
        else:
            out["chainage"] = np.nan
    else:
        out["chainage"] = pd.to_numeric(out["chainage"], errors="coerce")

    numeric_cols = [
        "推进速度", "推力", "刀盘扭矩", "刀盘实际转速", "推进给定速度", "掘进状态"
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["chainage"]).copy()

    if "运行时间-time" in out.columns:
        out = out.dropna(subset=["运行时间-time"]).sort_values("运行时间-time")
    else:
        out = out.sort_values("chainage")

    return out.reset_index(drop=True)


def build_fused_cache_path(plc_path: str, output_dir: str | Path) -> Path:
    plc_name = Path(plc_path).stem
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{plc_name}_geo_fused.parquet"


def load_or_build_fused_df(
    plc_path: str,
    evidence_db_path: str,
    output_dir: str | Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cache_path = build_fused_cache_path(plc_path, output_dir)

    if cache_path.exists() and not force_rebuild:
        print(f"[INFO] 读取融合缓存: {cache_path}")
        df_geo = pd.read_parquet(cache_path)
        df_geo = ensure_time_chainage(df_geo)
        return df_geo

    print(f"[INFO] 未找到融合缓存，开始读取原始 PLC: {plc_path}")
    plc_df = pd.read_csv(plc_path)
    plc_df = ensure_time_chainage(plc_df)
    print(f"[INFO] PLC 样本数: {len(plc_df)}")

    print(f"[INFO] 读取证据库: {evidence_db_path}")
    evidence_df = load_evidence_db(evidence_db_path)

    print("[INFO] 挂接地质融合标签...")
    df_geo = attach_geology_labels(plc_df, evidence_df)
    df_geo = ensure_time_chainage(df_geo)

    print(f"[INFO] 保存融合缓存: {cache_path}")
    df_geo.to_parquet(cache_path, index=False)

    return df_geo


def safe_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    if len(m) == 0:
        return s.iloc[0]
    return m.iloc[0]


def get_col(df: pd.DataFrame, *names, default=0):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def add_segment_id(df: pd.DataFrame, segment_len: float = 10.0) -> pd.DataFrame:
    out = df.copy()
    base = np.floor(out["chainage"].min() / segment_len) * segment_len
    out["segment_start"] = np.floor((out["chainage"] - base) / segment_len) * segment_len + base
    out["segment_end"] = out["segment_start"] + segment_len
    out["segment_id"] = (
        out["segment_start"].round(3).astype(str) + "_" + out["segment_end"].round(3).astype(str)
    )
    return out


# =========================
# 区段特征
# =========================
def build_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["segment_id", "segment_start", "segment_end"]
    agg_dict = {}

    geo_numeric_cols = [
        "active_source_count",
        "weighted_evidence_strength",
        "evidence_count",
        "grade_count",
        "water_evidence_count",
        "collapse_evidence_count",
        "deformation_evidence_count",
        "water_flag_fused",
        "collapse_flag_fused",
        "deformation_flag_fused",
    ]
    for col in geo_numeric_cols:
        if col in df.columns:
            agg_dict[col] = "max"

    geo_cat_cols = [
        "fused_grade",
        "hazard",
        "uncertainty",
        "coverage",
        "water_type_fused",
        "joint_degree_fused",
        "rock_mass_state_fused",
        "rock_uniformity_fused",
        "weathering_fused",
        "stability_fused",
        "lithology_fused",
    ]
    for col in geo_cat_cols:
        if col in df.columns:
            agg_dict[col] = safe_mode

    if "推进速度" in df.columns:
        agg_dict["推进速度"] = ["mean", "std", "median", "min", "max"]
    if "推力" in df.columns:
        agg_dict["推力"] = ["mean", "std", "median", "max"]
    if "刀盘扭矩" in df.columns:
        agg_dict["刀盘扭矩"] = ["mean", "std", "median", "max"]
    if "刀盘实际转速" in df.columns:
        agg_dict["刀盘实际转速"] = ["mean", "std", "median", "max"]

    if "掘进状态" in df.columns:
        df = df.copy()
        df["is_stop"] = (df["掘进状态"].fillna(0) == 0).astype(int)
        df["is_work"] = (df["掘进状态"].fillna(0) != 0).astype(int)
        agg_dict["is_stop"] = "mean"
        agg_dict["is_work"] = "mean"
    elif "推进速度" in df.columns:
        df = df.copy()
        df["is_stop"] = (df["推进速度"].fillna(0).abs() < 1e-8).astype(int)
        df["is_work"] = 1 - df["is_stop"]
        agg_dict["is_stop"] = "mean"
        agg_dict["is_work"] = "mean"

    seg = df.groupby(group_cols, as_index=False).agg(agg_dict)

    if isinstance(seg.columns, pd.MultiIndex):
        seg.columns = [
            "_".join([str(c) for c in col if c != ""]).strip("_")
            for col in seg.columns.to_flat_index()
        ]

    rename_map = {
        "推进速度_mean": "speed_mean",
        "推进速度_std": "speed_std",
        "推进速度_median": "speed_median",
        "推进速度_min": "speed_min",
        "推进速度_max": "speed_max",
        "推力_mean": "thrust_mean",
        "推力_std": "thrust_std",
        "推力_median": "thrust_median",
        "推力_max": "thrust_max",
        "刀盘扭矩_mean": "torque_mean",
        "刀盘扭矩_std": "torque_std",
        "刀盘扭矩_median": "torque_median",
        "刀盘扭矩_max": "torque_max",
        "刀盘实际转速_mean": "rpm_mean",
        "刀盘实际转速_std": "rpm_std",
        "刀盘实际转速_median": "rpm_median",
        "刀盘实际转速_max": "rpm_max",
        "is_stop_mean": "stop_ratio",
        "is_work_mean": "work_ratio",

        "active_source_count_max": "active_source_count",
        "weighted_evidence_strength_max": "weighted_evidence_strength",
        "evidence_count_max": "evidence_count",
        "grade_count_max": "grade_count",
        "water_evidence_count_max": "water_evidence_count",
        "collapse_evidence_count_max": "collapse_evidence_count",
        "deformation_evidence_count_max": "deformation_evidence_count",
        "water_flag_fused_max": "water_flag_fused",
        "collapse_flag_fused_max": "collapse_flag_fused",
        "deformation_flag_fused_max": "deformation_flag_fused",

        "fused_grade_safe_mode": "fused_grade",
        "hazard_safe_mode": "hazard",
        "uncertainty_safe_mode": "uncertainty",
        "coverage_safe_mode": "coverage",
        "water_type_fused_safe_mode": "water_type_fused",
        "joint_degree_fused_safe_mode": "joint_degree_fused",
        "rock_mass_state_fused_safe_mode": "rock_mass_state_fused",
        "rock_uniformity_fused_safe_mode": "rock_uniformity_fused",
        "weathering_fused_safe_mode": "weathering_fused",
        "stability_fused_safe_mode": "stability_fused",
        "lithology_fused_safe_mode": "lithology_fused",
    }
    seg = seg.rename(columns=rename_map)

    if "fused_grade" in seg.columns:
        seg["fused_grade_num"] = seg["fused_grade"].apply(normalize_grade)
    else:
        seg["fused_grade_num"] = np.nan

    return seg


def add_trend_features(seg_df: pd.DataFrame) -> pd.DataFrame:
    seg = seg_df.copy().sort_values("segment_start").reset_index(drop=True)

    trend_cols = [
        "speed_mean", "speed_std",
        "thrust_mean", "thrust_std",
        "torque_mean", "torque_std",
        "stop_ratio", "work_ratio",
        "rpm_mean", "rpm_std",
    ]

    for col in trend_cols:
        if col in seg.columns:
            seg[f"{col}_diff1"] = seg[col].diff()
            seg[f"{col}_pct_change"] = seg[col].pct_change().replace([np.inf, -np.inf], np.nan)

    if "speed_mean" in seg.columns and "torque_mean" in seg.columns:
        seg["speed_torque_ratio"] = seg["speed_mean"] / (seg["torque_mean"] + 1e-6)

    if "speed_mean" in seg.columns and "thrust_mean" in seg.columns:
        seg["speed_thrust_ratio"] = seg["speed_mean"] / (seg["thrust_mean"] + 1e-6)

    if "speed_std" in seg.columns and "speed_mean" in seg.columns:
        seg["speed_volatility"] = seg["speed_std"] / (seg["speed_mean"].abs() + 1e-6)

    return seg


# =========================
# 地质先验评分（完整版修订）
# =========================
def build_geology_prior(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    工程级改进版 geo_prior_score（最终修订版）

    逻辑：
    1. grade_score：围岩等级基础分
    2. consensus_score：多源一致性分
    3. hazard_score：危险组合门控分
       - 强危险必须更难触发
       - 中危险也要求有一定证据支撑
       - 危险项不再简单堆词累加
    """
    seg = seg_df.copy()

    active_source = get_col(seg, "active_source_count", default=0).fillna(0)
    weighted_strength = get_col(seg, "weighted_evidence_strength", default=0).fillna(0)

    water_flag = get_col(seg, "water_flag_fused", default=0).fillna(0)
    collapse_flag = get_col(seg, "collapse_flag_fused", default=0).fillna(0)
    deformation_flag = get_col(seg, "deformation_flag_fused", default=0).fillna(0)

    # 1) 围岩等级基础分
    if "fused_grade_num" in seg.columns:
        seg["grade_score"] = seg["fused_grade_num"].fillna(0) * 0.8
    else:
        seg["grade_score"] = 0.0

    # 2) 多源一致性分
    if weighted_strength.max() > weighted_strength.min():
        weighted_strength_norm = (weighted_strength - weighted_strength.min()) / (
            weighted_strength.max() - weighted_strength.min()
        )
    else:
        weighted_strength_norm = pd.Series(0, index=seg.index)

    seg["weighted_strength_norm"] = weighted_strength_norm

    seg["consensus_score"] = (
        active_source * 1.2 +
        weighted_strength_norm * 1.5
    )

    # 3) hazard 文本拆解
    if "hazard" in seg.columns:
        seg["hazard"] = seg["hazard"].fillna("").astype(str)

        seg["hazard_has_water"] = seg["hazard"].str.contains(
            "出水|涌水|突水", regex=True
        ).astype(int)

        seg["hazard_has_collapse"] = seg["hazard"].str.contains(
            "掉块|塌方|坍塌", regex=True
        ).astype(int)

        seg["hazard_has_deform"] = seg["hazard"].str.contains(
            "变形", regex=True
        ).astype(int)

        seg["hazard_has_joint"] = seg["hazard"].str.contains(
            "裂隙|节理", regex=True
        ).astype(int)

        seg["hazard_has_broken"] = seg["hazard"].str.contains(
            "破碎|极破碎", regex=True
        ).astype(int)

        seg["hazard_has_reflection"] = seg["hazard"].str.contains(
            "反射异常", regex=True
        ).astype(int)

    else:
        seg["hazard_has_water"] = 0
        seg["hazard_has_collapse"] = 0
        seg["hazard_has_deform"] = 0
        seg["hazard_has_joint"] = 0
        seg["hazard_has_broken"] = 0
        seg["hazard_has_reflection"] = 0

    # 4) 去重后的基础危险信号
    water_signal = np.maximum(water_flag, seg["hazard_has_water"])
    collapse_signal = np.maximum(collapse_flag, seg["hazard_has_collapse"])
    deform_signal = np.maximum(deformation_flag, seg["hazard_has_deform"])

    seg["water_signal"] = water_signal
    seg["collapse_signal"] = collapse_signal
    seg["deform_signal"] = deform_signal

    # 5) 工程门控逻辑（更严格）
    seg["core_hazard"] = (
        (water_signal == 1) &
        (collapse_signal == 1) &
        (active_source >= 2) &
        (weighted_strength_norm > 0.45)
    ).astype(int)

    seg["mid_hazard"] = (
        (
            (water_signal == 1) |
            (collapse_signal == 1)
        ) &
        (
            (seg["hazard_has_joint"] == 1) |
            (seg["hazard_has_broken"] == 1)
        ) &
        (active_source >= 2)
    ).astype(int)

    seg["light_hazard"] = (
        (
            (water_signal == 1) |
            (collapse_signal == 1) |
            (deform_signal == 1)
        ) &
        (seg["core_hazard"] == 0) &
        (seg["mid_hazard"] == 0)
    ).astype(int)

    seg["structure_risk"] = (
        seg["hazard_has_joint"] * 0.5 +
        seg["hazard_has_broken"] * 0.7 +
        seg["hazard_has_reflection"] * 0.4
    )

    # 6) 最终 hazard_score（整体降一点）
    seg["hazard_score"] = (
        seg["core_hazard"] * 2.5 +
        seg["mid_hazard"] * 1.4 +
        seg["light_hazard"] * 0.5 +
        seg["structure_risk"] +
        deform_signal * 0.7
    )

    # 7) 最终 geo_prior_score
    seg["geo_prior_score"] = (
        seg["grade_score"] +
        seg["consensus_score"] +
        seg["hazard_score"]
    )

    # 8) 分位数分级
    prior_q60 = seg["geo_prior_score"].quantile(0.60)
    prior_q75 = seg["geo_prior_score"].quantile(0.75)

    seg["geo_prior_mid"] = (seg["geo_prior_score"] >= prior_q60).astype(int)
    seg["geo_prior_high"] = (seg["geo_prior_score"] >= prior_q75).astype(int)

    return seg


# =========================
# 后续施工响应
# =========================
def compute_future_response_features(
    seg_df: pd.DataFrame,
    future_window_m: float = 10.0
) -> pd.DataFrame:
    seg = seg_df.copy().sort_values("segment_start").reset_index(drop=True)

    future_speed_mean = []
    future_thrust_mean = []
    future_torque_mean = []
    future_stop_ratio = []
    future_speed_std = []
    future_count = []

    for _, row in seg.iterrows():
        start = row["segment_end"]
        end = row["segment_end"] + future_window_m

        future_rows = seg[
            (seg["segment_start"] >= start) &
            (seg["segment_start"] < end)
        ]

        future_count.append(len(future_rows))

        if future_rows.empty:
            future_speed_mean.append(np.nan)
            future_thrust_mean.append(np.nan)
            future_torque_mean.append(np.nan)
            future_stop_ratio.append(np.nan)
            future_speed_std.append(np.nan)
        else:
            future_speed_mean.append(future_rows["speed_mean"].mean() if "speed_mean" in future_rows else np.nan)
            future_thrust_mean.append(future_rows["thrust_mean"].mean() if "thrust_mean" in future_rows else np.nan)
            future_torque_mean.append(future_rows["torque_mean"].mean() if "torque_mean" in future_rows else np.nan)
            future_stop_ratio.append(future_rows["stop_ratio"].mean() if "stop_ratio" in future_rows else np.nan)
            future_speed_std.append(future_rows["speed_std"].mean() if "speed_std" in future_rows else np.nan)

    seg["future_speed_mean"] = future_speed_mean
    seg["future_thrust_mean"] = future_thrust_mean
    seg["future_torque_mean"] = future_torque_mean
    seg["future_stop_ratio"] = future_stop_ratio
    seg["future_speed_std"] = future_speed_std
    seg["future_count"] = future_count

    return seg


# =========================
# 标签构造
# =========================
def build_response_based_label_v3(seg_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    seg = seg_df.copy()
    active_source = get_col(seg, "active_source_count", "active_source_count_max")

    thresholds = {}
    thresholds["speed_low_q35"] = float(seg["future_speed_mean"].quantile(0.35))
    thresholds["thrust_high_q70"] = float(seg["future_thrust_mean"].quantile(0.70))
    thresholds["torque_high_q70"] = float(seg["future_torque_mean"].quantile(0.70))
    thresholds["stop_high_q65"] = float(seg["future_stop_ratio"].quantile(0.65))
    thresholds["speed_std_high_q70"] = float(seg["future_speed_std"].quantile(0.70))
    thresholds["geo_prior_mid_q"] = float(seg["geo_prior_score"].quantile(0.60))
    thresholds["geo_prior_high_q"] = float(seg["geo_prior_score"].quantile(0.75))

    seg["flag_future_slowdown"] = (seg["future_speed_mean"] <= thresholds["speed_low_q35"]).astype(int)
    seg["flag_future_high_thrust"] = (seg["future_thrust_mean"] >= thresholds["thrust_high_q70"]).astype(int)
    seg["flag_future_high_torque"] = (seg["future_torque_mean"] >= thresholds["torque_high_q70"]).astype(int)
    seg["flag_future_high_stop"] = (seg["future_stop_ratio"] >= thresholds["stop_high_q65"]).astype(int)
    seg["flag_future_fluctuation"] = (seg["future_speed_std"] >= thresholds["speed_std_high_q70"]).astype(int)

    seg["future_response_bad"] = (
        (
            (seg["flag_future_slowdown"] == 1) &
            (
                (seg["flag_future_high_thrust"] == 1) |
                (seg["flag_future_high_torque"] == 1) |
                (seg["flag_future_fluctuation"] == 1)
            )
        ) |
        (
            (seg["flag_future_high_stop"] == 1) &
            (
                (seg["flag_future_high_thrust"] == 1) |
                (seg["flag_future_high_torque"] == 1) |
                (seg["flag_future_slowdown"] == 1)
            )
        )
    ).astype(int)

    seg["pure_stop_like"] = (
        (seg["flag_future_high_stop"] == 1) &
        (seg["flag_future_slowdown"] == 0) &
        (seg["flag_future_high_thrust"] == 0) &
        (seg["flag_future_high_torque"] == 0) &
        (seg["geo_prior_mid"] == 0)
    ).astype(int)

    seg["label_risk"] = (
        (
            (seg["geo_prior_high"] == 1) &
            (seg["future_response_bad"] == 1)
        ) |
        (
            (seg["geo_prior_mid"] == 1) &
            (seg["future_response_bad"] == 1) &
            (
                (seg["hazard_has_water"] == 1) |
                (seg["hazard_has_collapse"] == 1) |
                (seg["hazard_has_deform"] == 1) |
                (active_source.fillna(0) >= 3)
            )
        )
    ).astype(int)

    seg.loc[seg["pure_stop_like"] == 1, "label_risk"] = 0
    seg["label_valid"] = (seg["future_count"] > 0).astype(int)

    return seg, thresholds


# =========================
# 特征工程
# =========================
def prepare_training_data(seg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    exclude_cols = {
        "segment_id", "segment_start", "segment_end",
        "label_risk", "label_valid",
        "future_speed_mean", "future_thrust_mean", "future_torque_mean",
        "future_stop_ratio", "future_speed_std", "future_count",
        "flag_future_slowdown", "flag_future_high_thrust",
        "flag_future_high_torque", "flag_future_high_stop",
        "flag_future_fluctuation", "future_response_bad",
        "pure_stop_like",
        "risk_prob", "risk_prob_smooth", "risk_level",
    }

    X = seg[[c for c in seg.columns if c not in exclude_cols]].copy()
    y = seg["label_risk"].astype(int)

    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


def build_model(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return clf


def extract_feature_names_and_coefficients(clf: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = clf.named_steps["preprocessor"]
    model: LogisticRegression = clf.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coef = model.coef_.ravel()

    n = min(len(feature_names), len(coef))
    feature_names = feature_names[:n]
    coef = coef[:n]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return coef_df


# =========================
# 风险后处理
# =========================
def classify_risk_level(p: float) -> str:
    if pd.isna(p):
        return "未知"
    if p < 0.2:
        return "低"
    elif p < 0.3:
        return "中"
    return "高"


def postprocess_risk_result(result_df: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    df = result_df.copy().sort_values("segment_start").reset_index(drop=True)
    df["risk_prob_smooth"] = df["risk_prob"].rolling(smooth_window, center=True, min_periods=1).mean()
    df["risk_level"] = df["risk_prob_smooth"].apply(classify_risk_level)
    return df


# =========================
# 画图
# =========================
def plot_risk_profile(result_df: pd.DataFrame, output_dir: Path):
    df = result_df.sort_values("segment_start").copy()

    plt.figure(figsize=(12, 5))
    plt.plot(df["segment_start"], df["risk_prob"], linewidth=1.0, alpha=0.5, label="Raw Risk")
    plt.plot(df["segment_start"], df["risk_prob_smooth"], linewidth=2.0, label="Smoothed Risk")
    plt.axhline(0.3, linestyle=":", alpha=0.7, label="Alert Threshold=0.3")

    plt.xlim(1012000, 1018000)
    plt.ticklabel_format(style="plain", axis="x")
    plt.xlabel("Chainage")
    plt.ylabel("Risk Probability")
    plt.title("Risk Probability Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "risk_probability_profile.png", dpi=220)
    plt.close()


def plot_risk_speed_coupling(result_df: pd.DataFrame, output_dir: Path):
    df = result_df.copy()
    if "speed_mean" not in df.columns:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df["risk_prob_smooth"], df["speed_mean"], alpha=0.65)
    plt.axvline(0.3, linestyle=":", alpha=0.7)
    plt.xlabel("Smoothed Risk Probability")
    plt.ylabel("Mean Advance Speed")
    plt.title("Risk Probability vs Advance Speed")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_speed_coupling.png", dpi=220)
    plt.close()


def plot_risk_speed_profile(result_df: pd.DataFrame, output_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")

    df = result_df.sort_values("segment_start").copy()
    df = df[(df["segment_start"] >= 1012000) & (df["segment_start"] <= 1018000)]

    if df.empty:
        print("⚠️ 该里程范围内没有数据")
        return

    x = df["segment_start"]
    risk_raw = df["risk_prob"]
    risk_smooth = df["risk_prob_smooth"]
    speed = df["speed_mean"]

    fig, ax1 = plt.subplots(figsize=(15, 7))

    ax1.plot(x, risk_raw, color="tab:red", linewidth=1.0, alpha=0.35, label="Risk Probability")
    ax1.plot(x, risk_smooth, color="tab:red", linewidth=2.3, label="Smoothed Risk")
    ax1.fill_between(x, 0, risk_smooth, where=(risk_smooth > 0.3), color="tab:red", alpha=0.18, label="High Risk Zone")
    ax1.axhline(y=0.3, color="tab:red", linestyle=":", alpha=0.55)

    ax1.set_ylabel("Risk Probability", color="tab:red", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(x, speed, color="tab:blue", linestyle="--", linewidth=1.6, alpha=0.8, label="Advance Speed")
    ax2.set_ylabel("Advance Speed", color="tab:blue", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.grid(False)

    ax1.set_xlim(1012000, 1018000)

    def format_chainage(v, pos=None):
        return f"DK{v / 1000:.3f}"

    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(format_chainage))
    ax1.set_xlabel("Chainage", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.title("Tunnel Boring Machine Risk & Speed Profile", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "risk_speed_profile_optimized.png", dpi=300)
    plt.close()


def plot_risk_level_profile(result_df: pd.DataFrame, output_dir: Path):
    df = result_df.sort_values("segment_start").copy()
    df = df[(df["segment_start"] >= 1012000) & (df["segment_start"] <= 1018000)]

    if df.empty:
        return

    plt.figure(figsize=(15, 3.8))
    x = df["segment_start"]

    low = (df["risk_level"] == "低").astype(int)
    mid = (df["risk_level"] == "中").astype(int)
    high = (df["risk_level"] == "高").astype(int)

    plt.fill_between(x, 0, 1, where=low > 0, alpha=0.20, label="低风险")
    plt.fill_between(x, 0, 1, where=mid > 0, alpha=0.35, label="中风险")
    plt.fill_between(x, 0, 1, where=high > 0, alpha=0.45, label="高风险")

    plt.xlim(1012000, 1018000)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("Chainage")
    plt.title("Risk Level Zonation")
    plt.ticklabel_format(style="plain", axis="x")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_level_profile.png", dpi=220)
    plt.close()


# =========================
# 训练
# =========================
def train_probability_model(seg_df: pd.DataFrame, output_dir: Path):
    seg = seg_df.copy()
    seg = seg[seg["label_valid"] == 1].copy().reset_index(drop=True)

    X, y, numeric_features, categorical_features = prepare_training_data(seg)
    clf = build_model(numeric_features, categorical_features)

    seg = seg.sort_values("segment_start").reset_index(drop=True)
    split_idx = max(1, int(len(seg) * 0.8))

    train_df = seg.iloc[:split_idx].copy()
    test_df = seg.iloc[split_idx:].copy()

    X_train, y_train, _, _ = prepare_training_data(train_df)
    X_test, y_test, _, _ = prepare_training_data(test_df)

    clf.fit(X_train, y_train)

    train_df["risk_prob"] = clf.predict_proba(X_train)[:, 1]
    test_df["risk_prob"] = clf.predict_proba(X_test)[:, 1]

    metrics = {}
    try:
        metrics["train_auc"] = float(roc_auc_score(y_train, train_df["risk_prob"]))
    except Exception:
        metrics["train_auc"] = None

    try:
        metrics["test_auc"] = float(roc_auc_score(y_test, test_df["risk_prob"]))
    except Exception:
        metrics["test_auc"] = None

    try:
        metrics["test_ap"] = float(average_precision_score(y_test, test_df["risk_prob"]))
    except Exception:
        metrics["test_ap"] = None

    test_pred = (test_df["risk_prob"] >= 0.3).astype(int)
    metrics["decision_threshold"] = 0.3
    metrics["test_confusion_matrix"] = confusion_matrix(y_test, test_pred).tolist()
    metrics["test_classification_report"] = classification_report(
        y_test, test_pred, digits=4, output_dict=True, zero_division=0
    )

    result_df = pd.concat([
        train_df.assign(split="train"),
        test_df.assign(split="test")
    ], axis=0).sort_values("segment_start").reset_index(drop=True)

    result_df = postprocess_risk_result(result_df, smooth_window=5)
    coef_df = extract_feature_names_and_coefficients(clf)

    joblib.dump(clf, output_dir / "risk_probability_model_v3.joblib")
    result_df.to_csv(output_dir / "segment_risk_predictions_v3.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(output_dir / "feature_coefficients_v3.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "metrics_v3.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_risk_profile(result_df, output_dir)
    plot_risk_speed_coupling(result_df, output_dir)
    plot_risk_speed_profile(result_df, output_dir)
    plot_risk_level_profile(result_df, output_dir)

    return clf, result_df, coef_df, metrics


# =========================
# 主流程
# =========================
def run(
    plc_path: str,
    evidence_db_path: str,
    output_dir: str,
    segment_len: float,
    future_window_m: float,
    force_rebuild: bool = False
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_geo = load_or_build_fused_df(
        plc_path=plc_path,
        evidence_db_path=evidence_db_path,
        output_dir=output_dir,
        force_rebuild=force_rebuild,
    )
    print(f"[INFO] 融合后样本数: {len(df_geo)}")

    print("[INFO] 添加区段编号...")
    df_geo = add_segment_id(df_geo, segment_len=segment_len)

    print("[INFO] 构造区段级特征...")
    seg_df = build_segment_features(df_geo)

    print("[INFO] 增加趋势特征...")
    seg_df = add_trend_features(seg_df)

    print("[DEBUG] seg_df columns:")
    print(seg_df.columns.tolist())

    print("[INFO] 计算地质风险先验...")
    seg_df = build_geology_prior(seg_df)

    print("[INFO] 计算未来施工响应特征...")
    seg_df = compute_future_response_features(seg_df, future_window_m=future_window_m)

    print("[INFO] 构造改进版标签...")
    seg_df, thresholds = build_response_based_label_v3(seg_df)

    seg_df.to_csv(output_dir / "segment_features_with_labels_v3.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "label_thresholds_v3.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    print("[INFO] 开始训练模型...")
    clf, result_df, coef_df, metrics = train_probability_model(seg_df, output_dir)

    print("[INFO] 完成。输出如下：")
    print(output_dir / "segment_features_with_labels_v3.csv")
    print(output_dir / "segment_risk_predictions_v3.csv")
    print(output_dir / "risk_probability_model_v3.joblib")
    print(output_dir / "feature_coefficients_v3.csv")
    print(output_dir / "metrics_v3.json")
    print(output_dir / "risk_probability_profile.png")
    print(output_dir / "risk_speed_coupling.png")
    print(output_dir / "risk_speed_profile_optimized.png")
    print(output_dir / "risk_level_profile.png")

    show_cols = [c for c in [
        "segment_start", "segment_end", "risk_prob", "risk_prob_smooth", "risk_level",
        "label_risk", "geo_prior_score", "fused_grade", "hazard", "active_source_count",
        "speed_mean", "thrust_mean", "torque_mean", "stop_ratio"
    ] if c in result_df.columns]

    print("\n[INFO] 风险概率最高的前20个区段：")
    print(
        result_df.sort_values("risk_prob_smooth", ascending=False)[show_cols]
        .head(20)
        .to_string(index=False)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="TBM 风险概率模型 v3")
    parser.add_argument("--output_dir", type=str, default="outputs/risk_model_b_v3")
    parser.add_argument("--segment_len", type=float, default=10.0)
    parser.add_argument("--future_window_m", type=float, default=10.0)
    parser.add_argument("--force_rebuild", action="store_true", help="强制重新做地质融合并覆盖缓存")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        plc_path=r"C:\Users\22923\Desktop\伯舒拉岭_plc_超报数据\数据\tbm9伯舒拉岭右线\伯舒拉岭TBM_合并后.csv",
        evidence_db_path=r"C:\Users\22923\Desktop\evidence_db (2).csv",
        output_dir=args.output_dir,
        segment_len=args.segment_len,
        future_window_m=args.future_window_m,
        force_rebuild=args.force_rebuild,
    )