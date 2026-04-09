import pandas as pd
from pathlib import Path
from typing import Union
import glob

from analysis.dataprocess import (
    load_and_process,
    segments_to_text,
    compute_stats,
    stats_to_text,
)
from analysis.excavation_state import (
    detect_excavation_state,
    excavation_state_segments,
    explain_excavation_states,
    excavation_state_to_text,
    excavation_state_efficiency,
    efficiency_to_text,
    excavation_state_stats,
    excavation_state_stats_to_text,
)
from analysis.gas_analysis import (
    compute_gas_stats,
    gas_stats_to_text,
)
from geology.geology_fusion_backend import attach_geology_labels, load_evidence_db
from geology.geology_summary import summarize_geology, geology_summary_to_text
from llm.prompt_builder import build_prompt


# ==============================
# 1. 路径配置 (注意：这里新增了每日数据的文件夹路径)
# ==============================
from config import DATA_DIR, EVIDENCE_DB_PATH, DAILY_RESULT_DIR
RESULT_DIR = DAILY_RESULT_DIR
DAILY_DATA_DIR = DATA_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# 2. 工具函数
# ==============================
def infer_plc_day(df: pd.DataFrame) -> pd.Timestamp:
    if "运行时间-time" not in df.columns:
        raise ValueError("缺少时间列：运行时间-time")
    ts = pd.to_datetime(df["运行时间-time"], errors="coerce").dropna()
    if ts.empty:
        raise ValueError("运行时间-time 无法解析出有效时间")
    return ts.min().normalize()

def filter_evidence_by_day(evidence_df: pd.DataFrame, plc_day: pd.Timestamp) -> pd.DataFrame:
    out = evidence_df.copy()
    if "report_date" not in out.columns:
        return out
    out["report_date_dt"] = pd.to_datetime(out["report_date"], errors="coerce")
    out = out[out["report_date_dt"].isna() | (out["report_date_dt"] <= plc_day)].copy()
    out.drop(columns=["report_date_dt"], inplace=True, errors="ignore")
    return out

def build_llm_summary(geo_summary: dict, stats: dict, state_labels: dict, eff_df: pd.DataFrame) -> dict:
    summary = {
        "geology": geo_summary,
        "basic_stats": {
            "stop_count": stats.get("stop_count"),
            "transition_count": stats.get("transition_count"),
            "work_count": stats.get("work_count"),
            "abnormal_count": stats.get("abnormal_count"),
            "stop_total_min": stats.get("stop_total_min"),
            "transition_total_min": stats.get("transition_total_min"),
            "work_total_min": stats.get("work_total_min"),
            "abnormal_total_min": stats.get("abnormal_total_min"),
        },
        "state_labels": state_labels,
        "efficiency_columns": list(eff_df.columns) if isinstance(eff_df, pd.DataFrame) and not eff_df.empty else [],
    }
    return summary


# ==============================
# 3. 主流程 (核心引擎保持不变)
# ==============================
def run_daily_twin_analysis(plc_source: Union[str, Path, pd.DataFrame], save_outputs: bool = True):
    if isinstance(plc_source, pd.DataFrame):
        df = plc_source.copy()
        plc_name = "plc_dataframe"
    else:
        plc_source = Path(plc_source)
        plc_name = plc_source.stem
        df = pd.read_csv(plc_source)

    if "运行时间-time" not in df.columns:
        # 兼容性处理：如果列名叫 time，重命名一下
        if "time" in df.columns:
            df.rename(columns={"time": "运行时间-time"}, inplace=True)
        else:
            raise ValueError("缺少时间列：运行时间-time")

    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = df.dropna(subset=["运行时间-time"]).sort_values("运行时间-time").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("数据量过少，不足以进行分析。")

    plc_day = infer_plc_day(df)

    # 地质融合
    evidence_df = load_evidence_db(EVIDENCE_DB_PATH)
    evidence_df_day = filter_evidence_by_day(evidence_df, plc_day)
    df_geo = attach_geology_labels(df, evidence_df_day)

    # 基础工况
    segments = load_and_process(df_geo)
    seg_text = segments_to_text(segments)
    stats = compute_stats(segments)
    stats_text = stats_to_text(stats)

    # 状态识别
    df_state, _ = detect_excavation_state(df_geo)
    state_segments = excavation_state_segments(df_state, min_duration_sec=30)
    state_labels = explain_excavation_states(df_state)
    state_text = excavation_state_to_text(state_segments, state_labels)
    eff_df = excavation_state_efficiency(df_state)
    eff_text = efficiency_to_text(eff_df, state_labels)
    state_stats = excavation_state_stats(df_state, state_segments)
    state_stats_text = excavation_state_stats_to_text(state_stats, state_labels)

    # 气体分析
    gas_stats = compute_gas_stats(df_geo, df_state)
    gas_text = gas_stats_to_text(gas_stats)

    # 地质摘要
    geo_summary = summarize_geology(df_geo)
    geo_text = geology_summary_to_text(geo_summary)

    llm_summary = build_llm_summary(geo_summary, stats, state_labels, eff_df)

    prompt = build_prompt(
        seg_text=seg_text,
        stats_text=stats_text,
        state_text=state_text,
        eff_text=eff_text,
        state_stats_text=state_stats_text,
        gas_text=gas_text,
        geo_text=geo_text,
        llm_summary=llm_summary,
    )

    result = {
        "plc_day": plc_day,
        "out_dir": None,
        "geo_text": geo_text
    }

    if save_outputs:
        day_str = plc_day.strftime("%Y%m%d")
        out_dir = RESULT_DIR / f"{day_str}_{plc_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_geo.to_csv(out_dir / "plc_with_geology.csv", index=False, encoding="utf-8-sig")

        if "chainage" in df_geo.columns:
            unique_cols = [c for c in ["chainage", "coverage", "risk", "hazard", "active_source_count", "active_sources", "active_report_ids", "fused_grade", "uncertainty"] if c in df_geo.columns]
            df_geo[unique_cols].drop_duplicates().sort_values("chainage").to_csv(out_dir / "chainage_geology_unique.csv", index=False, encoding="utf-8-sig")

        (out_dir / "seg_text.txt").write_text(seg_text, encoding="utf-8")
        (out_dir / "stats_text.txt").write_text(stats_text, encoding="utf-8")
        (out_dir / "state_text.txt").write_text(state_text, encoding="utf-8")
        (out_dir / "eff_text.txt").write_text(eff_text, encoding="utf-8")
        (out_dir / "state_stats_text.txt").write_text(state_stats_text, encoding="utf-8")
        (out_dir / "gas_text.txt").write_text(gas_text, encoding="utf-8")
        (out_dir / "geo_text.txt").write_text(geo_text, encoding="utf-8")
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        summary_txt = f"PLC日期: {plc_day.strftime('%Y-%m-%d')}\n原始PLC记录数: {len(df)}\n融合后PLC记录数: {len(df_geo)}\n地质摘要:\n{geo_text}\n"
        (out_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")

        result["out_dir"] = out_dir

    return result


# ==============================
# 4. 脚本入口 (全面重构：支持文件夹批量遍历)
# ==============================
if __name__ == "__main__":
    print(f"正在扫描文件夹: {DAILY_DATA_DIR}")
    
    # 查找文件夹下的所有 CSV 文件
    daily_files = sorted(DAILY_DATA_DIR.glob("*.csv"))
    
    if not daily_files:
        print("❌ 未找到任何 CSV 文件，请检查 DAILY_DATA_DIR 路径是否正确！")
    else:
        print(f"共发现 {len(daily_files)} 个每日数据文件，开始批量处理...\n")
        
        success_count = 0
        for plc_file in daily_files:
            print(f"[{plc_file.name}] 开始分析...")
            try:
                # 调用核心引擎处理单个文件
                result = run_daily_twin_analysis(plc_file, save_outputs=True)
                print(f"  ✅ 成功！输出目录: {result['out_dir']}")
                success_count += 1
            except Exception as e:
                print(f"  ⚠️ 跳过 {plc_file.name}：{e}")
        
        print(f"\n🎉 批量处理完成！共成功处理 {success_count}/{len(daily_files)} 天的数据。")