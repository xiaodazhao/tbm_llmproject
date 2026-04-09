import os
import glob
from pathlib import Path
import pandas as pd

# =========================
# 导入你分好类的各个模块
# =========================
from analysis.dataprocess import load_and_process, segments_to_text, compute_stats, stats_to_text
from analysis.excavation_state import (
    detect_excavation_state, excavation_state_segments, explain_excavation_states,
    excavation_state_to_text, excavation_state_efficiency, excavation_state_stats,
    excavation_state_stats_to_text, efficiency_to_text
)
from analysis.gas_analysis import compute_gas_stats, gas_stats_to_text

from geology.geology_fusion_backend import attach_geology_labels, load_evidence_db
from geology.geology_summary import summarize_geology, geology_summary_to_text

from llm.prompt_builder import build_prompt

# =========================
# 路径配置
# =========================
from config import DATA_DIR, EVIDENCE_DB_PATH


def get_test_data() -> pd.DataFrame:
    files = glob.glob(str(DATA_DIR / "tbm_data_*.csv"))
    if not files:
        raise FileNotFoundError(f"❌ 在 {DATA_DIR} 目录下找不到任何测试数据！")
    latest_file = max(files)
    print(f"📄 加载测试数据: {os.path.basename(latest_file)}")
    
    df = pd.read_csv(latest_file)
    if "运行时间-time" not in df.columns and "time" in df.columns:
        df.rename(columns={"time": "运行时间-time"}, inplace=True)
        
    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = df.dropna(subset=["运行时间-time"]).sort_values("运行时间-time").reset_index(drop=True)
    return df


def run_diagnostics():
    print("==================================================")
    print("🚀 启动 TBM 数据分析全流程本地测试 (底层透视全量 + 文本详尽版)")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 0. 准备数据
    # ---------------------------------------------------------
    print("▶️ [步骤 0] 加载原始数据")
    try:
        df = get_test_data()
        print(f"  ✅ 成功加载数据，共 {len(df)} 行，包含列数: {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ 数据加载失败，程序终止: {e}")
        return

    # ---------------------------------------------------------
    # 1. 地质融合 (Geology)
    # ---------------------------------------------------------
    print("\n▶️ [步骤 1] 地质特征融合 (geology 模块)")
    try:
        if not EVIDENCE_DB_PATH.exists():
            print(f"  ⚠️ 警告: 找不到证据库文件 {EVIDENCE_DB_PATH}，将跳过地质融合。")
            df_geo = df.copy()
            geo_summary = {"has_geology": False}
            geo_text = "地质融合不可用。"
        else:
            evidence_df = load_evidence_db(EVIDENCE_DB_PATH)
            df_geo = attach_geology_labels(df, evidence_df)
            
            print("  ✅ 融合计算完成。")
            print("\n↓↓↓ [揭秘: 融合后底层新增的地质字段 (所有不重复里程)] ↓↓↓")
            if "chainage" in df_geo.columns:
                geo_cols = [c for c in ["chainage", "coverage", "risk", "hazard", "fused_grade", "active_source_count", "active_sources"] if c in df_geo.columns]
                if geo_cols:
                    # 【修改点】：去掉了 .head(5)，展示全部的不重复里程数据
                    unique_geo = df_geo[geo_cols].drop_duplicates(subset=["chainage"])
                    print(unique_geo.to_markdown(index=False))
                else:
                    print("  ⚠️ 融合后未生成地质扩展字段")
            else:
                 print("  ⚠️ 数据中未找到里程字段 (chainage)")
            print("↑↑↑-------------------------------------------------------↑↑↑\n")

            geo_summary = summarize_geology(df_geo)
            geo_text = geology_summary_to_text(geo_summary)
            print(f"  📝 [LLM将看到的摘要文本]: \n{geo_text}")
            
    except Exception as e:
        print(f"  ❌ 地质融合模块报错: {e}")
        df_geo = df.copy()
        geo_text = "报错跳过"
        geo_summary = {}

    # ---------------------------------------------------------
    # 2. 基础工况划分 (DataProcess)
    # ---------------------------------------------------------
    print("\n▶️ [步骤 2] 基础工况划分 (dataprocess 模块)")
    try:
        segments = load_and_process(df_geo)
        seg_text = segments_to_text(segments)
        stats = compute_stats(segments)
        stats_text = stats_to_text(stats)
        
        print(f"  ✅ 成功。共划分出 {len(segments)} 个工况段。")
        print("\n↓↓↓ [输出: seg_text (工况分段)] ↓↓↓")
        print(seg_text)
        print("\n↓↓↓ [输出: stats_text (工况统计)] ↓↓↓")
        print(stats_text)
        print("↑↑↑-----------------------------↑↑↑")
    except Exception as e:
        print(f"  ❌ 基础工况模块报错: {e}")
        seg_text, stats_text, stats = "", "", {}

    # ---------------------------------------------------------
    # 3. 隐含施工状态识别 (Excavation State Clustering)
    # ---------------------------------------------------------
    print("\n▶️ [步骤 3] 隐含施工状态聚类 (excavation_state 模块)")
    try:
        df_state, states_array = detect_excavation_state(df_geo.copy(), n_states=3)
        if df_state is not None and "state_id" in df_state.columns:
            state_labels = explain_excavation_states(df_state)
            state_segments = excavation_state_segments(df_state, min_duration_sec=30)
            state_text = excavation_state_to_text(state_segments, state_labels)
            
            eff_df = excavation_state_efficiency(df_state)
            eff_text = efficiency_to_text(eff_df, state_labels)
            
            state_stats = excavation_state_stats(df_state, state_segments)
            state_stats_text = excavation_state_stats_to_text(state_stats, state_labels)
            
            print(f"  ✅ 成功。识别出状态标签: {state_labels}")
            print("\n↓↓↓ [输出: state_text (状态分段)] ↓↓↓")
            print(state_text)
            print("\n↓↓↓ [输出: eff_text (状态效率)] ↓↓↓")
            print(eff_text)
            print("\n↓↓↓ [输出: state_stats_text (状态统计)] ↓↓↓")
            print(state_stats_text)
            print("↑↑↑---------------------------------↑↑↑")
        else:
            print("  ⚠️ 数据不足以进行聚类分析。")
            state_text, eff_text, state_stats_text, eff_df = "", "", "", pd.DataFrame()
    except Exception as e:
        print(f"  ❌ 状态聚类模块报错: {e}")
        state_text, eff_text, state_stats_text, eff_df = "", "", "", pd.DataFrame()

    # ---------------------------------------------------------
    # 4. 气体分析 (Gas Analysis)
    # ---------------------------------------------------------
    print("\n▶️ [步骤 4] 气体安全检测 (gas_analysis 模块)")
    try:
        gas_stats = compute_gas_stats(df_geo, df_state=df_state if 'df_state' in locals() else None)
        gas_text = gas_stats_to_text(gas_stats)
        print("  ✅ 成功。")
        print("\n↓↓↓ [输出: gas_text (气体分析)] ↓↓↓")
        print(gas_text.strip() if gas_text.strip() else "数据中无超标事件或无有效文本生成。")
        print("↑↑↑-----------------------------↑↑↑")
    except Exception as e:
        print(f"  ❌ 气体模块报错: {e}")
        gas_text = ""

    # ---------------------------------------------------------
    # 5. LLM 组装 (仅组装，不打印，不调用)
    # ---------------------------------------------------------
    print("\n▶️ [步骤 5] 大模型 Prompt 组装 (llm 模块)")
    try:
        llm_summary_mock = {
            "geology": geo_summary,
            "basic_stats": stats,
            "efficiency_cols": list(eff_df.columns) if not eff_df.empty else []
        }
        
        prompt = build_prompt(
            seg_text=seg_text,
            stats_text=stats_text,
            state_text=state_text,
            eff_text=eff_text,
            state_stats_text=state_stats_text,
            gas_text=gas_text,
            geo_text=geo_text,
            llm_summary=llm_summary_mock
        )
        print(f"  ✅ Prompt 组装成功！文本总长度: {len(prompt)} 字符。")
        print("  🛑 [设置默认]: 已跳过打印完整 Prompt 文本。")
        print("  🛑 [设置默认]: 已跳过调用大模型 API。")

    except Exception as e:
        print(f"  ❌ LLM Prompt 模块报错: {e}")

    print("\n🎉 本地全量透视测试完毕！")


if __name__ == "__main__":
    run_diagnostics()