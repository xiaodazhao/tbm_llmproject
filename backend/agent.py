# agent.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import re
import traceback
from typing import Callable, Dict, Any, List, Optional

import pandas as pd

from app import get_latest_df
from analysis.dataprocess import load_and_process, compute_stats, stats_to_text, segments_to_text
from analysis.excavation_state import (
    detect_excavation_state,
    excavation_state_segments,
    explain_excavation_states,
    excavation_state_to_text,
    excavation_state_efficiency,
    excavation_state_stats,
    excavation_state_stats_to_text,
)
from analysis.gas_analysis import compute_gas_stats, gas_stats_to_text
from analysis.forward_risk_advisor import generate_forward_risk_summary, forward_risk_to_text

from geology.geology_fusion_backend import attach_geology_labels, load_evidence_db
from geology.geology_summary import (
    summarize_geology_record_level,
    summarize_geology_segment_level,
    geology_summary_to_text,
)
from geology.segment_analysis import run_segment_analysis, build_typical_segments_table

from config import EVIDENCE_DB_PATH
from llm.llm_api import call_llm


STATE_FEATURES = ("推力", "刀盘扭矩", "刀盘实际转速", "推进速度")

# 固定必须先跑的工具
MANDATORY_TOOLS = ["basic_analysis", "forward_risk"]

# 当基础工况满足这些条件时，强制补跑状态分析
FORCE_STATE_STOP_TOTAL_MIN = 600
FORCE_STATE_ABNORMAL_COUNT = 3


# =========================================================
# 打印辅助
# =========================================================
def print_title(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subtitle(title: str):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def preview_text(text: str, max_len: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text)
    return text[:max_len] + ("\n...(已截断)" if len(text) > max_len else "")


def serialize_for_json(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


# =========================================================
# 与 app.py 一致的状态识别辅助逻辑
# =========================================================
def estimate_valid_samples(df: pd.DataFrame, feature_cols=STATE_FEATURES) -> int:
    valid_mask = pd.Series(True, index=df.index)

    if "掘进状态" in df.columns:
        valid_mask &= (pd.to_numeric(df["掘进状态"], errors="coerce").fillna(0) != 0)
    else:
        temp_mask = pd.Series(False, index=df.index)
        if "推力" in df.columns:
            temp_mask |= (pd.to_numeric(df["推力"], errors="coerce").fillna(0).abs() > 1e-8)
        if "推进速度" in df.columns:
            temp_mask |= (pd.to_numeric(df["推进速度"], errors="coerce").fillna(0).abs() > 1e-8)
        valid_mask &= temp_mask

    for col in feature_cols:
        if col in df.columns:
            valid_mask &= pd.to_numeric(df[col], errors="coerce").notna()

    return int(valid_mask.sum())


def choose_state_params(n_valid: int):
    if n_valid < 5:
        return {"do_cluster": False, "n_states": 0, "min_duration_sec": 0}
    elif n_valid < 10:
        return {"do_cluster": True, "n_states": 2, "min_duration_sec": 0}
    elif n_valid < 30:
        return {"do_cluster": True, "n_states": 3, "min_duration_sec": 20}
    else:
        return {"do_cluster": True, "n_states": 4, "min_duration_sec": 60}


# =========================================================
# 工具定义
# =========================================================
def tool_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    segments = load_and_process(df)
    stats = compute_stats(segments)
    return {
        "tool_name": "basic_analysis",
        "segments_count": len(segments),
        "stats": serialize_for_json(stats),
        "stats_text": stats_to_text(stats),
        "seg_text_preview": preview_text(segments_to_text(segments), 1800),
    }


def tool_state_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    n_valid = estimate_valid_samples(df, STATE_FEATURES)
    state_cfg = choose_state_params(n_valid)

    if not state_cfg["do_cluster"]:
        return {
            "tool_name": "state_analysis",
            "valid_samples": n_valid,
            "state_config": state_cfg,
            "state_labels": {},
            "state_text": "有效样本过少，未进行施工状态识别。",
            "efficiency_table": [],
            "state_stats_text": "有效样本过少，无施工状态统计。"
        }

    df_state, _ = detect_excavation_state(
        df.copy(),
        features=STATE_FEATURES,
        n_states=state_cfg["n_states"]
    )

    state_labels = explain_excavation_states(df_state)
    state_segments = excavation_state_segments(df_state, min_duration_sec=state_cfg["min_duration_sec"])
    state_text = excavation_state_to_text(state_segments, state_labels)

    raw_eff_df = excavation_state_efficiency(df_state)
    state_stats = excavation_state_stats(df_state, state_segments)
    state_stats_text = excavation_state_stats_to_text(state_stats, state_labels)

    eff_preview = []
    if not raw_eff_df.empty:
        eff_preview = raw_eff_df.reset_index().to_dict(orient="records")

    return {
        "tool_name": "state_analysis",
        "valid_samples": n_valid,
        "state_config": state_cfg,
        "state_labels": serialize_for_json(state_labels),
        "state_segments": serialize_for_json(state_segments),
        "state_text": preview_text(state_text, 1800),
        "efficiency_table": serialize_for_json(eff_preview),
        "state_stats": serialize_for_json(state_stats),
        "state_stats_text": preview_text(state_stats_text, 1800),
    }


def tool_geology_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    evidence_df = load_evidence_db(EVIDENCE_DB_PATH)
    df_geo = attach_geology_labels(df.copy(), evidence_df)

    geo_summary_record = summarize_geology_record_level(df_geo)
    segment_df = run_segment_analysis(df_geo, segment_length=10)
    typical_segments_df = build_typical_segments_table(segment_df, top_n=10)
    geo_summary_segment = summarize_geology_segment_level(segment_df)
    geo_text = geology_summary_to_text(geo_summary_segment)

    return {
        "tool_name": "geology_analysis",
        "record_summary": serialize_for_json(geo_summary_record),
        "segment_summary": serialize_for_json(geo_summary_segment),
        "geo_text": preview_text(geo_text, 1800),
        "segment_table_preview": serialize_for_json(
            segment_df.head(10).to_dict(orient="records") if not segment_df.empty else []
        ),
        "typical_segments_preview": serialize_for_json(
            typical_segments_df.head(10).to_dict(orient="records") if not typical_segments_df.empty else []
        ),
    }


def tool_forward_risk(df: pd.DataFrame) -> Dict[str, Any]:
    evidence_df = load_evidence_db(EVIDENCE_DB_PATH)

    try:
        df_geo = attach_geology_labels(df.copy(), evidence_df)
    except Exception:
        df_geo = df.copy()

    summary = generate_forward_risk_summary(
        df_plc=df_geo,
        evidence_df=evidence_df,
        lookahead_m=30
    )

    return {
        "tool_name": "forward_risk",
        "summary": serialize_for_json(summary),
        "forward_risk_text": preview_text(forward_risk_to_text(summary), 1800),
    }


def tool_gas_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    gas_stats = compute_gas_stats(df)
    gas_text = gas_stats_to_text(gas_stats)

    return {
        "tool_name": "gas_analysis",
        "gas_stats": serialize_for_json(gas_stats),
        "gas_text": preview_text(gas_text, 1800),
    }


TOOLS: Dict[str, Callable[[pd.DataFrame], Dict[str, Any]]] = {
    "basic_analysis": tool_basic_analysis,
    "state_analysis": tool_state_analysis,
    "geology_analysis": tool_geology_analysis,
    "forward_risk": tool_forward_risk,
    "gas_analysis": tool_gas_analysis,
}


# =========================================================
# LLM 输出解析
# =========================================================
def extract_json_from_text(text: str) -> dict:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        snippet = match.group(0)
        return json.loads(snippet)

    raise ValueError(f"无法解析 LLM JSON 输出：{text[:500]}")


# =========================================================
# 历史摘要
# =========================================================
def summarize_tool_result_for_llm(tool_name: str, result: Dict[str, Any]) -> str:
    if tool_name == "basic_analysis":
        stats = result.get("stats", {})
        return (
            f"基础工况结果：停机段 {stats.get('stop_count', 0)} 段，"
            f"稳定掘进段 {stats.get('work_count', 0)} 段，"
            f"异常扭矩段 {stats.get('abnormal_count', 0)} 段，"
            f"总停机时长 {round(stats.get('stop_total_min', 0), 1)} 分钟，"
            f"总稳定掘进时长 {round(stats.get('work_total_min', 0), 1)} 分钟。"
        )

    if tool_name == "state_analysis":
        return (
            f"施工状态识别结果：有效样本 {result.get('valid_samples', 0)} 个，"
            f"识别配置 {result.get('state_config', {})}，"
            f"状态标签 {result.get('state_labels', {})}。"
        )

    if tool_name == "geology_analysis":
        seg_sum = result.get("segment_summary", {})
        return (
            f"地质分析结果：是否有地质信息 {seg_sum.get('has_geology', False)}，"
            f"高风险区段数 {seg_sum.get('high_risk_segment_count', 0)}，"
            f"多源关注区段数 {seg_sum.get('multi_source_segment_count', 0)}。"
        )

    if tool_name == "forward_risk":
        summary = result.get("summary", {})
        return (
            f"前方风险结果：前方提示段 {summary.get('forward_segment_count', 0)} 个，"
            f"高风险段 {summary.get('high_risk_count', 0)} 个，"
            f"多源共同关注 {summary.get('multi_source_count', 0)}，"
            f"建议级别 {summary.get('advice_level', 'unknown')}。"
        )

    if tool_name == "gas_analysis":
        gas_stats = result.get("gas_stats", {})
        return f"气体分析结果：气体字段统计键数 {len(gas_stats) if isinstance(gas_stats, dict) else 0}。"

    return f"{tool_name} 已执行。"


def build_history_for_llm(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "暂无历史工具调用结果。"

    lines = []
    for i, item in enumerate(history, start=1):
        lines.append(
            f"第{i}轮：调用工具 {item['tool_name']}；"
            f"原因：{item.get('reason', '')}；"
            f"结果摘要：{item.get('result_summary', '')}"
        )
    return "\n".join(lines)


# =========================================================
# 规则层：强制步骤与条件触发
# =========================================================
def get_next_mandatory_tool(used_tools: List[str]) -> Optional[str]:
    for tool_name in MANDATORY_TOOLS:
        if tool_name not in used_tools:
            return tool_name
    return None


def get_basic_result_from_history(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in history:
        if item.get("tool_name") == "basic_analysis":
            return item.get("raw_result", {})
    return None


def should_force_state_analysis(history: List[Dict[str, Any]], used_tools: List[str]) -> bool:
    if "state_analysis" in used_tools:
        return False

    basic_result = get_basic_result_from_history(history)
    if not basic_result:
        return False

    stats = basic_result.get("stats", {})
    stop_total_min = float(stats.get("stop_total_min", 0) or 0)
    abnormal_count = int(stats.get("abnormal_count", 0) or 0)

    return (
        stop_total_min > FORCE_STATE_STOP_TOTAL_MIN
        or abnormal_count >= FORCE_STATE_ABNORMAL_COUNT
    )


# =========================================================
# 多轮 Agent 决策 Prompt
# =========================================================
def build_agent_decision_prompt(
    user_query: str,
    df: pd.DataFrame,
    history: List[Dict[str, Any]],
    used_tools: List[str],
    max_steps: int,
) -> str:
    cols_preview = df.columns[:30].tolist()
    time_info = "未知"
    if "运行时间-time" in df.columns:
        t = pd.to_datetime(df["运行时间-time"], errors="coerce").dropna()
        if not t.empty:
            time_info = f"{t.min()} ~ {t.max()}"

    history_text = build_history_for_llm(history)

    return f"""
你是 TBM 多轮分析 Agent。
你的任务是根据用户问题，逐步选择最合适的下一个工具。
你一次只能做一件事：要么调用一个工具，要么停止。
你不负责生成最终报告，只负责决定下一步分析动作。

【用户问题】
{user_query}

【当前数据概况】
- 数据条数: {len(df)}
- 列数: {len(df.columns)}
- 前30列: {cols_preview}
- 时间范围: {time_info}

【可用工具】
1. basic_analysis
   用途：基础工况分段、停机/过渡/稳定掘进/异常扭矩统计

2. state_analysis
   用途：施工状态聚类识别、状态分段、状态效率统计

3. geology_analysis
   用途：地质融合、风险区段分析、典型区段提取

4. forward_risk
   用途：基于当前掘进位置提取前方一定距离范围内的风险提示

5. gas_analysis
   用途：气体监测统计与异常提示

【已经调用过的工具】
{used_tools if used_tools else "无"}

【历史调用结果摘要】
{history_text}

【决策规则】
- 仅在已经掌握足够信息时才允许停止
- 不要重复调用已经调用过的工具
- 优先先做粗分析，再做补充分析
- 如果用户问风险或前方情况，通常需要 forward_risk
- 如果需要解释风险区段来源，可补充 geology_analysis
- 如果用户问气体相关，再调用 gas_analysis
- 如果历史结果里显示信息已经充分，也可以停止
- 当前最多总共允许 {max_steps} 轮，避免无意义循环

【输出要求】
只输出 JSON，不要输出解释，不要输出 markdown，不要输出代码块。

如果决定调用工具，输出：
{{
  "action": "tool_call",
  "tool": "geology_analysis",
  "reason": "一句简短原因"
}}

如果决定停止，输出：
{{
  "action": "stop",
  "reason": "一句简短原因"
}}
""".strip()


# =========================================================
# 单步决策
# =========================================================
def llm_decide_next_action(
    user_query: str,
    df: pd.DataFrame,
    history: List[Dict[str, Any]],
    used_tools: List[str],
    max_steps: int,
) -> Dict[str, Any]:
    prompt = build_agent_decision_prompt(
        user_query=user_query,
        df=df,
        history=history,
        used_tools=used_tools,
        max_steps=max_steps,
    )

    print_subtitle("调用 LLM 进行下一步决策")
    print(f"Prompt 长度：{len(prompt)}")
    print("\n[决策 Prompt 预览]")
    print(preview_text(prompt, 1200))

    llm_output = call_llm(prompt)

    print("\n[LLM 原始输出]")
    print(preview_text(llm_output, 1000))

    if isinstance(llm_output, str) and llm_output.startswith("[LLM Error]"):
        print("⚠️ LLM 调用失败，自动停止")
        return {
            "action": "stop",
            "reason": "LLM调用失败，提前终止"
        }

    try:
        parsed = extract_json_from_text(llm_output)
    except Exception as e:
        print(f"⚠️ JSON解析失败：{e}")
        return {
            "action": "stop",
            "reason": "LLM输出无法解析，提前终止"
        }

    action = parsed.get("action")
    reason = parsed.get("reason", "")

    if action == "tool_call":
        tool = parsed.get("tool")
        if tool not in TOOLS:
            print(f"⚠️ LLM 返回无效工具：{tool}")
            return {
                "action": "stop",
                "reason": f"LLM返回无效工具 {tool}，提前终止"
            }
        return {
            "action": "tool_call",
            "tool": tool,
            "reason": reason,
        }

    if action == "stop":
        return {
            "action": "stop",
            "reason": reason,
        }

    print(f"⚠️ LLM 返回无效 action：{parsed}")
    return {
        "action": "stop",
        "reason": "LLM返回无效动作，提前终止"
    }


# =========================================================
# 工具执行
# =========================================================
def execute_tool(tool_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    print_subtitle(f"执行工具：{tool_name}")
    result = TOOLS[tool_name](df)
    print("✅ 工具执行完成")
    return result


def print_tool_result(tool_name: str, result: Dict[str, Any]):
    print_title(f"工具结果：{tool_name}")

    for k, v in result.items():
        if k == "tool_name":
            continue

        if isinstance(v, (dict, list)):
            try:
                text = json.dumps(v, ensure_ascii=False, indent=2)
            except Exception:
                text = str(v)
            print(f"\n[{k}]")
            print(preview_text(text, 2200))
        else:
            print(f"\n[{k}]")
            print(preview_text(v, 2200))


# =========================================================
# 多轮 Agent 主循环
# =========================================================
def tbm_multi_agent_no_report(
    df: pd.DataFrame,
    user_query: str,
    max_steps: int = 6,
) -> Dict[str, Any]:
    print_title("TBM 多轮 Agent：规则约束 + LLM 决策 + 工具调用，不生成报告")

    history: List[Dict[str, Any]] = []
    used_tools: List[str] = []
    stop_reason = ""

    for step in range(1, max_steps + 1):
        print_title(f"第 {step} 轮")

        # 1) 先看是否有固定必跑工具
        next_mandatory = get_next_mandatory_tool(used_tools)
        if next_mandatory is not None:
            decision = {
                "action": "tool_call",
                "tool": next_mandatory,
                "reason": f"固定步骤：先执行 {next_mandatory}"
            }
            print(f"🧩 规则层决定：{decision}")

        # 2) mandatory 完成后，如果基础工况异常明显，强制补跑状态分析
        elif should_force_state_analysis(history, used_tools):
            decision = {
                "action": "tool_call",
                "tool": "state_analysis",
                "reason": (
                    f"基础工况触发条件：总停机时间 > {FORCE_STATE_STOP_TOTAL_MIN} 分钟 "
                    f"或异常段数 >= {FORCE_STATE_ABNORMAL_COUNT}"
                )
            }
            print(f"🧩 规则层决定：{decision}")

        # 3) 其余情况再交给 LLM
        else:
            decision = llm_decide_next_action(
                user_query=user_query,
                df=df,
                history=history,
                used_tools=used_tools,
                max_steps=max_steps,
            )

        if decision["action"] == "stop":
            stop_reason = decision.get("reason", "LLM 判断信息已足够。")
            print(f"\n🛑 Agent 停止：{stop_reason}")
            break

        tool_name = decision["tool"]

        if tool_name in used_tools:
            stop_reason = f"工具 {tool_name} 已调用过，为避免重复，停止。"
            print(f"\n🛑 Agent 停止：{stop_reason}")
            break

        result = execute_tool(tool_name, df)
        print_tool_result(tool_name, result)

        result_summary = summarize_tool_result_for_llm(tool_name, result)

        history.append({
            "step": step,
            "tool_name": tool_name,
            "reason": decision.get("reason", ""),
            "result_summary": result_summary,
            "raw_result": result,
        })
        used_tools.append(tool_name)

    else:
        stop_reason = f"达到最大轮数 {max_steps}，自动停止。"

    print_title("多轮 Agent 运行结束")
    print(f"已调用工具：{used_tools}")
    print(f"停止原因：{stop_reason}")

    return {
        "ok": True,
        "used_tools": used_tools,
        "history": history,
        "stop_reason": stop_reason,
    }


# =========================================================
# 主程序入口
# =========================================================
if __name__ == "__main__":
    try:
        print("🚀 启动 TBM 多轮 Agent（不生成报告）...")

        df = get_latest_df()
        print(f"✅ 数据加载完成，共 {len(df)} 条")

        user_query = "帮我分析今天施工情况和前方风险，必要时补充状态分析，但不要生成报告"

        output = tbm_multi_agent_no_report(
            df=df,
            user_query=user_query,
            max_steps=6,
        )

        print("\n✅ 程序运行结束")
        print(f"返回状态：{output['ok']}")
        print(f"已调用工具：{output['used_tools']}")
        print(f"停止原因：{output['stop_reason']}")

    except Exception as e:
        print("\n❌ 程序报错")
        print(str(e))
        print("\n完整报错栈：")
        traceback.print_exc()