# dataprocess.py
import pandas as pd


def _judge_condition(row):
    if "掘进状态" in row.index and pd.notna(row["掘进状态"]):
        status = row["掘进状态"]
        if status == 0:
            return 0
        else:
            thrust = row["推力"] if "推力" in row.index and pd.notna(row["推力"]) else 0
            speed = row["推进速度"] if "推进速度" in row.index and pd.notna(row["推进速度"]) else 0
            torque = row["刀盘扭矩"] if "刀盘扭矩" in row.index and pd.notna(row["刀盘扭矩"]) else 0

            thrust_on = abs(thrust) > 1e-8
            speed_on = abs(speed) > 1e-8
            torque_on = abs(torque) > 1e-8

            if thrust_on and speed_on:
                return 2
            elif thrust_on and (not speed_on):
                return 1
            elif (not thrust_on) and torque_on:
                return 3
            else:
                return 1

    thrust = row["推力"] if "推力" in row.index and pd.notna(row["推力"]) else 0
    speed = row["推进速度"] if "推进速度" in row.index and pd.notna(row["推进速度"]) else 0
    torque = row["刀盘扭矩"] if "刀盘扭矩" in row.index and pd.notna(row["刀盘扭矩"]) else 0

    thrust_on = abs(thrust) > 1e-8
    speed_on = abs(speed) > 1e-8
    torque_on = abs(torque) > 1e-8

    if not thrust_on and not speed_on and not torque_on:
        return 0
    elif thrust_on and not speed_on:
        return 1
    elif thrust_on and speed_on:
        return 2
    elif (not thrust_on) and torque_on:
        return 3
    else:
        return 0


def _condition_code_to_state(code):
    mapping = {
        0: "stop",
        1: "transition",
        2: "work",
        3: "abnormal"
    }
    return mapping.get(code, "unknown")


def _condition_code_to_cn(code):
    mapping = {
        0: "停机",
        1: "启动/过渡",
        2: "稳定掘进",
        3: "异常扭矩"
    }
    return mapping.get(code, "未知")


def load_and_process(source):
    """
    source: csv路径 或 DataFrame
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    if "运行时间-time" not in df.columns:
        raise ValueError("缺少时间列：运行时间-time")

    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = df.dropna(subset=["运行时间-time"]).sort_values("运行时间-time").reset_index(drop=True)

    df["condition_code"] = df.apply(_judge_condition, axis=1)
    df["condition_name"] = df["condition_code"].map(_condition_code_to_state)
    df["group"] = (df["condition_code"] != df["condition_code"].shift()).cumsum()

    segments_df = df.groupby("group").agg(
        start_time=("运行时间-time", "first"),
        end_time=("运行时间-time", "last"),
        condition_code=("condition_code", "first"),
        condition_name=("condition_name", "first")
    ).reset_index(drop=True)

    segments = []
    for _, row in segments_df.iterrows():
        seg = {
            "start": row["start_time"],
            "end": row["end_time"],
            "state": row["condition_name"],
            "state_code": int(row["condition_code"]),
            "duration_sec": (row["end_time"] - row["start_time"]).total_seconds()
        }
        segments.append(seg)

    return segments


def segments_to_text(segments):
    if not segments:
        return "未识别到有效工况段。"

    lines = []
    for s in segments:
        start = s["start"].strftime("%Y-%m-%d %H:%M:%S")
        end = s["end"].strftime("%Y-%m-%d %H:%M:%S")

        dur = s["duration_sec"]
        dur_str = f"{int(dur)} 秒" if dur < 60 else f"{dur / 60:.1f} 分钟"

        state_cn = _condition_code_to_cn(s["state_code"])
        lines.append(f"在 {start} 到 {end} 期间，TBM 处于{state_cn}状态，持续 {dur_str}。")

    return "\n".join(lines)


def compute_stats(segments):
    stop = [x for x in segments if x["state"] == "stop"]
    transition = [x for x in segments if x["state"] == "transition"]
    work = [x for x in segments if x["state"] == "work"]
    abnormal = [x for x in segments if x["state"] == "abnormal"]

    def total(xs):
        return sum(x["duration_sec"] for x in xs)

    def longest(xs):
        return max(xs, key=lambda x: x["duration_sec"]) if xs else None

    return {
        "stop_count": len(stop),
        "transition_count": len(transition),
        "work_count": len(work),
        "abnormal_count": len(abnormal),
        "stop_total_min": total(stop) / 60,
        "transition_total_min": total(transition) / 60,
        "work_total_min": total(work) / 60,
        "abnormal_total_min": total(abnormal) / 60,
        "longest_stop": longest(stop),
        "longest_transition": longest(transition),
        "longest_work": longest(work),
        "longest_abnormal": longest(abnormal),
        "short_stops": [x for x in stop if x["duration_sec"] < 60],
        "short_transitions": [x for x in transition if x["duration_sec"] < 60],
        "short_works": [x for x in work if x["duration_sec"] < 60],
        "short_abnormals": [x for x in abnormal if x["duration_sec"] < 60],
    }


def stats_to_text(stats):
    def fmt_seg(s):
        if not s:
            return "无"
        start = s["start"].strftime("%H:%M:%S")
        end = s["end"].strftime("%H:%M:%S")
        return f"{start}~{end}（约 {s['duration_sec'] / 60:.1f} 分钟）"

    return f"""
停机段数量：{stats['stop_count']}
启动/过渡段数量：{stats['transition_count']}
稳定掘进段数量：{stats['work_count']}
异常扭矩段数量：{stats['abnormal_count']}

总停机时长：{stats['stop_total_min']:.1f} 分钟
总启动/过渡时长：{stats['transition_total_min']:.1f} 分钟
总稳定掘进时长：{stats['work_total_min']:.1f} 分钟
总异常扭矩时长：{stats['abnormal_total_min']:.1f} 分钟

最长停机：{fmt_seg(stats['longest_stop'])}
最长启动/过渡：{fmt_seg(stats['longest_transition'])}
最长稳定掘进：{fmt_seg(stats['longest_work'])}
最长异常扭矩：{fmt_seg(stats['longest_abnormal'])}

短停机（<60s）：{len(stats['short_stops'])} 段
短启动/过渡（<60s）：{len(stats['short_transitions'])} 段
短稳定掘进（<60s）：{len(stats['short_works'])} 段
短异常扭矩（<60s）：{len(stats['short_abnormals'])} 段
""".strip()