#time_window_utils.py
import pandas as pd
from pathlib import Path


def load_df_by_time(source, start_time, end_time):
    """
    source: 可以是 csv路径(str/Path) 或者 已经读取好的 DataFrame
    start_time, end_time: 可被 pandas.to_datetime 解析的时间字符串
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, Path)):
        try:
            df = pd.read_csv(source)
        except FileNotFoundError:
            print(f"❌ 文件未找到: {source}")
            return pd.DataFrame()
    else:
        print("❌ source 类型错误，必须是 DataFrame 或路径字符串")
        return pd.DataFrame()

    # 兼容时间列命名
    if "运行时间-time" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "运行时间-time"})
        else:
            print("❌ 数据中缺少 '运行时间-time' 列")
            return pd.DataFrame()

    # 时间列转 datetime
    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = (
        df.dropna(subset=["运行时间-time"])
        .sort_values("运行时间-time")
        .reset_index(drop=True)
    )

    if df.empty:
        print("❌ 数据为空或时间列无有效值")
        return pd.DataFrame()

    # 解析时间窗
    s_time = pd.to_datetime(start_time, errors="coerce")
    e_time = pd.to_datetime(end_time, errors="coerce")

    if pd.isna(s_time) or pd.isna(e_time):
        print(f"❌ 时间解析失败: start={start_time}, end={end_time}")
        return pd.DataFrame()

    if s_time > e_time:
        print(f"❌ 时间范围非法: start={start_time} > end={end_time}")
        return pd.DataFrame()

    mask = (
        (df["运行时间-time"] >= s_time) &
        (df["运行时间-time"] <= e_time)
    )

    filtered_df = df.loc[mask].reset_index(drop=True)

    print(
        f"🔍 时间窗筛选: {s_time.strftime('%Y-%m-%d %H:%M:%S')} ~ "
        f"{e_time.strftime('%Y-%m-%d %H:%M:%S')} | 结果: {len(filtered_df)} 条数据"
    )

    return filtered_df