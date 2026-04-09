# io_utils.py
from pathlib import Path
import glob
import pandas as pd

from config import DATA_DIR, EVIDENCE_DB_PATH
from geology.geology_fusion_backend import load_evidence_db


def get_all_csv_paths() -> list[Path]:
    """
    获取 DATA_DIR 下所有 csv 文件
    """
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    return [Path(f) for f in files]


def get_latest_csv_path() -> Path:
    """
    获取最新一个 csv 文件（按文件名排序）
    """
    files = get_all_csv_paths()
    if not files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下没有找到任何 csv 文件")
    return max(files)


def get_csv_path_by_date(date_str: str) -> Path:
    """
    按日期获取 csv 文件路径
    约定文件名格式：tbm_data_YYYYMMDD.csv
    例如：2023-12-30 -> tbm_data_20231230.csv
    """
    date_compact = date_str.replace("-", "")
    file_path = DATA_DIR / f"tbm_data_{date_compact}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")
    return file_path


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    读取单个 csv，并标准化时间列
    """
    df = pd.read_csv(file_path)

    if "运行时间-time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "运行时间-time"})

    if "运行时间-time" not in df.columns:
        raise ValueError("缺少时间列：运行时间-time")

    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = (
        df.dropna(subset=["运行时间-time"])
        .sort_values("运行时间-time")
        .reset_index(drop=True)
    )
    return df


def load_latest_csv() -> tuple[Path, pd.DataFrame]:
    """
    读取最新一个 csv
    返回：(路径, DataFrame)
    """
    path = get_latest_csv_path()
    df = load_csv(path)
    return path, df


def load_csv_by_date(date_str: str) -> tuple[Path, pd.DataFrame]:
    """
    按日期读取 csv
    返回：(路径, DataFrame)
    """
    path = get_csv_path_by_date(date_str)
    df = load_csv(path)
    return path, df


def load_evidence() -> pd.DataFrame:
    """
    读取 evidence_db.csv
    """
    if not EVIDENCE_DB_PATH.exists():
        raise FileNotFoundError(
            f"找不到 evidence_db.csv：{EVIDENCE_DB_PATH}\n"
            f"请先运行：python scripts/build_evidence_db.py"
        )
    return load_evidence_db(EVIDENCE_DB_PATH)


def check_data_environment() -> dict:
    """
    检查当前数据环境，方便调试
    """
    csv_files = get_all_csv_paths()
    return {
        "DATA_DIR": str(DATA_DIR),
        "EVIDENCE_DB_PATH": str(EVIDENCE_DB_PATH),
        "data_dir_exists": DATA_DIR.exists(),
        "evidence_exists": EVIDENCE_DB_PATH.exists(),
        "csv_count": len(csv_files),
        "latest_csv": str(max(csv_files)) if csv_files else None,
    }