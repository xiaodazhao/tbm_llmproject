# config.py
from pathlib import Path
import os


def get_data_root() -> Path:
    """
    自动识别 Google Drive 中的 TBM9 根目录
    - Windows:
        G:/我的云端硬盘/TBM9
        G:/My Drive/TBM9
    - Mac:
        ~/Library/CloudStorage/GoogleDrive*/我的云端硬盘/TBM9
        ~/Library/CloudStorage/GoogleDrive*/My Drive/TBM9
    - 找不到时 fallback 到当前项目下的 ./data
    """
    # Windows
    if os.name == "nt":
        candidates = [
            Path("G:/我的云端硬盘/TBM9"),
            Path("G:/My Drive/TBM9"),
        ]
        for p in candidates:
            if p.exists():
                return p

    # macOS
    cloud_base = Path.home() / "Library/CloudStorage"
    if cloud_base.exists():
        drives = list(cloud_base.glob("GoogleDrive*"))
        for drive in drives:
            for root_name in ["我的云端硬盘", "My Drive"]:
                p = drive / root_name / "TBM9"
                if p.exists():
                    return p

    # fallback
    return Path(__file__).resolve().parent / "data"


# =========================
# 根目录
# =========================
DATA_ROOT = get_data_root()

# =========================
# 你的实际目录结构
# =========================
# CSV 数据目录
DATA_DIR = DATA_ROOT / "TBM9_2023"

# PDF 目录
TSP_DIR = DATA_ROOT / "TSP"
HSP_DIR = DATA_ROOT / "HSP"
SKETCH_DIR = DATA_ROOT / "SKETCH"

# 证据库目录
DB_DIR = DATA_ROOT / "DB"

# 输出目录
RESULT_DIR = DATA_ROOT / "result"
LOG_DIR = DATA_ROOT / "logs"
DAILY_RESULT_DIR = DATA_ROOT / "result_daily_twin"

# 自动创建输出目录
for d in [DB_DIR, RESULT_DIR, LOG_DIR, DAILY_RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# evidence_db.csv
EVIDENCE_DB_PATH = DB_DIR / "evidence_db.csv"

# 可选：如果以后还会用 drill
DRILL_DIR = DATA_ROOT / "DRILL"

# 参数
TOLERANCE_M = 3.0
HIGH_RISK_LOOKAHEAD_M = 10.0
NEXT_FORECAST_LOOKAHEAD_M = 5.0