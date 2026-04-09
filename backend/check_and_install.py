# check_and_install.py
import importlib
import subprocess
import sys

PACKAGE_MAP = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "dotenv": "python-dotenv",
    "fitz": "PyMuPDF",
    "pydantic": "pydantic",
}

def ensure_package(import_name: str, pip_name: str):
    try:
        importlib.import_module(import_name)
        print(f"[OK] {import_name}")
    except ImportError:
        print(f"[MISSING] {import_name} -> 安装 {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

def main():
    for import_name, pip_name in PACKAGE_MAP.items():
        ensure_package(import_name, pip_name)

    print("全部依赖检查完成。")

if __name__ == "__main__":
    main()