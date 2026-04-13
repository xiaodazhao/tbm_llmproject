import os
from dotenv import load_dotenv
load_dotenv()
import time
import traceback
import requests

# =========================================
# 可切换配置
# =========================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ===== Ollama =====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

# ===== Gemini =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/models"
)

# ===== 通用参数 =====
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_RETRY = int(os.getenv("LLM_RETRY", "1"))


# =========================================
# 通用重试包装
# =========================================
def _retry_call(func, *args, retry=1, **kwargs):
    last_err = None
    for i in range(retry + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"⚠️ 第 {i + 1} 次调用失败: {e}")
            if i < retry:
                time.sleep(1.5)
    raise last_err


# =========================================
# Ollama
# =========================================
def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1
        },
        "keep_alive": "10m"
    }

    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()

    # 正常情况下 Ollama chat 返回 data["message"]["content"]
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    raise ValueError(f"Ollama 返回格式异常: {data}")


# =========================================
# Gemini
# =========================================
def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("未设置 GEMINI_API_KEY")

    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1
        }
    }

    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()

    # Gemini 常见返回路径
    candidates = data.get("candidates", [])
    if candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]

    raise ValueError(f"Gemini 返回格式异常: {data}")


# =========================================
# 统一入口
# =========================================
def call_llm(prompt: str) -> str:
    try:
        if LLM_PROVIDER == "ollama":
            return _retry_call(call_ollama, prompt, retry=LLM_RETRY)

        if LLM_PROVIDER == "gemini":
            return _retry_call(call_gemini, prompt, retry=LLM_RETRY)

        raise ValueError(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}")

    except Exception as e:
        print(f"❌ LLM 调用报错: {e}")
        traceback.print_exc()
        return f"[LLM Error] {e}"


# =========================================
# 本地测试入口
# =========================================
if __name__ == "__main__":
    test_prompt = "请只回复：测试成功"

    print(f"当前 LLM_PROVIDER = {LLM_PROVIDER}")
    if LLM_PROVIDER == "ollama":
        print(f"当前 OLLAMA_MODEL = {OLLAMA_MODEL}")
    elif LLM_PROVIDER == "gemini":
        print(f"当前 GEMINI_MODEL = {GEMINI_MODEL}")

    result = call_llm(test_prompt)
    print("\n模型返回：")
    print(result)