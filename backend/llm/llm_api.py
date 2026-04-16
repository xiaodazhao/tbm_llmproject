import os
from dotenv import load_dotenv
from google import genai

# =========================================================
# 1. 初始化配置
# =========================================================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ 未找到 GOOGLE_API_KEY，请检查 .env 文件！")

client = genai.Client(api_key=API_KEY)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


# =========================================================
# 2. 通用 LLM 调用函数
# =========================================================
def call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    调用 Google Gemini 生成文本
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 8192
            }
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "⚠️ 模型返回了空内容"

    except Exception as e:
        print(f"❌ LLM 调用报错: {e}")
        return f"[LLM Error] {e}"


# =========================================================
# 3. RAG 专用调用
# =========================================================
def call_llm_rag(query: str, context: str, model: str = DEFAULT_MODEL) -> str:
    """
    RAG 调用封装：自动把 context 和 query 拼接
    """
    prompt = f"""
你是一名专业的 TBM（隧道掘进机）施工数据分析师。
请根据以下【监测数据背景】回答【分析任务】。

【监测数据背景】
{context}

【分析任务】
{query}

要求：
1. 语言专业、客观、工程化。
2. 若存在异常停机、频繁启停、状态波动或气体异常，应重点指出。
3. 严格依据输入数据，不得虚构。
4. 控制在 2000 字以内。
"""
    return call_llm(prompt, model=model)