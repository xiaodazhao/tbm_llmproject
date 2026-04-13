from llm.llm_api import call_llm

if __name__ == "__main__":
    text = call_llm("你好，请只回复：测试成功")
    print(text)