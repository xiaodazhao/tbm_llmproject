from pathlib import Path

# 定义源目录和输出文件
source_dir = Path(r"C:\Users\22923\Desktop\LLM_20260409\backend")
output_file = Path(r"C:\Users\22923\Desktop\merged_code_context.txt")


def export_code_to_txt():
    # 1. 使用 'w' 模式打开目标 txt，确保编码为 utf-8 以支持中文注释
    with output_file.open("w", encoding="utf-8") as f_out:
        
        # 2. rglob("*.py") 会自动深入所有子文件夹寻找 .py 文件
        # sorted() 确保输出的文件顺序是整齐的
        for py_file in sorted(source_dir.rglob("*.py")):
            
            # 过滤掉一些不需要的文件夹（比如 python 自动生成的缓存）
            if "__pycache__" in str(py_file):
                continue
                
            # 3. 准备标题：使用文件的绝对路径（地址）
            header = f"\n{'='*20}\n地址: {py_file}\n{'='*20}\n"
            f_out.write(header)
            
            # 4. 读取文件内容并写入
            try:
                # 即使是读代码，也建议加 errors="ignore"，防止极个别特殊字符导致崩溃
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                f_out.write(content)
                f_out.write("\n\n") # 文件间留白
                print(f"已处理: {py_file.name}")
            except Exception as e:
                f_out.write(f"读取失败，错误原因: {e}\n")

if __name__ == "__main__":
    export_code_to_txt()
    print(f"\n🎉 大功告成！所有代码已汇总至: {output_file}")