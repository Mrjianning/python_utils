
import os
import re
import tkinter as tk
from tkinter import filedialog

# 创建Tkinter窗口
root = tk.Tk()
root.withdraw()

# 弹出选择文件夹对话框
folder_path = filedialog.askdirectory()

# 匹配月和日的正则表达式
date_pattern = r"\d{1,2}[月/]\d{1,2}[日/]?"
date_regex = re.compile(date_pattern)

# 遍历文件夹中的文件
for i, filename in enumerate(os.listdir(folder_path)):
    # 从文件名中删除月和日
    new_filename = date_regex.sub("", filename)
    # 构造旧文件路径和新文件路径
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_filename)
    # 重命名文件
    os.rename(old_path, new_path)
    print(f"重命名文件 {filename} 为 {new_filename}")

