import os
import shutil
import tkinter as tk
from tkinter import filedialog

def move_files(source_folder, dest_folder, count):
    # 获取文件列表
    file_list = os.listdir(source_folder)
    # 遍历文件列表，并移动指定数量的文件到目标文件夹
    for i in range(count):
        # 判断是否已经移动完所有文件
        if i >= len(file_list):
            print("文件已经全部移动完成")
            break
        # 获取源文件路径和目标文件路径
        file_path = os.path.join(source_folder, file_list[i])
        dest_path = os.path.join(dest_folder, file_list[i])
        # 移动文件
        shutil.move(file_path, dest_path)
        print("移动文件 {} 到 {} 成功".format(file_path, dest_path))

def select_folders():
    # 创建tkinter窗口
    root = tk.Tk()
    # 隐藏窗口
    root.withdraw()
    # 打开选择文件夹对话框
    source_folder = filedialog.askdirectory(title="选择待移动文件夹")
    dest_folder = filedialog.askdirectory(title="选择目标文件夹")
    # 如果选择了文件夹，则执行移动操作
    if source_folder and dest_folder:
        # 指定要移动的文件数量
        count = 2000
        # 调用函数移动文件
        move_files(source_folder, dest_folder, count)

select_folders()
