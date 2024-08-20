import os
from tkinter import filedialog, Tk

# 弹窗选择文件夹路径
folder_path = r'H:\人工智能数据集\CVAT数据集\合浦车人\晚上\石膏左外\chanche-01\obj_train_data'

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件
for file in files:
    # 获取文件名和后缀名
    file_name, file_ext = os.path.splitext(file)
    
    # 替换日和月为字幕d和m
    new_file_name = file_name.replace('日', 'd').replace('月', 'm')
    
    # 删除文件名中的空格
    new_file_name = new_file_name.replace(' ', '')
    
    # 拼接新的文件名和后缀名
    new_file_name_with_ext = new_file_name + file_ext
    
    # 重命名文件
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name_with_ext))
    
print('文件名替换完成！')
