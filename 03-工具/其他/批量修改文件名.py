import os
from tqdm import tqdm

path = r"H:\下载数据集\02-安全帽\02-已标注\helmat02\xml"
prefix = "helmat02"

# 获取目录下的所有文件
file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# 添加进度条
for filename in tqdm(file_list):
    new_name = prefix + filename
    try:
        os.rename(os.path.join(path, filename), os.path.join(path, new_name))
        # print("修改成功：", os.path.join(path, filename), " --> ", os.path.join(path, new_name))
    except Exception as e:
        print("修改失败：", os.path.join(path, filename), " --> ", os.path.join(path, new_name), "，错误信息：", e)
