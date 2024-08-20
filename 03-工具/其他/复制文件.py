import os
import shutil
from tqdm import tqdm

# 源文件夹列表
source_folders = ['H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d1-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d2-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d3-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d4-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d5-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d6-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d7-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d8-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d9-ok',
                  'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\已纠错\images-d10-ok',
                 ]

# 目标文件夹
destination_folder = r'H:\人工智能数据集\yolov5-6.1\datasets\人员安全\smoking\smoking'

# 遍历源文件夹列表
for source_folder in source_folders:
    # 遍历源文件夹中的所有子目录和文件
    for root, dirs, files in os.walk(source_folder):
        # 构建目标子目录路径
        relative_path = os.path.relpath(root, source_folder)
        destination_path = os.path.join(destination_folder, relative_path)

        # 如果目标子目录不存在，则创建它
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # 复制所有文件到目标文件夹
        for file in tqdm(files, desc=f'Copying files from {source_folder}'):
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(destination_path, file)
            shutil.copy(source_file_path, destination_file_path)

print('所有文件和目录复制完成！')
