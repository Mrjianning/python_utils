
import os 
import shutil 

# 遍历指定目录下的所有图片文件，并复制到指定目录 
def copy_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # 创建目录
    for root, dirs, files in os.walk(source_dir): # 使用os.walk()函数遍历目录下的所有文件和文件夹 
        for file_name in files: # 遍历所有的文件 
            if file_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')): # 判断是否为图片文件 
                source_file_path = os.path.join(root, file_name) # 拼接出图片文件的绝对路径 
                target_file_path = os.path.join(target_dir, file_name) # 拼接出目标路径下的文件的绝对路径 
                shutil.copyfile(source_file_path, target_file_path) # 复制图片到目标路径下 
                print("success")

# 指定源目录和目标目录 
source_dir = r'H:\datasets\10.抽烟\Smoke-Detect-by-yolov5_v2\data' # 源目录 
target_dir = r'H:\datasets\10.抽烟\Smoke-Detect-by-yolov5_v2\smoke' # 目标目录 

# 复制所有图片到目标路径下 
copy_images(source_dir, target_dir)
