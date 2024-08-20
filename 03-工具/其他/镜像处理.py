import os
from PIL import Image
from tqdm import tqdm

# 定义要处理的文件夹路径、保存路径和文件名格式
folder_path = r"H:\1"
save_path = r"H:\2"
file_name_format = "{}_mirror.jpg" # 可以根据需要修改文件名格式

# 创建保存路径的文件夹（如果不存在）
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 获取文件夹中所有的图片文件
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]

# 对每个图片文件进行镜像处理，并保存处理后的图片到指定文件夹，并修改文件名
for image_file in tqdm(image_files, desc="Processing images", unit="image"):
    with Image.open(image_file) as img:
        # img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)

        file_name = os.path.splitext(os.path.basename(image_file))[0]
        save_file = os.path.join(save_path, file_name_format.format(file_name))
        img_mirror.save(save_file)
