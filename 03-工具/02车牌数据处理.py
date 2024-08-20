import os
import glob

# 指定要遍历的文件夹路径
train_path = r"/home/dell/paddlePaddle/PaddleOCR-release-2.6/datasets/car_data/train"
txt_path=r"/home/dell/paddlePaddle/PaddleOCR-release-2.6/datasets/car_data/train"

# 对应的txt文件
txt_file = os.path.join(txt_path, f"rec_train.txt")

# 遍历文件夹内的所有jpg文件
for jpg_file in glob.glob(os.path.join(train_path, "*.jpg")):
    # 获取jpg文件名（不带路径和扩展名）
    file_name = os.path.splitext(os.path.basename(jpg_file))[0]

    # 将jpg文件名写入txt文件 
    with open(txt_file, "a") as f:
        f.write("train/"+f"{file_name}.jpg" + '\t' + f"{file_name}\n")
        print("train/"+f"{file_name}.jpg" + '\t' + f"{file_name}")    
