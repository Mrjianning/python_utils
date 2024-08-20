import os
import glob
from pathlib import Path
from tqdm import tqdm

def file_write_jpg(path, jpg_file):
    filenames = set(os.listdir(path)) # 去重
    with open(jpg_file, 'w') as f:
        for filename in tqdm(filenames): # 添加进度条
            suffix = os.path.splitext(filename)[1].lower()
            if suffix in ('.jpg', '.png', '.jpeg', '.bmp'):  # 写入的文件后缀
                out_path = os.path.join('data', 'obj_train_data', filename) #输出特殊命名/相对位置
                f.write(out_path + '\n')

def main(files_root_path, train_txt_path, img_mod=["jpg"]):
    jpg_file = os.path.join(train_txt_path, 'train.txt')
    # 获取所有的txt文件
    txt_file_lists = glob.glob(os.path.join(files_root_path, "*.txt"))  # 返回完整路径
    txt_name_lists = [Path(t).stem for t in txt_file_lists]
    # 获取所有的jpg文件
    img_file_lists = []
    for img_t in img_mod:
        img_file_lists += glob.glob(os.path.join(files_root_path, f"*.{img_t}"))

    img_name_lists = [Path(t).stem for t in img_file_lists]

    if len(txt_name_lists) == len(img_name_lists):
        file_write_jpg(files_root_path, jpg_file)
    else:
        print("创建txt文件...........")
        for txt in set(img_name_lists) - set(txt_name_lists):
            # 创建空的txt文件
            txt_file = os.path.join(files_root_path, f"{txt}.txt")
            open(txt_file, "w+").close()
            print("创建", txt_file)
    
        file_write_jpg(files_root_path, jpg_file)

    # 添加进度条
    print("正在生成train.txt文件...")
    with open(jpg_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            pass
    print("train.txt文件已生成！")

if __name__ == "__main__":
    files_root_path = r"H:\人工智能数据集\CVAT数据集\mask\5000\obj_train_data"
    train_txt_path = r"H:\人工智能数据集\CVAT数据集\mask\5000"

    # files_root_path = r"H:\人工智能数据集\CVAT数据集\human\human-200\obj_train_data"
    # train_txt_path = r"H:\人工智能数据集\CVAT数据集\human\human-200"

    main(files_root_path, train_txt_path, img_mod=["jpg", "png", "jpeg", "bmp"])  # img_mod是图片文件的后缀名 
