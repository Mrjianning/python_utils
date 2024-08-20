import os
import shutil
import random
from tqdm import tqdm

def split_files():
 
    split_ratio = 0.8

    folder_path = r'D:\finallshell下载\person'

    # 获取待拆分文件夹内所有images子文件夹中所有文件的列表
    all_files_images = [file_name for file_name in os.listdir(os.path.join(folder_path, 'images')) if file_name.endswith('.jpg') or file_name.endswith('.png')or file_name.endswith('.JPG')]

    # 获取待拆分文件夹内所有labels子文件夹中所有文件的列表
    all_files_labels = [file_name for file_name in os.listdir(os.path.join(folder_path, 'labels')) if file_name.endswith('.txt')]

    # 计算需要分成训练集和验证集的文件数量
    train_num = int(len(all_files_images) * split_ratio)
    val_num = len(all_files_images) - train_num

    # 设置训练集和验证集的目录路径
    train_folder_images = os.path.join(folder_path, 'images', 'train')
    train_folder_labels = os.path.join(folder_path, 'labels', 'train')
    val_folder_images = os.path.join(folder_path, 'images', 'val')
    val_folder_labels = os.path.join(folder_path, 'labels', 'val')

    # 如果文件夹不存在则新建
    if not os.path.exists(train_folder_images):
        os.makedirs(train_folder_images)
    if not os.path.exists(train_folder_labels):
        os.makedirs(train_folder_labels)
    if not os.path.exists(val_folder_images):
        os.makedirs(val_folder_images)
    if not os.path.exists(val_folder_labels):
        os.makedirs(val_folder_labels)

    # 移动前一部分的images文件到训练集的文件夹下
    for file_name in tqdm(all_files_images[:train_num], desc='Moving training images'):
        # 使用shutil.move方法将文件从原路径移动到目标路径
        shutil.move(os.path.join(folder_path, 'images', file_name), os.path.join(train_folder_images, file_name))

    # 移动后一部分的images文件到验证集的文件夹下
    for file_name in tqdm(all_files_images[train_num:], desc='Moving validation images'):
        # 使用shutil.move方法将文件从原路径移动到目标路径
        shutil.move(os.path.join(folder_path, 'images', file_name), os.path.join(val_folder_images, file_name))

    # 移动前一部分的labels文件到训练集的文件夹下
    for file_name in tqdm(all_files_labels[:train_num], desc='Moving training labels'):
        # 使用shutil.move方法将文件从原路径移动到目标路径
        shutil.move(os.path.join(folder_path, 'labels', file_name), os.path.join(train_folder_labels, file_name.replace('.jpg','.txt').replace('.png','.txt').replace('.JPG','.txt')))

    # 移动后一部分的labels文件到验证集的文件夹下
    for file_name in tqdm(all_files_labels[train_num:], desc='Moving validation labels'):
        # 使用shutil.move方法将文件从原路径移动到目标路径
        shutil.move(os.path.join(folder_path, 'labels', file_name), os.path.join(val_folder_labels, file_name.replace('.jpg','.txt').replace('.png','.txt').replace('.JPG','.txt')))

    print('拆分完成')


# 调用函数执行操作
split_files()
