import os

path = r'C:\Users\admin\Desktop\jt-wu\jiantou-03-ok\images'

img_files = []
txt_files = []

# 遍历文件夹，将图片和 txt 文件分别存入两个列表中
for file_name in os.listdir(path):
    if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.dmp'):
        img_files.append(file_name)
    elif file_name.endswith('.txt'):
        txt_files.append(file_name)

# 遍历图片文件，如果没有对应的 txt 文件，则删除图片文件
for img_file in img_files:
    txt_file = img_file[:-4] + '.txt'
    if txt_file not in txt_files:
        os.remove(os.path.join(path, img_file))
        print(f"{img_file} has been removed")


# 遍历 txt 文件，如果没有对应的图片文件，则删除 txt 文件
for txt_file in txt_files:
    img_file_jpg = txt_file[:-4] + '.jpg'
    img_file_png = txt_file[:-4] + '.png'
    img_file_dmp = txt_file[:-4] + '.dmp'
    if (img_file_jpg not in img_files) and (img_file_png not in img_files) and (img_file_dmp not in img_files):
        os.remove(os.path.join(path, txt_file))
        print(f"{txt_file} has been removed")