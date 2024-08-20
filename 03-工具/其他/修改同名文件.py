
# import os

# search_dir = ''
# add_char = '_'

# # 遍历目录中所有文件
# for root, dirs, files in os.walk(search_dir):
#     for f in files:
#         # 如果文件名既以 .jpg 结尾，又以 .png 结尾
#         if f.endswith('.jpeg') and os.path.exists(os.path.join(root, f[:-4]+'.png')): 
#             print(f"Found same name files: {os.path.join(root,f)} and {os.path.join(root,f[:-4]+'.png')}")
#             # 修改同名的 png 文件
#             new_png_path = os.path.join(root, f[:-4]+add_char+'.png')
#             os.rename(os.path.join(root, f[:-4]+'.png'), new_png_path)
#             print(f"Modified {new_png_path}")



import os

# 指定目录路径
dir_path = "H:\人工智能数据集\CVAT数据集\mask-300\obj_train_data"

# 获取指定目录下的所有文件
file_list = os.listdir(dir_path)

# 循环遍历文件列表
for file_name in file_list:
    # 获取文件名和扩展名
    file, ext = os.path.splitext(file_name)
    # 如果当前文件的扩展名是 jpg、jpeg、png，则检查同名文件是否存在
    if ext[1:].lower() in ['jpg', 'jpeg', 'png']:
        # 遍历同名文件
        for same_name_file in file_list:
            # 如果同名文件的扩展名也是jpg、jpeg、png，且文件名与当前文件的文件名相同，则给同名文件添加后缀
            if os.path.splitext(same_name_file)[1][1:].lower() in ['jpg', 'jpeg', 'png'] and os.path.splitext(same_name_file)[0] == file and same_name_file != file_name:
                os.rename(os.path.join(dir_path, same_name_file), os.path.join(dir_path, f"{file}_{os.path.splitext(same_name_file)[1][1:].lower()}{os.path.splitext(same_name_file)[1]}"))

# 输出处理完成的信息
print("文件名处理完成！")
