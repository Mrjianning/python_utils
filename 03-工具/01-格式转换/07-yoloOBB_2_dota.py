import os

def yolo_to_dota(yolo_annotation, labels):
    parts = yolo_annotation.split()
    label_index = int(parts[0])
    coordinates = parts[1:]
    class_name = labels[label_index]
    dota_format = " ".join(coordinates) + f" {class_name} 0\n"
    return dota_format

def convert_file(yolo_file_path, dota_file_path, labels):
    with open(yolo_file_path, 'r') as file:
        lines = file.readlines()

    dota_lines = [yolo_to_dota(line.strip(), labels) for line in lines]

    with open(dota_file_path, 'w') as file:
        file.writelines(dota_lines)

    print(f"文件转换完成，保存到: {dota_file_path}")

def convert_folder(yolo_folder_path, dota_folder_path, labels):
    if not os.path.exists(dota_folder_path):
        os.makedirs(dota_folder_path)

    for filename in os.listdir(yolo_folder_path):
        if filename.endswith('.txt'):
            yolo_file_path = os.path.join(yolo_folder_path, filename)
            dota_file_path = os.path.join(dota_folder_path, filename)
            convert_file(yolo_file_path, dota_file_path, labels)

# 标签数组，索引 0 对应 'guliao'
labels = ['guliao']

# YOLO 格式的文件夹路径和 DOTA 格式的文件夹路径
yolo_folder_path = r'H:\360MoveData\Users\Administrator\Desktop\demo6\labels'
dota_folder_path = r'H:\360MoveData\Users\Administrator\Desktop\demo6\dota'

# 执行转换
convert_folder(yolo_folder_path, dota_folder_path, labels)
