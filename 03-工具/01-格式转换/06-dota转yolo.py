import os
import cv2

folder_path = r"H:\360MoveData\Users\Administrator\Desktop\dota"
output_path = r"H:\360MoveData\Users\Administrator\Desktop\yolo"  # YOLO格式文件路径

def dota_to_yolo(txt_filename, images_path, txt_path, output_path):
    # 读取图片
    image_name = os.path.join(images_path, txt_filename.split(".")[0] + ".jpg")
    image = cv2.imread(image_name)
    img_h, img_w = image.shape[:2]

    # 读取DOTA格式文件
    dota_path = os.path.join(txt_path, txt_filename)
    with open(dota_path, 'r') as file:
        lines = file.readlines()
        yolo_path = os.path.join(output_path, txt_filename)
        with open(yolo_path, 'w') as output:
            for line in lines:
                parts = line.strip().split()

                # 提取DOTA格式的坐标
                x1, y1, x2, y2, x3, y3, x4, y4, category, _ = parts[:10]
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, [x1, y1, x2, y2, x3, y3, x4, y4])

                # 计算矩形边界框
                xmin = min(x1, x2, x3, x4)
                xmax = max(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                ymax = max(y1, y2, y3, y4)

                # 计算中心点和宽高
                x_center = (xmin + xmax) / 2 / img_w
                y_center = (ymin + ymax) / 2 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h

                # 将类别映射到YOLO格式的类别ID（需要根据实际情况修改）
                category_id = 0  # 假设所有对象都属于类别0

                output.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

            print("DOTA to YOLO conversion completed for:", txt_filename)

# 转换函数
def convert_dota_to_yolo(folder_path, output_path):
    images_path = os.path.join(folder_path, "images")
    dota_path = os.path.join(folder_path, "dota")

    # 遍历DOTA格式的文件
    txt_files = [f for f in os.listdir(dota_path) if f.endswith(".txt")]
    for txt_filename in txt_files:
        dota_to_yolo(txt_filename, images_path, dota_path, output_path)

convert_dota_to_yolo(folder_path, output_path)
