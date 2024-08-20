import cv2
import os
import numpy as np

floder_path=r"H:\360MoveData\Users\Administrator\Desktop\yolo"
output_path = r"H:\360MoveData\Users\Administrator\Desktop\yolo"  # 新的文本文件路径

def yolo_to_dota(txt_filename,images_path,txt_path,output_path):

    # 读取图片
    image_name=os.path.join(images_path,txt_filename.split(".")[0]+".jpg")
    print(images_path)
    print(image_name)
    image = cv2.imread(image_name)

    # print(image.shape[1])
    # print(image.shape[0])
    img_w=image.shape[1]
    img_h=image.shape[0]

    
    # 格式转换
    txt_path=os.path.join(txt_path, txt_filename) 
    with open(txt_path, "r") as file: 
        lines = file.readlines()
        output_path=os.path.join(output_path,txt_filename)

        with open(output_path, "w") as output: 
            for line in lines:
                line = line.strip().split()
                # 将 line 转换为 numpy 数组
                line = np.array(line, dtype=np.float64)
                if len(line) == 5:  # 确保每行有5个值
                    category, x, y, width, height = line

                    category="guliao"
                    score=0
                    # 计算四个角点坐标（保持为浮点数）
                    x1 = (x - width/2)* img_w
                    y1 = (y - height/2) * img_h

                    x2 = (x - width/2)* img_w
                    y2 = (y + height/2) * img_h

                    x3 = (x + width/2) * img_w
                    y3 = (y + height/2) * img_h

                    x4 = (x + width/2) * img_w
                    y4 = (y - height/2) * img_h
                    
                    # Write the modified line to the output file
                    output.write(f" {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {category} {score}\n")

            print("yolo_to_dota:",output_path)


# 遍历txt
def convert_txt_files_to_dota(floder_path,output_path):

    images_path=os.path.join(floder_path, "images")
    txt_path=os.path.join(floder_path, "labels")

    # # 遍历文件中txt文件
    txt_files = [f for f in os.listdir(txt_path) if f.endswith(".txt")]

    for txt_filename in txt_files:
        yolo_to_dota(txt_filename,images_path,txt_path,output_path)

convert_txt_files_to_dota(floder_path,output_path)