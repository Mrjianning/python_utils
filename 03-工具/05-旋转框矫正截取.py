import cv2
import numpy as np
import os

folder_path = r'H:\360MoveData\Users\Administrator\Desktop\kuaidi'
output_folder = r'H:\360MoveData\Users\Administrator\Desktop\kuaidi\result'
txt_path=""


# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(folder_path):
    if file_name.endswith('.JPG') or file_name.endswith('.jpg'):
        file_path = os.path.join(folder_path, file_name)
        out_path=os.path.join(output_folder, file_name)
        txt_path = os.path.join(folder_path, file_name.split(".")[0]+".txt")

        # 读取你的图片
        image = cv2.imread(file_path)  # 替换为你的图片路径

        # 从txt文件中读取坐标
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            
            if not lines:
                print("文件为空")
                # 文件为空，退出循环
                continue  

            # 解析txt文件中的坐标
            coordinates = [float(value) for value in lines[0].split()[2:]]  # 从第三个值开始读取坐标并转为浮点数

            # 将坐标转换为numpy数组
            points = np.array(coordinates, dtype=np.float32).reshape(4, 2)
            coordinates=[]
            # print(points)

            # 计算长度1：第一个点到第二个点的距离
            p1_p2 = np.linalg.norm(points[1] - points[0])
            # 计算长度2：第一个点到第四个点的距离
            P1_P4 = np.linalg.norm(points[3] - points[0])

            h=p1_p2
            w=P1_P4

            # 定义截取区域的目标坐标
            target_points = np.array([
                [0, 0],   # 左上角
                [0, h],  # 设置截取后的高度  左下角的坐标
                [w, h],  # 设置截取后的宽度和高度  右下角
                [w, 0]  # 设置截取后的宽度  右上角
            ], dtype=np.float32)

            # 第一种情况
            if(p1_p2<P1_P4):
                # print("第一种情况：p1_p2<P1_P4")
                # 通过透视变换映射四个点到目标坐标
                matrix = cv2.getPerspectiveTransform(points, target_points)
                result = cv2.warpPerspective(image, matrix, (w, h))  # 设置截取后的宽度和高度

            # 第二种情况
            else :
                # print("第二种情况：p1_p2>P1_P4")
                # 通过透视变换映射四个点到目标坐标
                matrix = cv2.getPerspectiveTransform(points, target_points)
                result = cv2.warpPerspective(image, matrix, (w, h))  # 设置截取后的宽度和高度
                result=cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE )

            # 水平镜像
            result = cv2.flip(result,1)  # 第二个参数为1表示水平镜像，0表示垂直镜像，-1表示水平和垂直镜像

            cv2.imwrite(out_path, result)
            print("保存成功",file_path)

