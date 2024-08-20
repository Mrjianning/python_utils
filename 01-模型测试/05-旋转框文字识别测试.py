import cv2
import numpy as np
import base64
import requests
import tkinter as tk
import os

file_name="1"
floader_path=r'H:\360MoveData\Users\Administrator\Desktop\kuaidi_obb\images'
image_path=os.path.join(floader_path,file_name+".JPG")
txt_path=os.path.join(floader_path,file_name+".txt")
output_image_path = r'H:\360MoveData\Users\Administrator\Desktop\kuaidi_obb\result'+file_name+'.JPG' # 保存截取的图片的路径

print(txt_path)
print(image_path)

# 1、截取图片
image = cv2.imread(image_path)  # 替换为你的图片路径

# 从txt文件中读取坐标
with open(txt_path, 'r') as file:
    lines = file.readlines()

    # 解析txt文件中的坐标
    coordinates = [float(value) for value in lines[0].split()[2:]]  # 从第三个值开始读取坐标并转为浮点数
    # print(coordinates)

# 将坐标转换为numpy数组
points = np.array(coordinates, dtype=np.float32).reshape(4, 2)

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
    print("第一种情况：p1_p2<P1_P4")
    # 通过透视变换映射四个点到目标坐标
    matrix = cv2.getPerspectiveTransform(points, target_points)
    result = cv2.warpPerspective(image, matrix, (w, h))  # 设置截取后的宽度和高度

# 第二种情况
else :
    print("第二种情况：p1_p2>P1_P4")
    # 通过透视变换映射四个点到目标坐标
    matrix = cv2.getPerspectiveTransform(points, target_points)
    result = cv2.warpPerspective(image, matrix, (w, h))  # 设置截取后的宽度和高度
    result=cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE )

# 水平镜像
result = cv2.flip(result,1)  # 第二个参数为1表示水平镜像，0表示垂直镜像，-1表示水平和垂直镜像

# 2、发送正反判断post请求
z_or_f=""
url_direction= "http://192.168.1.66:8478/door"
_, image_base64 = cv2.imencode(".jpg", result)
image_base64_str = base64.b64encode(image_base64.tobytes()).decode('utf-8')
res_d = requests.post(url_direction, data=image_base64_str)
if res_d.status_code == 200:
    print("POST request successful")
    res_d = res_d.json()
    for results in res_d["data"]:
        z_or_f=results["class_id"]
    
# 特殊条件需要选择180度
if z_or_f=="fan":
    result=cv2.rotate(result, cv2.ROTATE_180)

cv2.imwrite(output_image_path, result)

# 3、文字识别
_, image_base64 = cv2.imencode(".jpg", result)
image_base64_str = base64.b64encode(image_base64.tobytes()).decode('utf-8')

# 发送文字识别POST请求
url = "http://192.168.1.66:10003/ocr1"
res = requests.post(url, data=image_base64_str)

# 检查响应
if res.status_code == 200:
    print("POST request successful")
    res = res.json()
    # print(res["data"])  # 打印响应内容

    strs=[]
    for results in res["data"]:
        text=str(results["rec_text"])
        strs.insert(0, text)  # 使用insert方法将元素插入到列表的开头

else:
    print(f"POST request failed with status code {res.status_code}")

print(strs)

# 显示截取的图片（可选）
cv2.imshow('Cropped Image', result)

# 创建一个主窗口
root = tk.Tk()
root.title("文本显示面板")

# 创建一个文本框
text_box = tk.Text(root, font=("Arial", 16),width=40, height=8)
text_box.pack()

# 将`strs`中的文本插入文本框
for text in strs:
    text_box.insert("end", text + "\n")

# 运行图形界面主循环
root.mainloop()

cv2.waitKey(0)
cv2.destroyAllWindows()
