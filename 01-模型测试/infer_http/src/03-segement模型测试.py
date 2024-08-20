import cv2
import requests
import base64
import time
import numpy as np
from typing import List

colors = [(0, 0, 255), (0, 255, 0)]

# 结果处理
def plot_result(frame, results):
    count=0
    for res in results:
        count=count+1
        class_id = res["class_id"]
        x = int(res["x"])
        y = int(res["y"])
        w = int(res["width"])
        h = int(res["height"])
        seg_data = res["seg_data"]

        # 在图像上绘制边界框和标识
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 绘制物体的轮廓
        if seg_data:
            polygon = np.array(seg_data).reshape((-1, 2)).astype(np.int32)
            # 连接多边形
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

def main():

    # url = r"C:\Users\admin\Desktop\x-anythinglaebl\guliao\11.jpg"
    url = r"C:\Users\admin\Desktop\1.png"
    model_url = "http://192.168.2.16:8800/seg"

    # 读取单张图片
    frame = cv2.imread(url)
    if frame is None:
        print("Failed to read image")
        exit()

    # 获取图片的宽度和高度
    height, width, _ = frame.shape

    # 设置窗口的高度为720
    window_height = 720

    # 计算窗口的宽度，保持图片比例
    window_width = int(width * (window_height / height))

    # 创建窗口
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', window_width, window_height)

    # 编码为二进制数据
    _, img = cv2.imencode('.jpg', frame)

    # 请求模型服务
    try:
        res = requests.post(model_url, data=img.tobytes())
        print(res)
        res = res.json()
        # print(res["data"])
        plot_result(frame, res["data"])
    except requests.RequestException as e:
        print(f"Request to model service failed: {str(e)}")
    

    cv2.imshow('Image', frame)

    # 等待按键事件，按下任意键退出程序
    cv2.waitKey(0)

    # 释放窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
