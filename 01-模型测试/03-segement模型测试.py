import cv2
import requests
import base64
import time
import numpy as np
from typing import List

colors = [(0, 0, 255), (0, 255, 0)]
# polygons = []

# 减少多边形点
def reduce_polygon(polygon: np.ndarray, angle_th: float = 0, distance_th: float = 0) -> np.ndarray:
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = []
        for i in range(0, len(polygon) - 2, 2):

            v01 = polygon[i - 1] - polygon[i]
            v12 = polygon[i] - polygon[i + 1]

            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)

            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i)
                continue  

            angle = np.arccos(np.sum(v01 * v12) / (d01 * d12))
            if angle < angle_th_rad:
                points_removed.append(i)

        polygon = np.delete(polygon, points_removed, axis=0)
    
    return polygon

# 二进制掩码转多边形
def mask_to_polygon(mask: np.array, min_area_threshold=100, report: bool = False, w=int, h=int, x=int, y=int, seg_w=int, seg_h=int) -> List[int]:
    
    # CHAIN_APPROX_TC89_KCOS
    # CHAIN_APPROX_TC89_L1
    # CHAIN_APPROX_SIMPLE
    # CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area_threshold:  # 只处理面积大于等于阈值的多边形
            # 映射坐标点并添加到多边形列表
            contour = contour.reshape(-1, 2)
            contour[:, 0] = (contour[:, 0] * (w / seg_w) + x).astype(int)
            contour[:, 1] = (contour[:, 1] * (h / seg_h) + y).astype(int)
            polygons.append(contour.ravel().tolist())

    polygons = reduce_polygon(polygons, angle_th=1, distance_th=2)

    if report:
        print(f"Number of points = {len(polygons[0])}")

    
    return polygons

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
        # seg_h = int(res["seg_h"])
        # seg_w = int(res["seg_w"])
        seg_data = res["seg_data"]

       
        # 在图像上绘制边界框和标识
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 绘制物体的轮廓
        if seg_data:
            # # 将 mask 数据转换为 NumPy 数组
            # seg_data = np.array(seg_data).reshape((seg_h, seg_w))
            # seg_data = (seg_data * 255).astype(np.uint8)

            # # 二进制掩码转多边形
            # polygons = mask_to_polygon(seg_data, report=True, w=w, h=h, x=x, y=y, seg_w=seg_w, seg_h=seg_h)
            # print(seg_data)

            polygon = np.array(seg_data).reshape((-1, 2)).astype(np.int32)

            # 连接多边形
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            # 绘制多边形
            # for polygon in seg_data:
            #     polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)

            #     # 连接多边形
            #     cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            #     color = (0, 255, 0)
            #     # cv2.fillPoly(frame, [polygon], color)  # 填充多边形区域

    print("数量：",count)    

def main():

    # 配置参数
    # url = r"rtsp://192.168.1.3/105"
    # url = r"C:\Users\admin\Desktop\shitou\1.jpg"
    # model_url = "http://192.168.2.16:8999/segment"
    
    # # 视频捕获初始化
    # cap = cv2.VideoCapture(url)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     return

    # # 创建窗口
    # cv2.namedWindow('IP Camera', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('IP Camera', 1280, 720)

    # # 处理视频流
    # while True:
    #     ret, frame = cap.read()
        
    #     if not ret:
    #         print("Failed to retrieve frame")
    #         break

    #     img = cv2.imencode('.jpg', frame)[1]
    #     img = str(base64.b64encode(img))[2:-1]
        
    #     # 请求模型服务
    #     try:
    #         res = requests.post(model_url, data=img)
    #         res = res.json()
    #         print(res["data"])
    #         plot_result(frame, res["data"])
    #     except requests.RequestException as e:
    #         print(f"Request to model service failed: {str(e)}")

    #     cv2.imshow('IP Camera', frame)

    #     # 按下ESC键退出程序
    #     if cv2.waitKey(1)==27:
    #         break

    # # 释放资源
    # cap.release()
    # cv2.destroyAllWindows()

    url = r"C:\Users\admin\Desktop\x-anythinglaebl\guliao\1.jpg"
    model_url = "http://192.168.2.16:8999/guliao"

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
