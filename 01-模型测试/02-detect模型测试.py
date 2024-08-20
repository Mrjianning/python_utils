import cv2
import requests
import base64
import time
import numpy as np

colors = [(0, 0, 255), (0, 255, 0)]

# 结果处理
def plot_result(frame, results):
    for res in results:

        class_id = res["class_id"]
        x = int(res["x"])
        y = int(res["y"])
        w = int(res["width"])
        h = int(res["height"])
        score=round(res["score"],2)

        text=str(class_id)+":"+str(score)

        # 在图像上绘制边界框和标识
        cv2.rectangle(frame, (int(x), int(y)), (int(w + x), int(h + y)), (255, 0, 0))
        cv2.putText(frame, str(text), (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    # 配置参数
    url = "rtsp://admin:abcd1234@192.168.1.105/smart265/ch1/sub/av_stream"
    model_url = "http://192.168.1.5:9999/car1"
    
    # 视频捕获初始化
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # 创建窗口
    cv2.namedWindow('IP Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('IP Camera', 1280, 720)

    # 处理视频流
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to retrieve frame")
            break

        img = cv2.imencode('.jpg', frame)[1]
        img = str(base64.b64encode(img))[2:-1]
        
        # 请求模型服务
        try:
            res = requests.post(model_url, data=img)
            res = res.json()
            
            plot_result(frame, res["data"])
        except requests.RequestException as e:
            print(f"Request to model service failed: {str(e)}")

        cv2.imshow('IP Camera', frame)

        # 按下ESC键退出程序
        if cv2.waitKey(1)==27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
