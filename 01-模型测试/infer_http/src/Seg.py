import cv2
import requests
import time
import concurrent.futures
import numpy as np
from typing import List

class Seg:
    def __init__(self):
        pass

    # 结果处理
    def plot_result(self, frame, results):
        
        for res in results:
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
        
    # 视频捕获线程
    @staticmethod
    def video_capture_thread(cap, video_queue, target_fps):
        frame_interval = 1.0 / target_fps  # 每帧间隔时间
        last_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to retrieve frame")
                break

            current_time = time.time()
            elapsed_time = current_time - last_time
            if elapsed_time < frame_interval:
                time.sleep(frame_interval - elapsed_time)
            last_time = current_time

            if not video_queue.full():
                video_queue.put(frame)

    # 可视化线程
    @staticmethod
    def visualization_thread(visualization_queue):
        while True:
            frame = visualization_queue.get()
            cv2.imshow('IP Camera', frame)

            # 按下ESC键退出程序
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    # 处理视频流线程
    def process_video(self, model_url, video_queue, visualization_queue):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)  # 使用线程池进行并发请求

        def process_request(img_data, frame):
            try:
                start_t = time.time()
                res = requests.post(model_url, data=img_data.tobytes(), headers={'Content-Type': 'image/jpeg'})
                end_t = time.time()
                dt = (end_t - start_t) * 1000
                print("网络请求耗时: {:.2f} 毫秒".format(dt))   

                res = res.json()

                if res["data"]:
                    self.plot_result(frame, res["data"])

                visualization_queue.put(frame)

            except requests.RequestException as e:
                print(f"Request to model service failed: {str(e)}")

        while True:
            frame = video_queue.get()
            _, img_data = cv2.imencode('.jpg', frame)
            executor.submit(process_request, img_data, frame)  # 使用线程池异步处理请求

