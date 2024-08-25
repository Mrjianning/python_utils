import cv2
import requests
import time
import queue
import threading
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random

# 初始化配置信息
class Config:
    def __init__(self):
        self.COUNT_DICT = {}
        self.count_enter = 0
        self.count_out = 0
        self.count_all = 0

# 计算点是否在直线的一侧
def bool_enter_out(point0, point1, point2):
    temp = (point1[1] - point2[1]) * point0[0] + (point2[0] - point1[0]) * point0[1] - point2[0] * point1[1] + point1[0] * point2[1]
    return temp

# 统计进出数量
def count_enter_out(t_id, temp, conf):
    if t_id not in conf.COUNT_DICT:
        conf.COUNT_DICT[t_id] = temp
    else:
        if conf.COUNT_DICT[t_id] < 0 and temp > 0:
            conf.count_out += 1
            conf.count_all -= 1
            conf.COUNT_DICT[t_id] = temp
        elif conf.COUNT_DICT[t_id] > 0 and temp < 0:
            conf.count_enter += 1
            conf.count_all += 1
            conf.COUNT_DICT[t_id] = temp

# 结果处理
def plot_result(frame, results, conf):
    for res in results:
        class_id = res["class_id"]
        t_id = res["track_id"]
        x = res["x"]
        y = res["y"]
        w = res["width"]
        h = res["height"]
        score = res.get("score", 0)
        txt = f"{class_id}:{t_id}:{score:.2f}"

        # 在图像上绘制边界框和标识
        cv2.rectangle(frame, (int(x), int(y)), (int(w + x), int(h + y)), (255, 0, 0), 2)
        cv2.putText(frame, txt, (int(x + w / 2), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        # 截取边界框内容并且保存，根据t_id创建文件夹存储对应id的图片，图片命名格式是{t_id}_c{cameraID}s{sequenceNumber}_{frameNumber}_{detectedID}.jpg
        cameraID = 102
        detectedID = 0  # 可以根据需要设定detectedID

        # 随机生成 sequenceNumber 和 frameNumber
        sequenceNumber = random.randint(1, 9999)
        frameNumber = int(time.time() * 1000)  # 使用时间戳的毫秒部分生成唯一的frameNumber

        # 截取边界框内容
        cropped_frame = frame[int(y):int(y+h), int(x):int(x+w)]

        # 创建文件夹（如果不存在）
        save_dir = fr"C:/Users/admin/Desktop/output/{t_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存图像，命名格式：{t_id}_c{cameraID}s{sequenceNumber}_{frameNumber}_{detectedID}.jpg
        img_name = f"{t_id}_c{cameraID}s{sequenceNumber}_{frameNumber}_{detectedID}.jpg"
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, cropped_frame)

        print(f"Saved: {save_path}")

        # 判断进出
        start_point = (int(frame.shape[1] * 0.5), int(0))
        end_point = (int(frame.shape[1] * 0.5), int(frame.shape[0]))
        point_0 = (int(x + (w / 2)), int(y + (h / 2)))
        cv2.circle(frame, point_0, 4, (0, 255, 0), 1)
        temp = bool_enter_out((x + (w / 2), y + 10), start_point, end_point)
        count_enter_out(t_id, temp, conf)

# 在图像上绘制直线
def plot_line(img, start_point, end_point):
    cv2.line(img, start_point, end_point, (0, 0, 255), 2)

# 视频捕获线程
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
def visualization_thread(visualization_queue):
    while True:
        frame = visualization_queue.get()
        cv2.imshow('IP Camera', frame)

        # 按下ESC键退出程序
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

def put_chinese_text(image, text, position, font_path, font_size, color):
    # 将 OpenCV 图像转换为 PIL 图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 将 PIL 图像转换回 OpenCV 图像
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image

# 处理视频流线程
def process_video(model_url, video_queue, visualization_queue):
    conf = Config()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)  # 使用线程池进行并发请求

    def process_request(img_data, frame):
        try:
            start_t = time.time()
            res = requests.post(model_url, data=img_data.tobytes(), headers={'Content-Type': 'image/jpeg'})
            end_t = time.time()
            dt = (end_t - start_t) * 1000
            # print("网络请求耗时: {:.2f} 毫秒".format(dt))   

            res = res.json()

            if res["data"]:
                plot_result(frame, res["data"], conf)

            # 画线和标记计数
            start_point = (int(frame.shape[1] * 0.5), int(0))
            end_point = (int(frame.shape[1] * 0.5), int(frame.shape[0]))
            plot_line(frame, start_point, end_point)

            # font_path = './fonts/uming.ttc'  # 替换为你自己的中文字体路径
            # font_size = 80  # 字体大小
            # color = (0, 0, 255)  # 文本颜色（BGR 格式）

            # # 添加中文文本
            # text = f'人进: {conf.count_enter}, 人出: {conf.count_out}, 库内人数: {conf.count_all}'
            # frame = put_chinese_text(frame, text, (0, 200), font_path, font_size, color)

            cv2.putText(frame, f'人进: {conf.count_enter}, 人出: {conf.count_out}', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            visualization_queue.put(frame)

        except requests.RequestException as e:
            print(f"Request to model service failed: {str(e)}")

    while True:
        frame = video_queue.get()

        _, img_data = cv2.imencode('.jpg', frame)
        size_in_mb = len(img_data) / (1024 * 1024)
        # print(size_in_mb, "MB")

        executor.submit(process_request, img_data, frame)  # 使用线程池异步处理请求

def main():
    url = r"rtsp://admin:abcd1234@192.168.2.102:554/h265/ch1/main/video"
    model_url = "http://192.168.2.16:8800/person_tracker" 
    target_fps = 25  # 目标帧率

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    video_queue = queue.Queue(maxsize=10)
    visualization_queue = queue.Queue(maxsize=10)

    video_thread = threading.Thread(target=video_capture_thread, args=(cap, video_queue, target_fps))
    video_thread.daemon = True
    video_thread.start()

    process_thread = threading.Thread(target=process_video, args=(model_url, video_queue, visualization_queue))
    process_thread.daemon = True
    process_thread.start()

    cv2.namedWindow('IP Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('IP Camera', 1280, 720)
    visualization_thread(visualization_queue)

if __name__ == "__main__":
    main()
