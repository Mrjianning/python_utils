import cv2
import requests
import time
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 初始化配置信息
class Config:
    def __init__(self):
        self.COUNT_DICT = {}
        self.count_enter = 0
        self.count_out = 0
        self.count_all = 0

class Tracker:
    def __init__(self):
        pass

    # 计算点是否在直线的一侧
    @staticmethod
    def bool_enter_out(point0, point1, point2):
        # 使用线性代数的方法，计算点 point0 相对于通过 point1 和 point2 所定义的直线的位置
        # 直线方程为：Ax + By + C = 0
        # 其中 A = (point1[1] - point2[1]), B = (point2[0] - point1[0]), C = -(point2[0] * point1[1] - point1[0] * point2[1])
        
        # temp 计算的是点 point0 代入直线方程后的结果
        # 如果 temp > 0，说明 point0 在直线的一侧 （比如右侧）
        # 如果 temp < 0，说明 point0 在直线的另一侧 （比如坐侧）
        # 如果 temp = 0，说明 point0 刚好在直线上

        temp = (point1[1] - point2[1]) * point0[0] + (point2[0] - point1[0]) * point0[1] - point2[0] * point1[1] + point1[0] * point2[1]
        return temp

    # 统计进出数量
    def count_enter_out(self, t_id, temp, conf):
        # 如果该跟踪ID (t_id) 不在 COUNT_DICT 中，说明这是第一次检测到这个对象
        # 将当前的 temp 值（表示对象相对于直线的位置）存储在 COUNT_DICT 中
        if t_id not in conf.COUNT_DICT:
            conf.COUNT_DICT[t_id] = temp
        
        # 如果该对象已经存在于 COUNT_DICT 中，则进行以下检查
        else:
            # 检查对象是否从负值跨越到正值
            # 这意味着对象从直线的一侧（负侧）移动到了另一侧（正侧），表示对象离开了指定区域
            if conf.COUNT_DICT[t_id] < 0 and temp > 0:
                conf.count_out += 1  # 增加离开计数
                conf.count_all -= 1  # 减少当前区域内的总人数计数
                conf.COUNT_DICT[t_id] = temp  # 更新该对象的最新位置状态
            
            # 检查对象是否从正值跨越到负值
            # 这意味着对象从直线的正侧移动到了负侧，表示对象进入了指定区域
            elif conf.COUNT_DICT[t_id] > 0 and temp < 0:
                conf.count_enter += 1  # 增加进入计数
                conf.count_all += 1    # 增加当前区域内的总人数计数
                conf.COUNT_DICT[t_id] = temp  # 更新该对象的最新位置状态


    # 结果处理
    def plot_result(self, frame, results, conf):
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
            cv2.putText(frame, txt, (int(x + w / 2), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)

            # 定义一条线段，用于判断对象是进入还是离开
            start_point = (int(frame.shape[1] * 0.5), int(0))
            end_point = (int(frame.shape[1] * 0.5), int(frame.shape[0]))
            # 计算对象的中心点 (point_0)，即边界框的中心位置
            point_0 = (int(x + (w / 2)), int(y + (h / 2)))
            cv2.circle(frame, point_0, 4, (0, 255, 0), 2)

            # 计算对象中心点相对于中线的相对位置
            # 使用self.bool_enter_out函数判断中心点是否在直线的某一侧
            temp = self.bool_enter_out((x + (w / 2), y + 10), start_point, end_point)
            # 根据对象相对直线的位置更新进入/离开计数
            self.count_enter_out(t_id, temp, conf)

    # 在图像上绘制直线
    @staticmethod
    def plot_line(img, start_point, end_point):
        cv2.line(img, start_point, end_point, (0, 0, 255), 2)

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

    @staticmethod
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
    def process_video(self, model_url, video_queue, visualization_queue):
        conf = Config()
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
                    self.plot_result(frame, res["data"], conf)

                # 画线和标记计数
                start_point = (int(frame.shape[1] * 0.5), int(0))
                end_point = (int(frame.shape[1] * 0.5), int(frame.shape[0]))
                self.plot_line(frame, start_point, end_point)

                font_path = './fonts/uming.ttc'  # 替换为你自己的中文字体路径
                font_size = 80  # 字体大小
                color = (255, 0, 0)  # 文本颜色（RGB 格式）

                # 添加中文文本
                text = f'人进: {conf.count_enter}, 人出: {conf.count_out}, 库内人数: {conf.count_all}'
                frame = self.put_chinese_text(frame, text, (0, 200), font_path, font_size, color)

                visualization_queue.put(frame)

            except requests.RequestException as e:
                print(f"Request to model service failed: {str(e)}")

        while True:
            frame = video_queue.get()

            _, img_data = cv2.imencode('.jpg', frame)
            size_in_mb = len(img_data) / (1024 * 1024)
            # print(size_in_mb, "MB")

            executor.submit(process_request, img_data, frame)  # 使用线程池异步处理请求

