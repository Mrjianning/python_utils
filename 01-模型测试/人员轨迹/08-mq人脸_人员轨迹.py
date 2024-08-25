import cv2
import numpy as np
import pika
import threading
import base64
import ast
import time 
import os 
from PIL import Image, ImageDraw, ImageFont
import urllib.parse

import locale
import sys
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
sys.stdout.reconfigure(encoding='utf-8')


# 用于存储每个track_id的轨迹
track_history = {}
track_last_seen = {}
track_name = {}
# 消失判定时间（秒）
disappear_time_threshold = 2.0

class RabbitMqConsumer:
    def __init__(self, queueName, shared_images, index):
        self.queueName = queueName
        self.shared_images = shared_images
        self.index = index
        credentials = pika.PlainCredentials('admin', 'mq@013tech')
        parameters = pika.ConnectionParameters(host='192.168.2.16', credentials=credentials)
        self.connection = pika.BlockingConnection(parameters)

        self.channel = self.connection.channel()

        arguments = {"x-max-length": 5}
        self.channel.queue_declare(queue=queueName,durable=False, arguments=arguments)
        self.channel.basic_consume(queue=queueName, on_message_callback=self.callback, auto_ack=True)
    
    def callback(self, ch, method, properties, body):
        start_time = time.time()  # 记录开始时间
        try:
            result = ast.literal_eval(body.decode('utf-8'))  # 解码消息体并安全地评估
            # print(result["result"])
            image_data = base64.b64decode(result["drawBase64"])

            pred = result["result"]
            # pred={}
            nparr = np.fromstring(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.shared_images[self.index] = [image, pred]
        except ValueError as e:
            print("Error evaluating message:", e)

        end_time = time.time()  # 记录结束时间
        processing_time = (end_time - start_time)*1000  # 计算处理时间
        # print(f"Frame processing time: {processing_time:.4f} seconds")  # 打印处理时间
        
    def start(self):
        thread = threading.Thread(target=self.channel.start_consuming)
        thread.start()

# 保存图像的目录
output_dir = r"C:\Users\admin\Desktop\output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_rec(image, results):
    global track_history, track_last_seen
    current_time = time.time()
    current_track_ids = set()

    r_es_image=image.copy()

       # print(track_history)
    print(track_name)

    for result in results:
        alg_name = result['alg_name']
        data = result['data']
        if alg_name == "NONE":
            return
        
        for res in data:
            # 获取坐标和类别信息
            left, top, right, bottom = int(res['x']), int(res['y']), int(res['x'] + res['width']), int(res['y'] + res['height'])
            class_name = res['class']
            track_id = res['track_id']
            confidence = res['score']
            name=res['info']

            # 将当前track_id加入到集合中，并更新最后检测时间
            current_track_ids.add(track_id)
            track_last_seen[track_id] = current_time
        
           # 如果 track_id 不在 track_name 中，或者当前的值为 "unknown"，则更新值
            if track_id not in track_name or track_name[track_id] == "unknown":
                track_name[track_id] = name

            # 计算目标的中心点坐标
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            # 如果track_id之前没有出现过，初始化它的轨迹列表
            if track_id not in track_history:
                track_history[track_id] = []
            
            # 更新track_id的轨迹
            track_history[track_id].append((center_x, center_y))
            
            # 绘制矩形框
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 4)
  
            # # 准备文本标签（类别和置信度）
            # label = f"{class_name}: {confidence:.2f}"
            # # 绘制文本
            # cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # 绘制轨迹
            # for i in range(1, len(track_history[track_id])):
            #     if track_history[track_id][i - 1] is None or track_history[track_id][i] is None:
            #         continue
            #     cv2.line(image, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 255), 2)
            
            # 绘制当前中心点
            cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

    # 检查哪些track_id超过两秒未更新，视为消失
    for track_id in list(track_history.keys()):
        if track_id not in current_track_ids:
                
            last_seen_time = track_last_seen.get(track_id, 0)
            if current_time - last_seen_time > disappear_time_threshold:

                # 获取最后的图像，并绘制轨迹和track_id
                final_image = r_es_image
                if len(track_history[track_id]) > 0:
                    # 绘制轨迹线
                    for i in range(1, len(track_history[track_id])):
                        if track_history[track_id][i - 1] is None or track_history[track_id][i] is None:
                            continue
                        cv2.line(final_image, track_history[track_id][i - 1], track_history[track_id][i], (0, 0, 255), 2)
                    
                    # 绘制起点和终点
                    start_point = track_history[track_id][0]
                    end_point = track_history[track_id][-1]

                    # 绘制方向箭头（起点）
                    if len(track_history[track_id]) > 1:
                        next_point = track_history[track_id][1]
                        cv2.arrowedLine(final_image, start_point, next_point, (255, 255, 0), 6, tipLength=0.5)
                        # 在起点标上文字
                        cv2.putText(final_image, "Start", (start_point[0], start_point[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    # 绘制方向箭头（终点）
                    if len(track_history[track_id]) > 1:
                        prev_point = track_history[track_id][-2]
                        cv2.arrowedLine(final_image, prev_point, end_point, (0, 255, 255), 6, tipLength=0.5)
                        # 在终点标上文字
                        cv2.putText(final_image, "End", (end_point[0], end_point[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                    # 在终点处绘制track_id
                    # cv2.putText(final_image, f"ID: {track_id} name: {track_name[track_id]}", (end_point[0], end_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 使用 PIL 绘制中文文本
                    pil_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    font = ImageFont.truetype("./uming.ttc", 50)  # 替换为你系统中的中文字体路径

                    text = f"ID: {track_id} name: {track_name[track_id]}运动轨迹"
                    draw.text((20, 150), text, font=font, fill=(0, 255, 0))

                    # 将 PIL 图像转换回 OpenCV 格式
                    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # 保存图像
                timestamp = int(time.time())
                # 先保存为一个临时文件名
                temp_filename = f"temp_{timestamp}.png"
                temp_output_path = os.path.join(output_dir, temp_filename)
                cv2.imwrite(temp_output_path, final_image)

                # 然后重命名文件到带有中文名的文件名
                filename = f"track_{track_id}_{track_name[track_id]}_{timestamp}.png"
                final_output_path = os.path.join(output_dir, filename)
                os.rename(temp_output_path, final_output_path)
                
                # 清除track_history和track_last_seen中该track_id的记录
                del track_history[track_id]
                del track_last_seen[track_id]
                del track_name[track_id]

 
def create_grid_layout(images, n_rows, n_cols):
    target_height = 720     # 调整为合适的尺寸
    target_width = 1280     # 调整为合适的尺寸
    grid_height = target_height * n_rows
    grid_width = target_width * n_cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, (img, pred) in enumerate(images):
        if img is not None:
            # if pred:

            plot_rec(img, pred)

            resized_img = cv2.resize(img, (target_width, target_height))
            row = i // n_cols
            col = i % n_cols
            start_row, start_col = row * target_height, col * target_width
            grid[start_row:start_row+target_height, start_col:start_col+target_width] = resized_img
 
    return grid

def display_images(shared_images):
    cv2.namedWindow('RabbitMQ Images', cv2.WINDOW_NORMAL)
    while True:
        grid = create_grid_layout(shared_images, n_rows, n_cols)
        cv2.imshow('RabbitMQ Images', grid)
        if  cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

# Parameters
queue_names = [
    'test'
]  
n_rows = 1
n_cols = 1

shared_images = [[None, None]] * len(queue_names)

consumers = [RabbitMqConsumer(queue_names[i], shared_images, i) for i in range(len(queue_names))]
for consumer in consumers:
    consumer.start()

# Start a new thread for displaying images
display_thread = threading.Thread(target=display_images, args=(shared_images,))
display_thread.start()

# Wait for the display thread to finish
display_thread.join()

# Close RabbitMQ connections
for consumer in consumers:
    consumer.connection.close()
