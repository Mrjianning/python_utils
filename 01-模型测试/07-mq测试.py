import cv2
import numpy as np
import pika
import threading
import base64
import ast
import time 


class RabbitMqConsumer:
    def __init__(self, queueName, shared_images, index):
        self.queueName = queueName
        self.shared_images = shared_images
        self.index = index
        credentials = pika.PlainCredentials('admin', 'mq@013tech')
        parameters = pika.ConnectionParameters(host='192.168.2.12', credentials=credentials)
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

            # if result["drawBase64"]=="null":
            #     image_data = base64.b64decode(result["image"])
            # else :
            #     image_data = base64.b64decode(result["drawBase64"])    

            # pred = result["result"]
            pred={}
            nparr = np.fromstring(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.shared_images[self.index] = [image, pred]
        except ValueError as e:
            print("Error evaluating message:", e)

        end_time = time.time()  # 记录结束时间
        processing_time = (end_time - start_time)*1000  # 计算处理时间
        print(f"Frame processing time: {processing_time:.4f} seconds")  # 打印处理时间
        
    def start(self):
        thread = threading.Thread(target=self.channel.start_consuming)
        thread.start()

def plot_rec(image, results):
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
            # print("========",track_id)
            confidence = res['score']
            if class_name == "person":
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 4)
            elif class_name == "hat":
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 4)
            elif class_name == "no_hat":
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 4)
            
            # 准备文本标签（类别和置信度）
            label = f"{class_name}: {confidence:.2f}"
            # 绘制文本
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def create_grid_layout(images, n_rows, n_cols):
    target_height = 720     # 调整为合适的尺寸
    target_width = 1280     # 调整为合适的尺寸
    grid_height = target_height * n_rows
    grid_width = target_width * n_cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, (img, pred) in enumerate(images):
        if img is not None:
            if pred:
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
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Parameters
queue_names = [
    # 'rtsp://admin:abcd1234@192.168.2.103:554/h265/ch1/main/video',
    # 'rtsp://admin:abcd1234@192.168.2.104:554/h265/ch1/main/video',
    # 'rtsp://admin:abcd1234@192.168.2.105:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.106:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.107:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.108:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.109:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.110:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.111:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.126:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.160:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.161:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.162:554/h265/ch2/main/av_stream',
    'test',
    # 'rtsp://admin:abcd1234@192.168.2.164:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.165:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.166:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.167:554/h265/ch2/main/av_stream',
    # 'rtsp://admin:abcd1234@192.168.2.168:554/h265/ch2/main/av_stream',
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
