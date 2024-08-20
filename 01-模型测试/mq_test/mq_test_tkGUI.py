import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import pika
import threading
import base64
import ast

class Application(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master, padding="3 3 12 12")
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()

    def create_widgets(self):
        # 设置默认值
        self.server_ip_var = tk.StringVar(value="192.168.2.12")
        self.queue_names_var = tk.StringVar(value="test")
        self.rows_var = tk.StringVar(value="1")
        self.cols_var = tk.StringVar(value="1")
        self.username_var = tk.StringVar(value="admin")
        self.password_var = tk.StringVar(value="mq@013tech")
        self.image_data_var = tk.StringVar(value="image")

        # 创建输入框和标签
        self.server_ip_label = ttk.Label(self, text="服务器IP:")
        self.server_ip_label.grid(column=0, row=0, sticky=tk.W)
        self.server_ip_entry = ttk.Entry(self, textvariable=self.server_ip_var, justify='center')
        self.server_ip_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))

        self.queue_names_label = ttk.Label(self, text="队列名称 (逗号分隔):")
        self.queue_names_label.grid(column=0, row=1, sticky=tk.W)
        self.queue_names_entry = ttk.Entry(self, textvariable=self.queue_names_var, justify='center')
        self.queue_names_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

        self.rows_label = ttk.Label(self, text="行数:")
        self.rows_label.grid(column=0, row=2, sticky=tk.W)
        self.rows_entry = ttk.Entry(self, textvariable=self.rows_var, justify='center')
        self.rows_entry.grid(column=1, row=2, sticky=(tk.W, tk.E))

        self.cols_label = ttk.Label(self, text="列数:")
        self.cols_label.grid(column=0, row=3, sticky=tk.W)
        self.cols_entry = ttk.Entry(self, textvariable=self.cols_var, justify='center')
        self.cols_entry.grid(column=1, row=3, sticky=(tk.W, tk.E))

        self.username_label = ttk.Label(self, text="用户名:")
        self.username_label.grid(column=0, row=4, sticky=tk.W)
        self.username_entry = ttk.Entry(self, textvariable=self.username_var, justify='center')
        self.username_entry.grid(column=1, row=4, sticky=(tk.W, tk.E))

        self.password_label = ttk.Label(self, text="密码:")
        self.password_label.grid(column=0, row=5, sticky=tk.W)
        self.password_entry = ttk.Entry(self, textvariable=self.password_var, justify='center')
        self.password_entry.grid(column=1, row=5, sticky=(tk.W, tk.E))

        self.image_data_label = ttk.Label(self, text="图像数据键名:")
        self.image_data_label.grid(column=0, row=6, sticky=tk.W)
        self.image_data_combobox = ttk.Combobox(self, textvariable=self.image_data_var, values=["drawBase64", "image"], justify='center')
        self.image_data_combobox.grid(column=1, row=6, sticky=(tk.W, tk.E))
        self.image_data_combobox.current(0)  # 设置默认选择值

        # 创建开始按钮
        self.start_button = ttk.Button(self, text="开始", command=self.start_process)
        self.start_button.grid(column=0, row=7, columnspan=2, sticky=(tk.W, tk.E))

        # 创建退出按钮
        self.quit = ttk.Button(self, text="退出", command=self.master.destroy)
        self.quit.grid(column=0, row=8, columnspan=2, sticky=(tk.W, tk.E))

        # 设置grid布局的权重
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        for row in range(9):
            self.rowconfigure(row, weight=1)

    def start_process(self):
        server_ip = self.server_ip_entry.get()
        queue_names = self.queue_names_entry.get().split(',')
        n_rows = int(self.rows_entry.get())
        n_cols = int(self.cols_entry.get())
        username = self.username_entry.get()
        password = self.password_entry.get()
        image_data_key = self.image_data_combobox.get()

        shared_images = [[None, None]] * len(queue_names)
        try:
            consumers = [RabbitMqConsumer(queue_names[i], shared_images, i, server_ip, username, password, image_data_key) for i in range(len(queue_names))]
            for consumer in consumers:
                consumer.start()
        except Exception as e:
            messagebox.showerror("错误", f"连接到服务器时出错: {e}")
            return

        display_thread = threading.Thread(target=display_images, args=(shared_images, n_rows, n_cols))
        display_thread.start()

class RabbitMqConsumer:
    def __init__(self, queueName, shared_images, index, server_ip, username, password, image_data_key):
        self.queueName = queueName
        self.shared_images = shared_images
        self.index = index
        self.image_data_key = image_data_key
        credentials = pika.PlainCredentials(username, password)
        parameters = pika.ConnectionParameters(host=server_ip, credentials=credentials)
        try:
            self.connection = pika.BlockingConnection(parameters)
        except pika.exceptions.AMQPConnectionError as e:
            messagebox.showerror("错误", f"连接到RabbitMQ服务器失败: {e}")
            raise
        self.channel = self.connection.channel()

        arguments = {"x-max-length": 5}
        self.channel.queue_declare(queue=queueName, durable=False, arguments=arguments)
        self.channel.basic_consume(queue=queueName, on_message_callback=self.callback, auto_ack=True)

    def callback(self, ch, method, properties, body):
        try:
            result = ast.literal_eval(body.decode('utf-8'))
            image_data = base64.b64decode(result[self.image_data_key])
            pred = {}
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.shared_images[self.index] = [image, pred]
        except ValueError as e:
            print("Error evaluating message:", e)

    def start(self):
        thread = threading.Thread(target=self.channel.start_consuming)
        thread.start()

def create_grid_layout(images, n_rows, n_cols):
    target_height = 720
    target_width = 1280
    grid_height = target_height * n_rows
    grid_width = target_width * n_cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, (img, pred) in enumerate(images):
        if img is not None:
            resized_img = cv2.resize(img, (target_width, target_height))
            row = i // n_cols
            col = i % n_cols
            start_row, start_col = row * target_height, col * target_width
            grid[start_row:start_row+target_height, start_col:start_col+target_width] = resized_img

    return grid

def display_images(shared_images, n_rows, n_cols):
    cv2.namedWindow('RabbitMQ Images', cv2.WINDOW_NORMAL)
    while True:
        grid = create_grid_layout(shared_images, n_rows, n_cols)
        cv2.imshow('RabbitMQ Images', grid)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("RabbitMQ 图像处理系统")
    root.geometry("600x400")
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()
