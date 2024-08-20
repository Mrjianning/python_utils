import cv2
import requests
import time
import concurrent.futures


# 关键点颜色
pose_kpt_color = [
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [255, 128, 0],
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255],
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255]
]

# 骨架线颜色
pose_limb_color = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [255, 51, 255], [255, 51, 255],
    [255, 51, 255], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [0, 255, 0],
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0]
]

# 骨架线结构
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

class Pose:
    def __init__(self):
        pass
    
    # 绘制关键点和骨架线
    def draw_kpoints_and_limbs(self,kpts, image):
        height, width = image.shape[:2]

        # 绘制关键点
        for i, (x, y, conf) in enumerate(zip(kpts[0::3], kpts[1::3], kpts[2::3])):
            if conf > 0.5:
                R, G, B = pose_kpt_color[i % len(pose_kpt_color)]
                cv2.circle(image, (int(x), int(y)), 3, (R, G, B), -1)
                cv2.putText(image, str(i + 1), (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)

        # 绘制骨架线
        for limb in skeleton:
            p1, p2 = limb[0] - 1, limb[1] - 1
            pos1 = (int(kpts[p1 * 3]), int(kpts[p1 * 3 + 1]))
            pos2 = (int(kpts[p2 * 3]), int(kpts[p2 * 3 + 1]))
            conf1, conf2 = kpts[p1 * 3 + 2], kpts[p2 * 3 + 2]

            if conf1 > 0.2 and conf2 > 0.2:
                R, G, B = pose_limb_color[limb[0] - 1]
                cv2.line(image, pos1, pos2, (R, G, B), 2)


    # 结果处理
    def plot_result(self, frame, results):

       for ibox in results:
        kpts = ibox["kpts"]
        class_id = ibox['class_id']
        x, y, width, height = ibox['x'], ibox['y'], ibox['width'], ibox['height']
        score = ibox['score']

        cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 2)
        caption = f"{class_id} {score:.2f}"
        cv2.putText(frame, caption, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        self.draw_kpoints_and_limbs(kpts, frame)

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

