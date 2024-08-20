import cv2
import requests
import base64
import time
import numpy as np
import math

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

# 定义一个空的手部关键点映射字典
hand_mapped = {}
body_mapped = {}

# 骨架线结构
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

#  姿态--拜
def pose_bai(kpts,image):
    # 1、计算角度
    # 1_1计算点7、9、11连起来的角度 ---- 向量内积---右手角度
    vec_97 = [hand_mapped[9]['x'] - hand_mapped[7]['x'], hand_mapped[9]['y'] - hand_mapped[7]['y']]
    vec_911 = [hand_mapped[9]['x'] - hand_mapped[11]['x'], hand_mapped[9]['y'] - hand_mapped[11]['y']]
    angle_right = math.degrees(np.arccos(np.dot(vec_97, vec_911) / (np.linalg.norm(vec_97) * np.linalg.norm(vec_911))))
    # print('右手角度为:', angle_right)

    # 1_2 左手角度
    vec_86 = [hand_mapped[8]['x'] - hand_mapped[6]['x'], hand_mapped[8]['y'] - hand_mapped[6]['y']]
    vec_810 = [hand_mapped[8]['x'] - hand_mapped[10]['x'], hand_mapped[8]['y'] - hand_mapped[10]['y']]
    angle_left = math.degrees(np.arccos(np.dot(vec_86, vec_810) / (np.linalg.norm(vec_86) * np.linalg.norm(vec_810))))
    # print('左手角度为:', angle_left)

    # 2、计算距离
    # 2.1 获取点10和点11的坐标--手掌距离
    x10, y10 = hand_mapped[10]['x'], hand_mapped[10]['y']
    x11, y11 = hand_mapped[11]['x'], hand_mapped[11]['y']
    distance = math.sqrt((x11 - x10)**2 + (y11 - y10)**2)
    # print('手掌距离为:', distance)
    cv2.putText(image, "hand="+str(distance), (int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 2.2 获取点7和点6的坐标--肩膀距离
    x6, y6 = hand_mapped[6]['x'], hand_mapped[6]['y']
    x7, y7 = hand_mapped[7]['x'], hand_mapped[7]['y']
    distance_67 = math.sqrt((x7 - x6)**2 + (y7 - y6)**2)
    # print('肩膀距离为:', distance_67)
    cv2.putText(image, "shoulder="+str(distance_67), (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3、判断
    if distance<distance_67*0.5 and angle_right<90 and angle_left<90:
        cv2.putText(image, "pose="+str("bai"), (int(20), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # 绘制手臂骨架线
        for limb in skeleton:
            p1, p2 = limb[0] - 1, limb[1] - 1
            # 手臂关键点之间的骨架线
            if p1 >= 5 and p1 <= 11 and p2 >= 4 and p2 <=10:
                pos1 = (int(kpts[p1*3]), int(kpts[p1*3 + 1]))
                pos2 = (int(kpts[p2*3]), int(kpts[p2*3 + 1]))

                conf1, conf2 = kpts[p1*3 + 2], kpts[p2*3 + 2]

                if conf1 > 0.1 and conf2 > 0.1:
                    R, G, B = pose_limb_color[p1]
                    cv2.line(image, pos1, pos2, (R, G, B), 1)

#  
def pose_body(kpts,image,body_mapped):

    # 绘制身体骨架线
    for limb in skeleton:
        p1, p2 = limb[0] - 1, limb[1] - 1
        # 身体关键点之间的骨架线
        if (p1 ==6-1 and p2==7-1) or  (p1 ==6-1 and p2==12-1) or  (p1 ==7-1 and p2==13-1) or  (p1 ==12-1 and p2==13-1) or (p1>=14-1 and p2>=12-1) :
            pos1 = (int(kpts[p1*3]), int(kpts[p1*3 + 1]))
            pos2 = (int(kpts[p2*3]), int(kpts[p2*3 + 1]))
            conf1, conf2 = kpts[p1*3 + 2], kpts[p2*3 + 2]

            if conf1 > 0.5 and conf2 > 0.5:
                R, G, B = pose_limb_color[p1]
                cv2.line(image, pos1, pos2, (R, G, B), 1)

    # 姿态判断
    # 右脚
    vec_13=body_mapped[13]['y'] 
    vec_15=body_mapped[15]['y'] 
    vec_17=body_mapped[17]['y'] 
    if body_mapped[17]['kpt_conf'] < 0.5:
        vec_17=0
    y_13_15=vec_15-vec_13
    y_17_15=vec_17-vec_15 

    # 左脚
    vec_12=body_mapped[12]['y'] 
    vec_14=body_mapped[14]['y'] 
    vec_16=body_mapped[16]['y'] 
    if body_mapped[16]['kpt_conf'] < 0.5:
        vec_16=0

    y_12_14=vec_14-vec_12
    y_16_14=vec_16-vec_14 

    # 姿态判断
    if vec_15>0 and vec_14>0:
        # 跪
        if (y_17_15 < y_13_15*0.5 and  y_16_14 < y_12_14 *0.5) or y_17_15 <=0 or y_16_14 <=0 :
            cv2.putText(image, "pose="+str("kneel"), (int(20), int(80)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 站
        else :
            cv2.putText(image, "pose="+str("station"), (int(20), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 222, 0), 1)

# 关键点信息提取
def get_kpoints_info(kpts, image):
    i=0
    for k in range(17):
        i=i+1
        # 1_判断 i 是否在 4 到 11 的范围内--提取手臂
        if i >= 6 and i <= 11:  
            R, G, B = pose_kpt_color[k]
            x_coord = kpts[k*3+0]   
            y_coord = kpts[k*3 + 1]
            kpt_conf = kpts[k*3 + 2]

            hand_mapped[i] = {'x': x_coord, 'y': y_coord, 'kpt_conf': kpt_conf}

            if kpt_conf > 0.2:
                cv2.circle(image, (int(x_coord), int(y_coord)), 3, (R, G, B), -1)
                cv2.putText(image, str(i), (int(x_coord), int(y_coord) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 2_提取身体躯干
        if (i == 6 or i == 7) or (i >= 12 and i <= 17):
            R, G, B = pose_kpt_color[k]
            
            x_coord = kpts[k*3+0]   
            y_coord = kpts[k*3 + 1]
            kpt_conf = kpts[k*3 + 2]

            if kpt_conf > 0.5:
                cv2.circle(image, (int(x_coord), int(y_coord)), 3, (R, G, B), -1)
                cv2.putText(image, str(i), (int(x_coord), int(y_coord) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                body_mapped[i] = {'x': x_coord, 'y': y_coord, 'kpt_conf': kpt_conf}
            else :
                body_mapped[i] = {'x': 0, 'y': 0, 'kpt_conf': kpt_conf}

    # 识别手臂动作判断--拜
    pose_bai(kpts,image)
       
    # # 识别身体动作判断-- 站 or 跪
    pose_body(kpts,image,body_mapped)

# 绘制关键点
def draw_kpoints_and_limbs(kpts, image):
    height, width = image.shape[:2]

    # 绘制关键点
    i=0
    for k in range(17):
        i=i+1
        R, G, B = pose_kpt_color[k]
        x_coord = kpts[k*3+0]   
        y_coord = kpts[k*3 + 1]
        kpt_conf = kpts[k*3 + 2]
        if kpt_conf > 0.5:
            cv2.circle(image, (int(x_coord), int(y_coord)), 3, (R, G, B), -1)
            cv2.putText(image, str(i), (int(x_coord), int(y_coord) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # 绘制骨架线
    for limb in skeleton:
        p1, p2 = limb[0] - 1, limb[1] - 1

        pos1 = (int(kpts[p1*3] ), int(kpts[p1*3 + 1] ))
        pos2 = (int(kpts[p2*3] ), int(kpts[p2*3 + 1]))

        conf1, conf2 = kpts[p1*3 + 2], kpts[p2*3 + 2]

        if conf1 > 0.2 and conf2 > 0.2:
            R, G, B = pose_limb_color[limb[0] - 1]
            cv2.line(image, pos1, pos2, (R, G, B), 1)

def visualImage(image, box_result):

    for ibox in box_result:

        kpts = ibox["kpts"]
        class_id = ibox['class_id']
        x=ibox['x'] 
        y=ibox['y']
        width=ibox['width']
        height=ibox['height']
        score =ibox['score']

        cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 2)
        caption = f"{class_id} {score:.2f}"
        cv2.putText(image, caption, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 关键点信息提取
        get_kpoints_info(kpts, image)

        draw_kpoints_and_limbs(kpts, image)

# 辅助函数：根据图像尺寸调整关键点坐标  
def scale_kpts(x, y, width, height):
    return x * width, y * height

def main():
    # 配置参数
    url = "rtsp://admin:abcd1234@192.168.2.163/smart265/ch1/sub/av_stream"
    # url=r"H:\测试视频\人\666.mp4"
    model_url = "http://192.168.2.16:10100/pose"
    
    # 视频捕获初始化
    # cap = cv2.VideoCapture(url)
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

        # 将图像帧编码为JPEG格式的二进制数据
        _, img_data = cv2.imencode('.jpg', frame)

        # 请求模型服务
        try:
            res = requests.post(model_url, data=img_data.tobytes())
            res = res.json()
            
            # 对于每个检测到的对象，绘制边界框和关键点
            if 'data' in res:
                visualImage(frame, res['data'])

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
