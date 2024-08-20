import requests
import time
import json

# 服务器的URL
url = "http://192.168.2.16:8888/push_stream"

# 请求的初始数据
data = {
    "cameraId": "5678",
    "rtsp": "rtsp://admin:abcd1234@192.168.2.163:554/h265/ch1/main/av_stream",
    "send_draw_img": "false"
}

# 切换send_draw_img的值并发送请求
while True:
    try:
        # 切换send_draw_img的值
        if data["send_draw_img"] == "false":
            data["send_draw_img"] = "true"
        else:
            data["send_draw_img"] = "false"
        
        # 发送POST请求
        response = requests.post(url, json=data)
        
        # 打印响应状态和内容
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

    # 等待5秒钟
    time.sleep(3)
