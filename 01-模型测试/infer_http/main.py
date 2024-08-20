import cv2
import queue
import threading

from src import Tracker 
from src import Detect 
from src import Pose 
from src import Seg 

def infer(url,infer_,model_url,target_fps):

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    video_queue = queue.Queue(maxsize=10)
    visualization_queue = queue.Queue(maxsize=10)

    video_thread = threading.Thread(target=infer_.video_capture_thread, args=(cap, video_queue, target_fps))
    video_thread.daemon = True
    video_thread.start()

    process_thread = threading.Thread(target=infer_.process_video, args=(model_url, video_queue, visualization_queue))
    process_thread.daemon = True
    process_thread.start()

    cv2.namedWindow('IP Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('IP Camera', 1280, 720)
    infer_.visualization_thread(visualization_queue)

if __name__ == "__main__":

    pose = Pose.Pose()
    detect = Detect.Detect()
    seg = Seg.Seg()
    tracker = Tracker.Tracker()

    url = r"rtsp://admin:abcd1234@192.168.2.99:554/h265/ch1/main/video"
    infer(url,seg,"http://192.168.2.16:8800/seg",25)