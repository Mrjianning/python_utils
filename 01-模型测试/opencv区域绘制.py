import cv2

# 定义全局变量
drawing = False  # 是否正在绘制
start_x, start_y = -1, -1  # 矩形区域的起点坐标

# 鼠标回调函数
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_x, start_y
    
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下，开始绘制
        drawing = True
        start_x, start_y = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键释放，结束绘制
        drawing = False
        end_x, end_y = x, y
        
        # 画矩形
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow('Image', image)

# 读取图像
image = cv2.imread('image.jpg')

# 创建窗口并显示图像
cv2.namedWindow('Image')
cv2.imshow('Image', image)

# 设置鼠标回调函数
cv2.setMouseCallback('Image', draw_rectangle)

# 等待用户关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
