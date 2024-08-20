import cv2

# 读取图像
image = cv2.imread(r"C:\Users\admin\Desktop\sw\out\T_img (3).jpg")

# 从文本文件读取标注数据
with open(r"C:\Users\admin\Desktop\sw\out\T_img (3).txt", "r") as file:
    lines = file.readlines()

for line in lines:
    # 解析每一行的标注数据
    data = line.strip().split()
    if len(data) == 5:
        class_id, center_x, center_y, width, height = map(float, data)

        # 图像的宽度和高度
        image_width, image_height = image.shape[1], image.shape[0]

        # 计算边界框的左上角和右下角坐标
        left = int((center_x - width / 2) * image_width)
        top = int((center_y - height / 2) * image_height)
        right = int((center_x + width / 2) * image_width)
        bottom = int((center_y + height / 2) * image_height)

        # 绘制边界框
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# Set a fixed window height
fixed_height = 600

# Calculate the corresponding window width based on the aspect ratio
aspect_ratio = image.shape[1] / image.shape[0]
fixed_width = int(fixed_height * aspect_ratio)

# Create a window with the WINDOW_NORMAL flag to allow resizing
cv2.namedWindow("Image with Bounding Boxes", cv2.WINDOW_NORMAL)

# Resize the window to the fixed size
cv2.resizeWindow("Image with Bounding Boxes", fixed_width, fixed_height)

# Display the image
cv2.imshow("Image with Bounding Boxes", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
