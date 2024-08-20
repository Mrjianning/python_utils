import os
import cv2
from tqdm import tqdm

def flip_bbox(box_info, image_width):
    try:
        class_id, x_center, y_center, width, height = map(float, box_info.split())
        flipped_x_center = 1 - x_center                      # 水平镜像后，中心点的 x 坐标为原始图片宽度减去中心点的 x 坐标 
        return int(class_id), flipped_x_center, y_center, width, height
    except ValueError:
        # 如果解析 box_info 时出现异常（比如空字符串或格式不正确），则返回 None
        return None


def flip_images_and_txt(input_image_folder, input_txt_folder, output_image_folder, output_txt_folder):
    # 如果输出文件夹不存在，则创建它们
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_txt_folder, exist_ok=True)

    # 遍历输入图片文件夹中的所有文件
    for filename in tqdm(os.listdir(input_image_folder)):
        if filename.endswith(".jpg"):
            # 读取图片
            image_path = os.path.join(input_image_folder, filename)
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape

            # 读取对应的txt文件
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(input_txt_folder, txt_filename)
            if os.path.exists(txt_filepath):
                with open(txt_filepath, 'r') as file:
                    box_info = file.readline().strip()

                # 镜像处理图片
                flipped_image = cv2.flip(image, 1)

                # 镜像处理边界框
                flipped_box_info = flip_bbox(box_info, image_width)

                # 保存镜像处理后的图片到输出图片文件夹
                output_image_filename = os.path.splitext(filename)[0] + "_flipped.jpg"
                output_image_path = os.path.join(output_image_folder, output_image_filename)
                cv2.imwrite(output_image_path, flipped_image)

                # 保存镜像处理后的txt文件到输出txt文件夹
                output_txt_filename = os.path.splitext(filename)[0] + "_flipped.txt"
                output_txt_path = os.path.join(output_txt_folder, output_txt_filename)

                if flipped_box_info is not None:
                    # 如果 flipped_box_info 不是 None，则保存其内容到 txt 文件中
                    flipped_box_info_str = " ".join(map(str, flipped_box_info))
                    with open(output_txt_path, 'w') as file:
                        file.write(flipped_box_info_str)
                else:
                    # 如果 flipped_box_info 是 None，则保存空的 txt 文件
                    open(output_txt_path, 'a').close()


# 输入和输出文件夹路径
input_image_folder = r"C:\Users\admin\Desktop\test_jx\out_images"
input_txt_folder = r"C:\Users\admin\Desktop\test_jx\out_labels"

output_image_folder = r"C:\Users\admin\Desktop\test_jx\output_images"
output_txt_folder = r"C:\Users\admin\Desktop\test_jx\output_txt"

# 批量处理图片和对应的txt文件
flip_images_and_txt(input_image_folder, input_txt_folder, output_image_folder, output_txt_folder)
