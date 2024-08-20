import xml.etree.ElementTree as ET
import os
import math

def convert_to_coordinates(cx, cy, width, height, angle):
    """
    根据中心点坐标、宽度、高度和旋转角度计算四个角点的坐标。
    """
    angle_rad = float(angle)
    dx = width / 2
    dy = height / 2

    # 四个角点的相对坐标
    corners = [
        (-dx, -dy),
        (-dx, dy),
        (dx, dy),
        (dx, -dy)
    ]

    # 旋转每个角点并加上中心坐标
    rotated_corners = []
    for corner in corners:
        rx = corner[0] * math.cos(angle_rad) - corner[1] * math.sin(angle_rad)
        ry = corner[0] * math.sin(angle_rad) + corner[1] * math.cos(angle_rad)
        rotated_corners.append((rx + cx, ry + cy))

    return [coord for corner in rotated_corners for coord in corner]

def xml_to_dota(xml_folder_path, output_path):
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历文件夹中的xml文件
    for xml_file in os.listdir(xml_folder_path):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder_path, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 构建DOTA格式的字符串
            dota_data = ""
            for obj in root.findall('object'):
                robndbox = obj.find('robndbox')
                cx, cy, w, h, angle = [robndbox.find(tag).text for tag in ('cx', 'cy', 'w', 'h', 'angle')]
                obj_name = obj.find('name').text

                # 计算四个角点的坐标
                coordinates = convert_to_coordinates(float(cx), float(cy), float(w), float(h), float(angle))
                coordinates_str = " ".join(map(str, coordinates))
                dota_data += f"{coordinates_str} {obj_name} 0\n"  # 添加对象名称标签和最后的0

            # 将DOTA数据保存到.txt文件
            txt_filename = os.path.splitext(xml_file)[0] + '.txt'
            with open(os.path.join(output_path, txt_filename), 'w') as file:
                file.write(dota_data)

            print(f"转换完成: {xml_file}")

# 转换函数
xml_folder_path = r'H:\360MoveData\Users\Administrator\Desktop\shitou'
output_path = r'H:\360MoveData\Users\Administrator\Desktop\shitou\dota'
xml_to_dota(xml_folder_path, output_path)
