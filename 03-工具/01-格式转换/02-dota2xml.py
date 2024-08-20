import xml.etree.ElementTree as ET
import math
import os

# 指定文件夹路径
folder_path = r'H:\360MoveData\Users\Administrator\Desktop\guliao_obb\images'
# xml保存路径
output_path=r'H:\360MoveData\Users\Administrator\Desktop\guliao_obb\xml'


# 格式转换
def dota_to_xml(folder_path, txt_filename,output_path):

    # 从文本文件中读取数据
    data = []
    txt_path=os.path.join(folder_path, txt_filename) 
    if not (os.path.exists(txt_path) and os.path.getsize(txt_path) > 0):
        print(f"The text file {txt_filename} is empty or does not exist.")
        return

    with open(txt_path, 'r') as file:
       
        lines = file.readlines()

        # 创建XML根元素
        root = ET.Element("annotation")
        root.set("verified", "no")

        # 创建子元素并添加内容
        folder = ET.SubElement(root, "folder")
        folder.text = "guliao_obb"  # 文件夹名称

        filename = ET.SubElement(root, "filename")
        filename.text = os.path.splitext(txt_filename)[0]  # 文件名

        path = ET.SubElement(root, "path")
        img_filename = os.path.splitext(txt_filename)[0] +'.JPG'
        img_path = os.path.join(folder_path, img_filename) 
        path.text =  img_path # 文件路径

        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"  # 数据库名称

        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        width.text = "2048"  # 图像宽度
        height = ET.SubElement(size, "height")
        height.text = "1536"  # 图像高度
        depth = ET.SubElement(size, "depth")
        depth.text = "3"  # 图像深度

        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"  # 是否分割

        for line in lines:
            line = line.strip().split()
            if len(line) == 10:
                item=([line[0]] + [float(x) for x in line[1:]])  # 将字符串转换为浮点数
                
                obj = ET.SubElement(root, "object")

                obj_type = ET.SubElement(obj, "type")
                obj_type.text = "robndbox"  # 对象类型

                obj_name = ET.SubElement(obj, "name")
                obj_name.text = item[0]  # 对象名称

                pose = ET.SubElement(obj, "pose")
                pose.text = "Unspecified"  # 姿势

                truncated = ET.SubElement(obj, "truncated")
                truncated.text = "0"  # 是否被截断

                difficult = ET.SubElement(obj, "difficult")
                difficult.text = "1"  # 是否为难以识别的对象

                x1, y1, x2, y2, x3, y3, x4, y4 = item[2:]
                
                # 计算宽度和高度
                width = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 宽度
                height = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)  # 高度

                # 计算中心坐标
                cx = (x1 + x2 + x3 + x4) / 4  # 中心x坐标
                cy = (y1 + y2 + y3 + y4) / 4  # 中心y坐标

                # 计算旋转角度
                delta_x = x2 - x1
                delta_y = y2 - y1
                angle_rad = math.atan2(delta_y, delta_x)  # 弧度表示的角度
    
                # 创建'robndbox'元素及其子元素
                robndbox = ET.SubElement(obj, "robndbox")
                cx_elem = ET.SubElement(robndbox, "cx")
                cx_elem.text = str(cx)  # 中心x坐标
                cy_elem = ET.SubElement(robndbox, "cy")
                cy_elem.text = str(cy)  # 中心y坐标
                width_elem = ET.SubElement(robndbox, "w")
                width_elem.text = str(width)  # 宽度
                height_elem = ET.SubElement(robndbox, "h")
                height_elem.text = str(height)  # 高度
                angle_elem = ET.SubElement(robndbox, "angle")
                angle_elem.text = str(angle_rad)  # 旋转角度

                segmentation = ET.SubElement(obj, "segmentation")
                x1 = ET.SubElement(segmentation, "x1")
                x1.text = str(item[2])
                y1 = ET.SubElement(segmentation, "y1")
                y1.text = str(item[3])
                x2 = ET.SubElement(segmentation, "x2")
                x2.text = str(item[4])
                y2 = ET.SubElement(segmentation, "y2")
                y2.text = str(item[5])
                x3 = ET.SubElement(segmentation, "x3")
                x3.text = str(item[6])
                y3 = ET.SubElement(segmentation, "y3")
                y3.text = str(item[7])
                x4 = ET.SubElement(segmentation, "x4")
                x4.text = str(item[8])
                y4 = ET.SubElement(segmentation, "y4")
                y4.text = str(item[9])

        # 创建XML树并保存到文件
        xml_filename = os.path.splitext(txt_filename)[0]+".xml"
        xml_path = os.path.join(output_path, xml_filename) 
        tree = ET.ElementTree(root)
        tree.write(xml_path)

# 遍历txt
def convert_txt_files_to_xml(folder_path,output_path):

    # 遍历文件中txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for txt_filename in txt_files:
        dota_to_xml(folder_path, txt_filename,output_path)
        print("转换完成:", txt_filename)

convert_txt_files_to_xml(folder_path,output_path)