import xml.etree.ElementTree as ET
import os

def convert_voc_xml_to_yolo(xml_path, class_map):
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_data = []
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        class_id = class_map.get(class_name, -1)
        if class_id == -1:
            continue

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        yolo_data.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    return yolo_data

def convert_folder_voc_to_yolo(xml_folder, yolo_folder, class_map):
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            yolo_data = convert_voc_xml_to_yolo(xml_path, class_map)
            yolo_filename = os.path.splitext(xml_file)[0] + '.txt'
            yolo_path = os.path.join(yolo_folder, yolo_filename)
            print(f"Converted {xml_file} to YOLO format.")
            with open(yolo_path, 'w') as file:
                for line in yolo_data:
                    file.write(line + '\n')

class_map = {
    'helmal': 0,
    'nohelmat': 1
}

xml_folder = r'H:\下载数据集\01-人\02-已标注\person-03\xml'
yolo_folder = r'H:\下载数据集\01-人\02-已标注\person-03\labels'

convert_folder_voc_to_yolo(xml_folder, yolo_folder, class_map)
