import os
from tqdm import tqdm  # 导入 tqdm

# 指定根文件夹路径
root_folder_path = r'H:\360MoveData\Users\Administrator\Desktop\新建文件夹'
# 定义替换字符
replacement = '2'

def process_folder(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for item in tqdm(os.listdir(folder_path), desc='Processing Files'):
        item_path = os.path.join(folder_path, item)
        
        # 如果是文件夹，则递归处理子文件夹
        if os.path.isdir(item_path):
            process_folder(item_path)
        
        # 如果是以 .txt 扩展名结尾的文件
        elif item.endswith('.txt'):
            with open(item_path, 'r') as file:
                # 读取文件内容
                lines = file.readlines()

            # 修改文件内容
            modified_lines = []
            for line in lines:
                # 将每一行的第一个字符替换为指定的字符
                if line:
                    modified_line = replacement + line[1:]
                    modified_lines.append(modified_line)

            # 将修改后的内容写回文件
            with open(item_path, 'w') as file:
                file.writelines(modified_lines)

# 调用处理函数以处理根文件夹及其子文件夹中的所有 .txt 文件
process_folder(root_folder_path)
