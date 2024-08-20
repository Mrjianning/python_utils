import os

# 指定根文件夹路径
root_folder_path = r'H:\360MoveData\Users\Administrator\Desktop\新建文件夹'

def process_folder(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for item in (os.listdir(folder_path)):
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
                # 检查第一个字符的值并设置替换字符
                if line:
                    first_char = line[0]
                    if first_char == '0':
                        replacement = '4'
                    elif first_char == '1':
                        replacement = '5'
                    elif first_char == '2':
                        replacement = '6'
                    else:
                        replacement = first_char  # 如果不是0、1、2，则保持不变

                    modified_line = replacement + line[1:]
                    print(modified_line)
                    modified_lines.append(modified_line)

            # 将修改后的内容写回文件
            with open(item_path, 'w') as file:
                file.writelines(modified_lines)

# 调用处理函数以处理根文件夹及其子文件夹中的所有 .txt 文件
process_folder(root_folder_path)
