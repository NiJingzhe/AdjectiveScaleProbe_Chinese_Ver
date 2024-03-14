import os
import fileinput

# 指定要处理的文件夹路径
folder_path = '../data/EntailmentInference_Chinese/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理文本文件
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        # 使用fileinput模块进行原地修改
        with fileinput.FileInput(file_path, inplace=True, encoding="utf-8") as file:
            for line in file:
                # 将"."替换为".\t"
                line = line.replace('。', '。\t')
                # 将".\t\n"替换为".\tnot_entailment\n"
                line = line.replace('。\t\n', '。\tnot_entailment\n')
                print(line, end='')