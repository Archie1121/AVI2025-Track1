import os
import shutil

# 设置每个文件夹对应要保留的 question
folder_question_map = {
    "/data2/public_datasets/AVI/AVI_features/video/faceXformer": "q3",
    "/data2/public_datasets/AVI/AVI_features/video/faceXformer": "q4",
    "/data2/public_datasets/AVI/AVI_features/video/faceXformer": "q5",  # 可扩展
    "/data2/public_datasets/AVI/AVI_features/video/faceXformer": "q6",
}

# 输出目录
output_dir = "/data2/heyichao/AVI_track1_code/AVI-track1/features/video"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个文件夹及其对应的目标 question
for folder, target_question in folder_question_map.items():
    for file in os.listdir(folder):
        if file.endswith(".npy") and f"_{target_question}.npy" in file:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(output_dir, file)
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {file} from {folder}")
