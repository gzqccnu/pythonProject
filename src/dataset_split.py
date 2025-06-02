import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 配置参数
DATASET_DIR = "F:\File\Camellia oleifera\dataset"  # 原始数据集目录
OUTPUT_DIR = "yolo_split_dataset"          # 输出目录
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']  # 支持的图片格式
TRAIN_RATIO = 0.8                          # 训练集比例
SEED = 42                                  # 随机种子

# 创建目标目录结构
os.makedirs(os.path.join(OUTPUT_DIR, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "labels"), exist_ok=True)

# 获取有效图片列表（存在对应标注文件的图片）
valid_samples = []
for filename in os.listdir(DATASET_DIR):
    if any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS):
        txt_path = os.path.join(DATASET_DIR, os.path.splitext(filename)[0] + ".txt")
        if os.path.exists(txt_path):
            valid_samples.append(filename)
        else:
            print(f"警告: {filename} 没有对应的标注文件，已跳过")

# 划分数据集
train_files, val_files = train_test_split(
    valid_samples,
    train_size=TRAIN_RATIO,
    random_state=SEED
)

def copy_files(file_list, split_type):
    """复制文件到目标目录"""
    for filename in file_list:
        # 原始路径
        img_src = os.path.join(DATASET_DIR, filename)
        txt_src = os.path.join(DATASET_DIR, os.path.splitext(filename)[0] + ".txt")
        
        # 目标路径
        img_dst = os.path.join(OUTPUT_DIR, split_type, "images", filename)
        txt_dst = os.path.join(OUTPUT_DIR, split_type, "labels", os.path.basename(txt_src))
        
        # 执行复制
        shutil.copy(img_src, img_dst)
        shutil.copy(txt_src, txt_dst)

# 执行复制操作
copy_files(train_files, "train")
copy_files(val_files, "val")

# 打印统计信息
print(f"总有效样本: {len(valid_samples)}")
print(f"训练集样本: {len(train_files)}")
print(f"验证集样本: {len(val_files)}")
print(f"数据集结构已保存在: {OUTPUT_DIR}")