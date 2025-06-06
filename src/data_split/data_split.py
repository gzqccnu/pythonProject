import os
import shutil
import random

def split_dataset_with_labels(img_dir, label_dir, output_dir, 
                             img_ext='.jpg', label_ext='.txt',
                             train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    """
    同步划分图片和标签数据集
    
    参数:
        img_dir (str): 图片源目录
        label_dir (str): 标签源目录
        output_dir (str): 输出根目录
        img_ext (str): 图片文件扩展名 (默认.jpg)
        label_ext (str): 标签文件扩展名 (默认.txt)
        比例参数同上
    """
    # 验证比例
    assert round(train_ratio + test_ratio + val_ratio, 10) == 1.0, "比例总和必须为1"
    
    # 创建输出目录结构
    subsets = ['train', 'test', 'val']
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labels'), exist_ok=True)

    # 获取匹配的文件对
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_ext)]
    paired_files = []
    
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}{label_ext}"
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件 {label_file} 不存在")
        
        paired_files.append((img_file, label_file))
    
    # 打乱顺序
    random.shuffle(paired_files)
    
    # 划分数据集
    total = len(paired_files)
    train_end = int(train_ratio * total)
    test_end = train_end + int(test_ratio * total)
    
    train_pairs = paired_files[:train_end]
    test_pairs = paired_files[train_end:test_end]
    val_pairs = paired_files[test_end:]
    
    # 复制文件的函数
    def copy_paired_files(pairs, subset):
        for img_file, label_file in pairs:
            # 复制图片
            shutil.copy2(
                os.path.join(img_dir, img_file),
                os.path.join(output_dir, subset, 'images', img_file)
            )
            # 复制标签
            shutil.copy2(
                os.path.join(label_dir, label_file),
                os.path.join(output_dir, subset, 'labels', label_file)
            )
    
    # 执行复制
    copy_paired_files(train_pairs, 'train')
    copy_paired_files(test_pairs, 'test')
    copy_paired_files(val_pairs, 'val')

if __name__ == '__main__':
    split_dataset_with_labels(
        img_dir='/root/code/.code/pic/images',
        label_dir='/root/code/.code/pic/labels',  # 新增标签目录参数
        output_dir='/root/code/.code/pic/dataset',
        img_ext='.jpg',
        label_ext='.txt',  # 可修改为实际使用的标签格式
        train_ratio=0.8,
        test_ratio=0.1,
        val_ratio=0.1
    )
