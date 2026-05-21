import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# 定义物种到类别ID的映射
SPECIES_MAP = {
    'frugiperda': 0,  # 草地贪夜蛾
    'litura': 1,  # 斜纹夜蛾
    'separata': 2,  # 黏虫
    'ypsilon': 3  # 小地老虎
}


class MaizeHerbivoryDataset(Dataset):
    """
    玉米受食图像自定义数据集类
    """

    def __init__(self, dataframe, root_dir, transform=None):
        """
        :param dataframe: 包含 image_path, species, instar, days 信息的 pandas DataFrame
        :param root_dir: 图像所在的根目录 (例如 'data')
        :param transform: 图像预处理/增强操作
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 获取相对路径并拼接绝对路径
        rel_path = self.dataframe.loc[idx, 'image_path']
        img_path = os.path.join(self.root_dir, rel_path)

        # 读取图像 (转换为RGB，防止有灰度图或四通道图)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取物种标签
        species_str = self.dataframe.loc[idx, 'species']
        label = SPECIES_MAP[species_str]

        # 获取其他元数据（可选，在多任务学习或误差分析时非常有用）
        instar = self.dataframe.loc[idx, 'instar']
        days = self.dataframe.loc[idx, 'days_post_hatching']

        # 将标签转为 tensor
        label = torch.tensor(label, dtype=torch.long)

        # 返回字典格式，方便后续获取元数据进行 t-SNE 分析
        return {
            'image': image,
            'label': label,
            'instar': instar,
            'days': days,
            'path': rel_path
        }


# ================= 数据预处理定义 =================
# 基于 ImageNet 预训练模型的标准预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # 在这里可以加入你想要的数据增强，如翻转、颜色微调等
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ================= 划分策略 1: 基线随机划分 =================
def get_baseline_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4):
    """按照 70:10:20 的比例进行随机基础划分"""
    df = pd.read_csv(csv_path)

    # 第一次划分：70% 训练，30% (验证+测试)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['species'], random_state=42)
    # 第二次划分：验证和测试 1:2 (即总体 10% 和 20%)
    val_df, test_df = train_test_split(temp_df, test_size=(2 / 3), stratify=temp_df['species'], random_state=42)

    print(f"[Baseline Split] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    datasets = {
        'train': MaizeHerbivoryDataset(train_df, root_dir, transform=data_transforms['train']),
        'val': MaizeHerbivoryDataset(val_df, root_dir, transform=data_transforms['val_test']),
        'test': MaizeHerbivoryDataset(test_df, root_dir, transform=data_transforms['val_test'])
    }

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'val', 'test']
    }
    return dataloaders


# ================= 划分策略 2: 跨龄期分布偏移划分 =================
def get_cross_instar_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4):
    """
    分布偏移挑战：
    Source Domain (训练+验证): 1-3 龄期
    Target Domain (测试): 4-6 龄期
    """
    df = pd.read_csv(csv_path)

    # 根据龄期拆分源域和目标域
    source_df = df[df['instar'].isin([1, 2, 3])]
    target_df = df[df['instar'].isin([4, 5, 6])]

    # 源域内部再划分为训练集和验证集 (例如 80%训练, 20%验证)
    train_df, val_df = train_test_split(source_df, test_size=0.2, stratify=source_df['species'], random_state=42)
    test_df = target_df  # 测试集完全由未见过的晚期阶段组成

    print(
        f"[Cross-Instar Split] Source Train(1-3龄): {len(train_df)}, Source Val(1-3龄): {len(val_df)}, Target Test(4-6龄): {len(test_df)}")

    datasets = {
        'train': MaizeHerbivoryDataset(train_df, root_dir, transform=data_transforms['train']),
        'val': MaizeHerbivoryDataset(val_df, root_dir, transform=data_transforms['val_test']),
        'test': MaizeHerbivoryDataset(test_df, root_dir, transform=data_transforms['val_test'])  # 重点：测试集用 target_df
    }

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'val', 'test']
    }
    return dataloaders


# ================= 测试代码 =================
if __name__ == '__main__':
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'  # 替换为你的实际图片根目录

    # 1. 测试基线 DataLoader
    print("--- 正在构建 Baseline DataLoaders ---")
    baseline_loaders = get_baseline_dataloaders(CSV_FILE, DATA_DIR, batch_size=16, num_workers=0)

    # 获取一个 Batch 试试
    batch = next(iter(baseline_loaders['train']))
    print(f"Batch Image Shape: {batch['image'].shape}")  # 预期: [16, 3, 224, 224]
    print(f"Batch Label Shape: {batch['label'].shape}")  # 预期: [16]
    print(f"Batch Instar Sample: {batch['instar'][:5]}")

    print("\n--- 正在构建 Cross-Instar DataLoaders ---")
    cross_loaders = get_cross_instar_dataloaders(CSV_FILE, DATA_DIR, batch_size=16, num_workers=0)