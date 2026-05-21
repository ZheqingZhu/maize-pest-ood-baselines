import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 导入跨龄期专用的 DataLoader
from dataset import get_cross_instar_dataloaders, SPECIES_MAP


def train_cross_instar_model(model, dataloaders, criterion, optimizer, device, num_epochs=15, output_dir='results'):
    """
    模型训练与验证循环 (仅在 Source Domain: 1-3龄 上进行)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'best_cross_instar_model.pth')

    best_acc = 0.0

    print("\n[开始在源域 (Source Domain: 1-3龄) 上训练模型...]")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存源域验证集上表现最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"🌟 发现更好的模型，已保存至 {save_path}")

    print(f'\n源域训练完成！最高源域验证集准确率: {best_acc:.4f}')
    return model


def evaluate_cross_instar_model(model, test_loader, device, class_names, output_dir='results'):
    """
    在目标域 (Target Domain: 4-6龄) 上进行灾难性测试
    """
    print("\n======================================================")
    print("🚀 正在未知目标域 (Target Domain: 4-6龄) 上进行泛化测试...")
    print("======================================================")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. 打印并保存分类报告
    print("\n[跨龄期泛化分类报告 - Cross-Instar Classification Report]")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    with open(os.path.join(output_dir, 'cross_instar_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Cross-Instar Distribution Shift Challenge ===\n")
        f.write("Source Training: Instar 1-3\n")
        f.write("Target Testing: Instar 4-6\n\n")
        f.write(report)

    # 2. 绘制并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names,
                yticklabels=class_names)  # 换成红色系，体现"灾难性挑战"
    plt.ylabel('True Label (实际晚期物种)')
    plt.xlabel('Predicted Label (源自早期特征的预测)')
    plt.title('Cross-Instar Distribution Shift: Testing on Instars 4-6')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'cross_instar_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ 跨龄期混淆矩阵已保存至 '{cm_path}'")
    print(f"✅ 跨龄期文本报告已保存至 '{os.path.join(output_dir, 'cross_instar_report.txt')}'")


if __name__ == '__main__':
    # 配置
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda item: item[1])]

    # 1. 加载跨龄期分布偏移的 DataLoader
    # 训练集和验证集只有 1-3 龄，测试集全是 4-6 龄
    print("\n加载数据集 (Cross-Instar Distribution Shift)...")
    dataloaders = get_cross_instar_dataloaders(CSV_FILE, DATA_DIR, batch_size=32, num_workers=0)

    # 2. 初始化分类模型 (预测物种 4 分类)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3. 训练 (仅在1-3龄)
    model = train_cross_instar_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        DEVICE,
        num_epochs=10,
        output_dir=OUTPUT_DIR
    )

    # 4. 评估 (直接冲击4-6龄)
    best_model_path = os.path.join(OUTPUT_DIR, 'best_cross_instar_model.pth')
    model.load_state_dict(torch.load(best_model_path))

    evaluate_cross_instar_model(model, dataloaders['test'], DEVICE, class_names, output_dir=OUTPUT_DIR)