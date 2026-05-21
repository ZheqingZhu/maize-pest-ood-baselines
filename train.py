import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 导入你刚刚跑通的 DataLoader 函数
from dataset import get_baseline_dataloaders, get_cross_instar_dataloaders, SPECIES_MAP


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=15, output_dir='results',
                save_filename='best_model.pth'):
    """
    模型训练与验证循环
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_filename)

    best_acc = 0.0

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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"🌟 发现更好的模型，已保存至 {save_path}")

    print(f'\n训练完成！最高验证集准确率: {best_acc:.4f}')
    return model


def evaluate_model(model, test_loader, device, class_names, output_dir='results'):
    """
    在测试集上评估模型，并将结果保存至指定文件夹
    """
    print("\n--- 正在测试集上评估模型 ---")
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

    # 1. 保存并打印分类报告
    print("\n[分类报告 Classification Report]")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # 将报告也保存为 txt 文件
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # 2. 绘制并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label (实际物种)')
    plt.xlabel('Predicted Label (预测物种)')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ 混淆矩阵已保存至 '{cm_path}'")
    print(f"✅ 分类报告文本已保存至 '{os.path.join(output_dir, 'classification_report.txt')}'")


if __name__ == '__main__':
    # 配置
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'  # 所有的产出都会放在这个文件夹里
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda item: item[1])]

    # 加载数据
    print("\n加载数据集 (Baseline)...")
    dataloaders = get_baseline_dataloaders(CSV_FILE, DATA_DIR, batch_size=32, num_workers=0)

    # 初始化模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 1. 训练
    # 结果会自动存入 results 文件夹
    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        DEVICE,
        num_epochs=10,
        output_dir=OUTPUT_DIR,
        save_filename='resnet50_baseline.pth'
    )

    # 2. 评估
    # 加载刚才保存的最好权重
    best_model_path = os.path.join(OUTPUT_DIR, 'resnet50_baseline.pth')
    model.load_state_dict(torch.load(best_model_path))

    evaluate_model(model, dataloaders['test'], DEVICE, class_names, output_dir=OUTPUT_DIR)