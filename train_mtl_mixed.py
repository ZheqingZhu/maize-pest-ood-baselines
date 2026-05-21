import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
import matplotlib.pyplot as plt

# 导入 DataLoader
from dataset import get_baseline_dataloaders, SPECIES_MAP


# ================= 1. 混合多任务 ResNet 模型 =================
class MixedMTLResNet(nn.Module):
    def __init__(self, num_species=4):
        super(MixedMTLResNet, self).__init__()
        # 加载预训练主干
        shared_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(shared_resnet.children())[:-1])
        num_ftrs = shared_resnet.fc.in_features

        # 头 A：物种分类头 (输出 4 个类别的 Logits)
        self.fc_species = nn.Linear(num_ftrs, num_species)

        # 头 B：日龄回归头 (输出 1 个连续的浮点数)
        self.fc_days = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        out_species = self.fc_species(x)
        out_days = self.fc_days(x)
        return out_species, out_days


# ================= 2. 训练与验证循环 =================
def train_mixed_mtl(model, dataloaders, optimizer, device, num_epochs=15, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'best_mixed_mtl_model.pth')

    # 定义两种不同的损失函数
    criterion_species = nn.CrossEntropyLoss()
    criterion_days = nn.MSELoss()

    # 权重平衡：因为 MSE Loss 的数值通常比 CrossEntropy 大很多，这里给回归 Loss 加一个缩放系数
    weight_regression = 0.1

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_species = 0

            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)

                # 获取双标签：物种 (LongTensor) 和 天数 (FloatTensor)
                labels_species = batch['label'].to(device)
                labels_days = batch['days'].float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out_species, out_days = model(inputs)

                    _, preds_species = torch.max(out_species, 1)

                    # 计算双重损失
                    loss_class = criterion_species(out_species, labels_species)
                    loss_reg = criterion_days(out_days, labels_days)

                    # 混合总损失
                    total_loss = loss_class + weight_regression * loss_reg

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                running_loss += total_loss.item() * inputs.size(0)
                corrects_species += torch.sum(preds_species == labels_species.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc_species = corrects_species.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Total Loss: {epoch_loss:.4f} | Species Acc: {acc_species:.4f}')

            # 保存验证集 Total Loss 最低的权重
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)
                print(f"🌟 发现更低的总损失，混合 MTL 模型已保存！")

    print(f'\n混合 MTL 训练完成！')
    return model


# ================= 3. 测试与评估 =================
def evaluate_mixed_mtl(model, test_loader, device, output_dir='results'):
    print("\n--- 正在测试集上评估混合 MTL 模型 ---")
    model.eval()

    all_preds_species, all_labels_species = [], []
    all_preds_days, all_labels_days = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels_species = batch['label'].to(device)
            labels_days = batch['days'].float().numpy()

            out_species, out_days = model(inputs)

            _, preds_species = torch.max(out_species, 1)
            preds_days = out_days.cpu().numpy().flatten()

            all_preds_species.extend(preds_species.cpu().numpy())
            all_labels_species.extend(labels_species.cpu().numpy())

            all_preds_days.extend(preds_days)
            all_labels_days.extend(labels_days)

    # 1. 物种分类报告
    species_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda item: item[1])]
    print("\n[任务A: 物种分类报告 - Species Classification]")
    report_species = classification_report(all_labels_species, all_preds_species, target_names=species_names, digits=4)
    print(report_species)

    # 2. 日龄回归报告
    all_preds_days = np.array(all_preds_days)
    all_labels_days = np.array(all_labels_days)

    mae = mean_absolute_error(all_labels_days, all_preds_days)
    rmse = np.sqrt(mean_squared_error(all_labels_days, all_preds_days))
    r2 = r2_score(all_labels_days, all_preds_days)

    print("\n[任务B: 连续日龄回归预测 - Chronological Age Regression]")
    print(f"Mean Absolute Error (MAE): {mae:.4f} 天")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} 天")
    print(f"R-squared (R2 Score): {r2:.4f}")

    # 保存报告
    with open(os.path.join(output_dir, 'mixed_mtl_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Task A: Species Classification ===\n" + report_species + "\n\n")
        f.write("=== Task B: Age Regression ===\n")
        f.write(f"MAE: {mae:.4f} Days\nRMSE: {rmse:.4f} Days\nR2: {r2:.4f}")

    print(f"✅ 混合 MTL 综合报告已保存至 {os.path.join(output_dir, 'mixed_mtl_report.txt')}")


if __name__ == '__main__':
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n加载数据集 (混合 MTL)...")
    dataloaders = get_baseline_dataloaders(CSV_FILE, DATA_DIR, batch_size=32, num_workers=0)

    model = MixedMTLResNet(num_species=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练 10 个 epoch
    model = train_mixed_mtl(model, dataloaders, optimizer, DEVICE, num_epochs=10, output_dir=OUTPUT_DIR)

    # 评估
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_mixed_mtl_model.pth')))
    evaluate_mixed_mtl(model, dataloaders['test'], DEVICE, output_dir=OUTPUT_DIR)