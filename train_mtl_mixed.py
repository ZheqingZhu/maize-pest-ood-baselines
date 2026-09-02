import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
import matplotlib.pyplot as plt

# Import DataLoader
from dataset import get_baseline_dataloaders, SPECIES_MAP


# ================= 1. Mixed multi-task ResNet model =================
class MixedMTLResNet(nn.Module):
    def __init__(self, num_species=4):
        super(MixedMTLResNet, self).__init__()
        # Load the pretrained backbone
        shared_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(shared_resnet.children())[:-1])
        num_ftrs = shared_resnet.fc.in_features

        # Head A: species classification head (outputs logits for 4 classes)
        self.fc_species = nn.Linear(num_ftrs, num_species)

        # Head B: chronological age regression head (outputs 1 continuous float)
        self.fc_days = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        out_species = self.fc_species(x)
        out_days = self.fc_days(x)
        return out_species, out_days


# ================= 2. Training and validation loop =================
def train_mixed_mtl(model, dataloaders, optimizer, device, num_epochs=15, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'best_mixed_mtl_model.pth')

    # Define two different loss functions
    criterion_species = nn.CrossEntropyLoss()
    criterion_days = nn.MSELoss()

    # Weight balancing: MSE loss is usually much larger than CrossEntropy, so scale the regression loss down
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

                # Get both labels: species (LongTensor) and days (FloatTensor)
                labels_species = batch['label'].to(device)
                labels_days = batch['days'].float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out_species, out_days = model(inputs)

                    _, preds_species = torch.max(out_species, 1)

                    # Compute the dual loss
                    loss_class = criterion_species(out_species, labels_species)
                    loss_reg = criterion_days(out_days, labels_days)

                    # Combined total loss
                    total_loss = loss_class + weight_regression * loss_reg

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                running_loss += total_loss.item() * inputs.size(0)
                corrects_species += torch.sum(preds_species == labels_species.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc_species = corrects_species.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Total Loss: {epoch_loss:.4f} | Species Acc: {acc_species:.4f}')

            # Save the weights with the lowest val Total Loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Lower total loss found, mixed MTL model saved!")

    print(f'\nMixed MTL training complete!')
    return model


# ================= 3. Testing and evaluation =================
def evaluate_mixed_mtl(model, test_loader, device, output_dir='results'):
    print("\n--- Evaluating mixed MTL model on the test set ---")
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

    # 1. Species classification report
    species_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda item: item[1])]
    print("\n[Task A: Species Classification Report]")
    report_species = classification_report(all_labels_species, all_preds_species, target_names=species_names, digits=4)
    print(report_species)

    # 2. Chronological age regression report
    all_preds_days = np.array(all_preds_days)
    all_labels_days = np.array(all_labels_days)

    mae = mean_absolute_error(all_labels_days, all_preds_days)
    rmse = np.sqrt(mean_squared_error(all_labels_days, all_preds_days))
    r2 = r2_score(all_labels_days, all_preds_days)

    print("\n[Task B: Continuous Chronological Age Regression]")
    print(f"Mean Absolute Error (MAE): {mae:.4f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days")
    print(f"R-squared (R2 Score): {r2:.4f}")

    # Save the report
    with open(os.path.join(output_dir, 'mixed_mtl_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Task A: Species Classification ===\n" + report_species + "\n\n")
        f.write("=== Task B: Age Regression ===\n")
        f.write(f"MAE: {mae:.4f} Days\nRMSE: {rmse:.4f} Days\nR2: {r2:.4f}")

    print(f"✅ Mixed MTL combined report saved to {os.path.join(output_dir, 'mixed_mtl_report.txt')}")


if __name__ == '__main__':
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nLoading dataset (mixed MTL)...")
    dataloaders = get_baseline_dataloaders(CSV_FILE, DATA_DIR, batch_size=32, num_workers=0)

    model = MixedMTLResNet(num_species=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for 10 epochs
    model = train_mixed_mtl(model, dataloaders, optimizer, DEVICE, num_epochs=10, output_dir=OUTPUT_DIR)

    # Evaluate
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_mixed_mtl_model.pth')))
    evaluate_mixed_mtl(model, dataloaders['test'], DEVICE, output_dir=OUTPUT_DIR)
