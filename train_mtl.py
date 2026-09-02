import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report
import os
import numpy as np

# Import DataLoader
from dataset import get_baseline_dataloaders, SPECIES_MAP


# ================= 1. Define the multi-task ResNet model =================
class MultiTaskResNet(nn.Module):
    def __init__(self, num_species=4, num_instars=6):
        super(MultiTaskResNet, self).__init__()
        # Load pretrained ResNet50
        shared_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Take the feature extractor part before the fully-connected layer
        self.features = nn.Sequential(*list(shared_resnet.children())[:-1])
        num_ftrs = shared_resnet.fc.in_features

        # Define two independent classification heads
        self.fc_species = nn.Linear(num_ftrs, num_species)  # species classification head (4 classes)
        self.fc_instar = nn.Linear(num_ftrs, num_instars)  # instar classification head (6 classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        out_species = self.fc_species(x)
        out_instar = self.fc_instar(x)
        return out_species, out_instar


# ================= 2. Training and validation loop =================
def train_mtl_model(model, dataloaders, criterion, optimizer, device, num_epochs=15, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'best_mtl_model.pth')

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
            corrects_instar = 0

            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)

                # Get both labels
                labels_species = batch['label'].to(device)
                # NOTE: instars in the dataset are 1-6, PyTorch class indices must be 0-5, so subtract 1
                labels_instar = (batch['instar'].clone().detach() - 1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out_species, out_instar = model(inputs)

                    _, preds_species = torch.max(out_species, 1)
                    _, preds_instar = torch.max(out_instar, 1)

                    # Compute the dual loss (1:1 weights)
                    loss_species = criterion(out_species, labels_species)
                    loss_instar = criterion(out_instar, labels_instar)
                    total_loss = loss_species + loss_instar

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                running_loss += total_loss.item() * inputs.size(0)
                corrects_species += torch.sum(preds_species == labels_species.data)
                corrects_instar += torch.sum(preds_instar == labels_instar.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc_species = corrects_species.double() / len(dataloaders[phase].dataset)
            acc_instar = corrects_instar.double() / len(dataloaders[phase].dataset)

            print(
                f'{phase.capitalize()} Total Loss: {epoch_loss:.4f} | Species Acc: {acc_species:.4f} | Instar Acc: {acc_instar:.4f}')

            # Save the weights with the lowest val Total Loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Lower val loss found, model saved!")

    print(f'\nMTL training complete!')
    return model


# ================= 3. Testing and evaluation =================
def evaluate_mtl_model(model, test_loader, device, output_dir='results'):
    print("\n--- Evaluating MTL model on the test set ---")
    model.eval()

    all_preds_species, all_labels_species = [], []
    all_preds_instar, all_labels_instar = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels_species = batch['label'].to(device)
            labels_instar = (batch['instar'].clone().detach() - 1).to(device)

            out_species, out_instar = model(inputs)

            _, preds_species = torch.max(out_species, 1)
            _, preds_instar = torch.max(out_instar, 1)

            all_preds_species.extend(preds_species.cpu().numpy())
            all_labels_species.extend(labels_species.cpu().numpy())

            all_preds_instar.extend(preds_instar.cpu().numpy())
            all_labels_instar.extend(labels_instar.cpu().numpy())

    species_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda item: item[1])]
    instar_names = [f'Instar_{i}' for i in range(1, 7)]

    print("\n[Species Classification Report]")
    report_species = classification_report(all_labels_species, all_preds_species, target_names=species_names, digits=4)
    print(report_species)

    print("\n[Instar Prediction Report]")
    report_instar = classification_report(all_labels_instar, all_preds_instar, target_names=instar_names, digits=4)
    print(report_instar)

    with open(os.path.join(output_dir, 'mtl_classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Species Prediction ===\n" + report_species + "\n\n")
        f.write("=== Instar Prediction ===\n" + report_instar)

    print(f"✅ MTL combined report saved to {os.path.join(output_dir, 'mtl_classification_report.txt')}")


if __name__ == '__main__':
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nLoading MTL dataset...")
    dataloaders = get_baseline_dataloaders(CSV_FILE, DATA_DIR, batch_size=32, num_workers=0)

    model = MultiTaskResNet(num_species=4, num_instars=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 10 epochs are enough to see the trend
    model = train_mtl_model(model, dataloaders, criterion, optimizer, DEVICE, num_epochs=10, output_dir=OUTPUT_DIR)

    # Evaluate
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_mtl_model.pth')))
    evaluate_mtl_model(model, dataloaders['test'], DEVICE, output_dir=OUTPUT_DIR)
