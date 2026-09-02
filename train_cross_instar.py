import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import the cross-instar specific DataLoader
from dataset import get_cross_instar_dataloaders, SPECIES_MAP


def train_cross_instar_model(model, dataloaders, criterion, optimizer, device, num_epochs=15, output_dir='results'):
    """
    Model training and validation loop (only on Source Domain: instars 1-3)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'best_cross_instar_model.pth')

    best_acc = 0.0

    print("\n[Start training the model on the source domain (Source Domain: instars 1-3)...]")
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

            # Save the model that performs best on the source-domain val set
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Found a better model, saved to {save_path}")

    print(f'\nSource-domain training complete! Best source-domain val accuracy: {best_acc:.4f}')
    return model


def evaluate_cross_instar_model(model, test_loader, device, class_names, output_dir='results'):
    """
    Catastrophic test on the target domain (Target Domain: instars 4-6)
    """
    print("\n======================================================")
    print("🚀 Running generalization test on the unseen target domain (Target Domain: instars 4-6)...")
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

    # 1. Print and save the classification report
    print("\n[Cross-Instar Generalization Classification Report]")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    with open(os.path.join(output_dir, 'cross_instar_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Cross-Instar Distribution Shift Challenge ===\n")
        f.write("Source Training: Instar 1-3\n")
        f.write("Target Testing: Instar 4-6\n\n")
        f.write(report)

    # 2. Plot and save the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names,
                yticklabels=class_names)  # use a red palette to reflect the "catastrophic challenge"
    plt.ylabel('True Label (actual late-instar species)')
    plt.xlabel('Predicted Label (predicted from early-instar features)')
    plt.title('Cross-Instar Distribution Shift: Testing on Instars 4-6')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'cross_instar_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Cross-instar confusion matrix saved to '{cm_path}'")
    print(f"✅ Cross-instar text report saved to '{os.path.join(output_dir, 'cross_instar_report.txt')}'")


if __name__ == '__main__':
    # Configuration
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda item: item[1])]

    # 1. Load the cross-instar distribution shift DataLoader
    # train and val sets only have instars 1-3, the test set is all instars 4-6
    print("\nLoading dataset (Cross-Instar Distribution Shift)...")
    dataloaders = get_cross_instar_dataloaders(CSV_FILE, DATA_DIR, batch_size=32, num_workers=0)

    # 2. Initialize the classification model (4-class species prediction)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3. Train (only on instars 1-3)
    model = train_cross_instar_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        DEVICE,
        num_epochs=10,
        output_dir=OUTPUT_DIR
    )

    # 4. Evaluate (directly on instars 4-6)
    best_model_path = os.path.join(OUTPUT_DIR, 'best_cross_instar_model.pth')
    model.load_state_dict(torch.load(best_model_path))

    evaluate_cross_instar_model(model, dataloaders['test'], DEVICE, class_names, output_dir=OUTPUT_DIR)
