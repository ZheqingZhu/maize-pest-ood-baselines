import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image
from sklearn.model_selection import train_test_split

# Import the Dataset class
from dataset import MaizeHerbivoryDataset, SPECIES_MAP


def extract_features_and_plot_tsne(csv_path, data_dir, model_path, output_dir='figure'):
    print("🚀 Extracting high-dimensional features for the t-SNE projection...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the model trained in the cross-instar experiment (best illustrates the distribution shift)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Drop the final classification layer to output 2048-d feature vectors
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    # 2. Load the data and stratify-sample 2,000 images (keeps the plot readable)
    df = pd.read_csv(csv_path)
    # Stratified by species and instar so every species and instar is represented
    _, sample_df = train_test_split(df, test_size=2000 / len(df), stratify=df[['species', 'instar']], random_state=42)
    sample_df = sample_df.reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = MaizeHerbivoryDataset(sample_df, data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # 3. Extract features
    features = []
    species_labels = []
    domain_labels = []  # distinguishes early (source) and late (target) stages

    print(f"Extracting 2048-d features of {len(sample_df)} images with ResNet-50...")
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            preds = model(inputs)
            features.extend(preds.cpu().numpy())

            # record species names
            species_strs = [list(SPECIES_MAP.keys())[list(SPECIES_MAP.values()).index(lbl)] for lbl in
                            batch['label'].numpy()]
            species_labels.extend(species_strs)

            # record domain (instars 1-3 = Early, 4-6 = Late)
            domains = ['Early (Instar 1-3)' if inst <= 3 else 'Late (Instar 4-6)' for inst in batch['instar'].numpy()]
            domain_labels.extend(domains)

    features = np.array(features)

    # 4. t-SNE projection (2048-d to 2-d)
    print("Running t-SNE (this may take 1-2 minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 5. Plot with Seaborn
    plot_df = pd.DataFrame({
        'TSNE_1': features_2d[:, 0],
        'TSNE_2': features_2d[:, 1],
        'Species': species_labels,
        'Stage': domain_labels
    })

    # enlarge the canvas to make room for the legend
    plt.figure(figsize=(11, 8))

    # color = species, marker = developmental stage
    sns.scatterplot(
        data=plot_df,
        x='TSNE_1', y='TSNE_2',
        hue='Species',
        style='Stage',
        palette='Set1',
        markers={'Early (Instar 1-3)': 'o', 'Late (Instar 4-6)': '^'},
        s=100, alpha=0.8, edgecolor='w', linewidth=0.5
    )

    # enlarge title and axis labels
    plt.title('t-SNE Feature Space: Cross-Instar Distribution Shift', fontsize=20, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)

    # enlarge tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # enlarge the legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, frameon=True)
    plt.tight_layout()

    output_pdf = os.path.join(output_dir, 'Figure5B_tSNE.pdf')
    plt.savefig(output_pdf, dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure5B_tSNE.png'), dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE figure saved: {output_pdf}")


if __name__ == '__main__':
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'
    # weights of the cross-instar model best show how late-stage features drift out of the source domain
    MODEL_WEIGHTS = 'results/best_cross_instar_model.pth'
    OUTPUT_FOLDER = 'figure'

    extract_features_and_plot_tsne(CSV_FILE, DATA_DIR, MODEL_WEIGHTS, OUTPUT_FOLDER)