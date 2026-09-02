import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Grad-CAM core components
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def auto_find_exact_paths(log_csv_path, meta_csv_path, targets):
    """
    Reverse-lookup of exact image paths: first search the candidates log (the pool
    that was visually inspected), then fall back to metadata.csv.
    """
    exact_paths = {}

    # 1. Prefer the candidates log (most precise, it is the inspected candidate pool)
    if os.path.exists(log_csv_path):
        log_df = pd.read_csv(log_csv_path)
        for sp, base in targets.items():
            # regex-match paths ending with /basename.jpg or .png
            matches = log_df[log_df[sp].str.contains(f"/{base}\\.(jpg|png|jpeg)$", regex=True, na=False, case=False)][
                sp].unique()
            if len(matches) > 0:
                exact_paths[sp] = matches[0]

    # 2. Otherwise search metadata.csv for late-instar entries
    if len(exact_paths) < 4 and os.path.exists(meta_csv_path):
        meta_df = pd.read_csv(meta_csv_path)
        for sp, base in targets.items():
            if sp not in exact_paths:
                subset = meta_df[(meta_df['species'] == sp) & (meta_df['instar'] >= 4)]
                matches = subset[
                    subset['image_path'].str.contains(f"/{base}\\.(jpg|png|jpeg)$", regex=True, na=False, case=False)][
                    'image_path'].unique()
                if len(matches) > 0:
                    exact_paths[sp] = matches[0]

    return exact_paths


def generate_smart_gradcam(data_dir, model_path, output_dir='figure'):
    print("🤖 Starting the path reverse-lookup module...")

    # the four base filenames specified for the figure
    selected_basenames = {
        'frugiperda': 'ct_d7_062',
        'litura': 'xw_d11_132',
        'separata': '110',
        'ipsilon': '105'
    }

    # resolve full paths automatically
    log_csv = os.path.join(output_dir, 'candidates', 'candidates_log.csv')
    meta_csv = 'metadata.csv'
    exact_paths = auto_find_exact_paths(log_csv, meta_csv, selected_basenames)

    # check that all were found
    for sp, base in selected_basenames.items():
        if sp not in exact_paths:
            print(f"❌ Fatal: could not locate image {base} for {sp}!")
            return
        else:
            print(f"✅ Locked path for {sp}: {exact_paths[sp]}")

    print("\n🚀 Loading the model and generating the publication figure...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Model weights not found: {model_path}!")
        return

    model = model.to(device)
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    species_list = ['frugiperda', 'litura', 'separata', 'ipsilon']
    species_names_latex = {
        'frugiperda': 'S. frugiperda',
        'litura': 'S. litura',
        'separata': 'M. separata',
        'ipsilon': 'A. ipsilon'
    }

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.05, hspace=0.25)

    for col, species in enumerate(species_list):
        img_path = os.path.join(data_dir, exact_paths[species])

        # load and resize the original image
        pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
        rgb_img = np.float32(pil_img) / 255
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # generate the CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # plot the original
        ax_orig = axes[0, col]
        ax_orig.imshow(rgb_img)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        ax_orig.set_title(f'$\\mathit{{{species_names_latex[species]}}}$\nOriginal Crop', fontsize=18, pad=15)

        # plot the heatmap
        ax_cam = axes[1, col]
        ax_cam.imshow(cam_image)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        ax_cam.set_title('Grad-CAM Attention', fontsize=18, pad=15)

    # export
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = os.path.join(output_dir, 'Figure6_Final_GradCAM.pdf')
    output_png = os.path.join(output_dir, 'Figure6_Final_GradCAM.png')

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')

    print(f"\n🎉 Done! Figure saved to:\n 📄 {output_pdf}")


if __name__ == '__main__':
    DATA_DIR = 'data'
    OUTPUT_FOLDER = 'figure'
    MODEL_WEIGHTS = 'results/resnet50_baseline.pth'

    generate_smart_gradcam(DATA_DIR, MODEL_WEIGHTS, OUTPUT_FOLDER)