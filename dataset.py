import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Mapping from species name to class ID
SPECIES_MAP = {
    'frugiperda': 0,  # Spodoptera frugiperda
    'litura': 1,  # Spodoptera litura
    'separata': 2,  # Mythimna separata
    'ipsilon': 3  # Agrotis ipsilon (black cutworm)
}


class MaizeHerbivoryDataset(Dataset):
    """
    Custom dataset class for maize herbivory images
    """

    def __init__(self, dataframe, root_dir, transform=None):
        """
        :param dataframe: pandas DataFrame containing image_path, species, instar, days
        :param root_dir: root directory of the images (e.g. 'data_real')
        :param transform: image preprocessing/augmentation operations
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the relative path and join it into an absolute path
        rel_path = self.dataframe.loc[idx, 'image_path']
        img_path = os.path.join(self.root_dir, rel_path)

        # Read the image (convert to RGB in case of grayscale or 4-channel images)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get the species label
        species_str = self.dataframe.loc[idx, 'species']
        label = SPECIES_MAP[species_str]

        # Get other metadata (optional, very useful for multi-task learning or error analysis)
        instar = self.dataframe.loc[idx, 'instar']
        days = self.dataframe.loc[idx, 'days_post_hatching']

        # Convert the label to a tensor
        label = torch.tensor(label, dtype=torch.long)

        # Return a dict so metadata can be retrieved later for t-SNE analysis
        return {
            'image': image,
            'label': label,
            'instar': instar,
            'days': days,
            'path': rel_path
        }


# ================= Data preprocessing definitions =================
# Standard preprocessing for ImageNet pretrained models
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # Additional augmentations (flips, color jitter, etc.) can be added here
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


# ================= Split strategy 1: baseline random split =================
def get_baseline_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4):
    """Basic random split with a 70:10:20 ratio"""
    df = pd.read_csv(csv_path)

    # First split: 70% train, 30% (val+test)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['species'], random_state=42)
    # Second split: val and test 1:2 (i.e. 10% and 20% overall)
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


# ================= Split strategy 2: cross-instar distribution shift split =================
def get_cross_instar_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4):
    """
    Distribution shift challenge:
    Source Domain (train+val): instars 1-3
    Target Domain (test): instars 4-6
    """
    df = pd.read_csv(csv_path)

    # Split into source and target domains by instar
    source_df = df[df['instar'].isin([1, 2, 3])]
    target_df = df[df['instar'].isin([4, 5, 6])]

    # Split the source domain further into train and val sets (e.g. 80% train, 20% val)
    train_df, val_df = train_test_split(source_df, test_size=0.2, stratify=source_df['species'], random_state=42)
    test_df = target_df  # the test set consists entirely of unseen late instars

    print(
        f"[Cross-Instar Split] Source Train(instar 1-3): {len(train_df)}, Source Val(instar 1-3): {len(val_df)}, Target Test(instar 4-6): {len(test_df)}")

    datasets = {
        'train': MaizeHerbivoryDataset(train_df, root_dir, transform=data_transforms['train']),
        'val': MaizeHerbivoryDataset(val_df, root_dir, transform=data_transforms['val_test']),
        'test': MaizeHerbivoryDataset(test_df, root_dir, transform=data_transforms['val_test'])  # NOTE: test set uses target_df
    }

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'val', 'test']
    }
    return dataloaders


# ================= Test code =================
if __name__ == '__main__':
    CSV_FILE = 'metadata.csv'
    DATA_DIR = 'data'  # replace with your actual image root directory

    # 1. Test the baseline DataLoader
    print("--- Building Baseline DataLoaders ---")
    baseline_loaders = get_baseline_dataloaders(CSV_FILE, DATA_DIR, batch_size=16, num_workers=0)

    # Grab one batch to check
    batch = next(iter(baseline_loaders['train']))
    print(f"Batch Image Shape: {batch['image'].shape}")  # expected: [16, 3, 224, 224]
    print(f"Batch Label Shape: {batch['label'].shape}")  # expected: [16]
    print(f"Batch Instar Sample: {batch['instar'][:5]}")

    print("\n--- Building Cross-Instar DataLoaders ---")
    cross_loaders = get_cross_instar_dataloaders(CSV_FILE, DATA_DIR, batch_size=16, num_workers=0)
