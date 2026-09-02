# -*- coding: utf-8 -*-
"""
Grouped (leaf-level) dataset module for biologically decorrelated evaluation.

Data source: grouped_manifest.csv (columns: image_path, species, instar,
days_post_hatching, leaf_id, source). All image_path values are relative to the
repository root (e.g., 'data/frugiperda/1/0/ct_d0_001.jpg' or
'data/supplementary/separata/d04/d4-99.jpg').

Splitting principle: all images of a physical leaf (leaf_id) always reside in a
single partition (train/val/test), eliminating within-leaf leakage by construction.
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SPECIES_MAP = {
    'frugiperda': 0,  # Spodoptera frugiperda
    'litura': 1,      # Spodoptera litura
    'separata': 2,    # Mythimna separata
    'ipsilon': 3      # Agrotis ipsilon
}

MANIFEST = 'grouped_manifest.csv'


class GroupedHerbivoryDataset(Dataset):
    """Same interface as the original MaizeHerbivoryDataset."""

    def __init__(self, dataframe, root_dir='.', transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        rel_path = self.dataframe.loc[idx, 'image_path']
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(SPECIES_MAP[self.dataframe.loc[idx, 'species']], dtype=torch.long)
        return {
            'image': image,
            'label': label,
            'instar': int(self.dataframe.loc[idx, 'instar']),
            'days': int(self.dataframe.loc[idx, 'days_post_hatching']),
            'path': rel_path,
        }


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
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


def load_manifest(manifest_path=MANIFEST):
    df = pd.read_csv(manifest_path)
    df['instar'] = df['instar'].astype(int)
    df['days_post_hatching'] = df['days_post_hatching'].astype(int)
    return df


def get_grouped_fold_dataloaders(df, train_leaves, val_leaves, test_leaves,
                                 batch_size=32, num_workers=4):
    splits = {
        'train': df[df['leaf_id'].isin(train_leaves)],
        'val': df[df['leaf_id'].isin(val_leaves)],
        'test': df[df['leaf_id'].isin(test_leaves)],
    }
    for k, v in splits.items():
        print(f'  [{k}] {len(v)} images, {v["leaf_id"].nunique()} leaves')
    datasets = {
        'train': GroupedHerbivoryDataset(splits['train'], transform=data_transforms['train']),
        'val': GroupedHerbivoryDataset(splits['val'], transform=data_transforms['val_test']),
        'test': GroupedHerbivoryDataset(splits['test'], transform=data_transforms['val_test']),
    }
    return {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train'),
                      num_workers=num_workers)
        for x in ['train', 'val', 'test']
    }


def iter_grouped_folds(df, n_splits=5, base_seed=42):
    """5-fold GroupKFold over leaf_id; per fold yields (train, val, test) leaf lists."""
    gkf = GroupKFold(n_splits=n_splits)
    leaves = df['leaf_id'].values
    for fold, (_, test_idx) in enumerate(gkf.split(df, groups=leaves)):
        test_leaves = sorted(set(leaves[test_idx]))
        rest = sorted(set(leaves) - set(test_leaves))
        gss = GroupShuffleSplit(n_splits=1, test_size=1 / 8,
                                random_state=base_seed + fold)
        rest_df = df[df['leaf_id'].isin(rest)]
        tr_idx, val_idx = next(gss.split(rest_df, groups=rest_df['leaf_id'].values))
        train_leaves = sorted(set(rest_df['leaf_id'].values[tr_idx]))
        val_leaves = sorted(set(rest_df['leaf_id'].values[val_idx]))
        yield fold, train_leaves, val_leaves, test_leaves


def iter_cross_instar_splits(df, n_reps=3, base_seed=42):
    """
    Leaf-disjoint cross-instar splits: source/target leaves are disjoint sets;
    source leaves contribute instar 1-3 images (train/val), target leaves
    contribute instar 4-6 images (test). Repeated n_reps times.
    """
    for rep in range(n_reps):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5,
                                random_state=base_seed + 1000 + rep)
        src_idx, tgt_idx = next(gss.split(df, groups=df['leaf_id'].values))
        src_leaves = sorted(set(df['leaf_id'].values[src_idx]))
        tgt_leaves = sorted(set(df['leaf_id'].values[tgt_idx]))

        src_df = df[df['leaf_id'].isin(src_leaves) & df['instar'].isin([1, 2, 3])]
        tgt_df = df[df['leaf_id'].isin(tgt_leaves) & df['instar'].isin([4, 5, 6])]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                 random_state=base_seed + 2000 + rep)
        tr_idx, val_idx = next(gss2.split(src_df, groups=src_df['leaf_id'].values))
        train_leaves = sorted(set(src_df['leaf_id'].values[tr_idx]))
        val_leaves = sorted(set(src_df['leaf_id'].values[val_idx]))

        train_df = src_df[src_df['leaf_id'].isin(train_leaves)]
        val_df = src_df[src_df['leaf_id'].isin(val_leaves)]
        yield rep, train_df, val_df, tgt_df


def get_dataloaders_from_frames(train_df, val_df, test_df, batch_size=32, num_workers=4):
    datasets = {
        'train': GroupedHerbivoryDataset(train_df, transform=data_transforms['train']),
        'val': GroupedHerbivoryDataset(val_df, transform=data_transforms['val_test']),
        'test': GroupedHerbivoryDataset(test_df, transform=data_transforms['val_test']),
    }
    return {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train'),
                      num_workers=num_workers)
        for x in ['train', 'val', 'test']
    }
