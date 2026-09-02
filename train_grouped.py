# -*- coding: utf-8 -*-
"""
Grouped (leaf-level GroupKFold) technical validation experiments, addressing
reviewer concerns on data leakage (R1.6/1.7, R3.1/3.2/3.3/3.10/3.13/3.14).

Four benchmarks (same models/hyperparameters as the original Technical Validation):
  1. baseline   : ResNet-50 species classification, 5-fold GroupKFold
  2. mtl        : dual-head MTL (species + instar), 5-fold GroupKFold
  3. mixed_mtl  : species classification + age regression, 5-fold GroupKFold
  4. cross_instar: train on instars 1-3, test on instars 4-6, 3 leaf-grouped
                   repetitions (source/target leaves disjoint; stricter than the original)

Full classification reports or regression metrics are recorded per fold/repetition and
summarized as mean +/- SD with 95% CI (t-distribution).
Usage: python3 train_grouped.py [baseline|mtl|mixed_mtl|cross_instar|all] [--epochs 10]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from torchvision import models

from dataset_grouped import (SPECIES_MAP, get_dataloaders_from_frames,
                             get_grouped_fold_dataloaders, iter_cross_instar_splits,
                             iter_grouped_folds, load_manifest)

RESULTS_DIR = 'results_grouped'
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 8


# ================= Model definitions (identical to the original code) =================
class MultiTaskResNet(nn.Module):
    def __init__(self, num_species=4, num_instars=6):
        super().__init__()
        shared = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(shared.children())[:-1])
        num_ftrs = shared.fc.in_features
        self.fc_species = nn.Linear(num_ftrs, num_species)
        self.fc_instar = nn.Linear(num_ftrs, num_instars)

    def forward(self, x):
        x = torch.flatten(self.features(x), 1)
        return self.fc_species(x), self.fc_instar(x)


class MixedMTLResNet(nn.Module):
    def __init__(self, num_species=4):
        super().__init__()
        shared = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(shared.children())[:-1])
        num_ftrs = shared.fc.in_features
        self.fc_species = nn.Linear(num_ftrs, num_species)
        self.fc_days = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = torch.flatten(self.features(x), 1)
        return self.fc_species(x), self.fc_days(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ================= Training loops =================
def train_classification(model, loaders, device, epochs, seed):
    """Single-classification-head training (baseline / cross_instar). Returns the best weights."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            corrects, total = 0, 0
            for batch in loaders[phase]:
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
                corrects += torch.sum(preds == labels).item()
                total += labels.size(0)
            acc = corrects / total
            if phase == 'val' and acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  epoch {epoch+1}/{epochs} done, val acc {acc:.4f}', flush=True)
    return best_state, best_acc


def train_mtl(model, loaders, device, epochs, seed):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            run_loss, total = 0.0, 0
            for batch in loaders[phase]:
                inputs = batch['image'].to(device)
                ls = batch['label'].to(device)
                li = (batch['instar'] - 1).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    os_, oi = model(inputs)
                    loss = criterion(os_, ls) + criterion(oi, li)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                run_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
            el = run_loss / total
            if phase == 'val' and el < best_loss:
                best_loss = el
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  epoch {epoch+1}/{epochs} done, val loss {el:.4f}', flush=True)
    return best_state


def train_mixed(model, loaders, device, epochs, seed, weight_reg=0.1):
    crit_cls, crit_reg = nn.CrossEntropyLoss(), nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            run_loss, total = 0.0, 0
            for batch in loaders[phase]:
                inputs = batch['image'].to(device)
                ls = batch['label'].to(device)
                ld = batch['days'].float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    os_, od = model(inputs)
                    loss = crit_cls(os_, ls) + weight_reg * crit_reg(od, ld)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                run_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
            el = run_loss / total
            if phase == 'val' and el < best_loss:
                best_loss = el
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  epoch {epoch+1}/{epochs} done, val loss {el:.4f}', flush=True)
    return best_state


# ================= Evaluation =================
def eval_classification(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['image'].to(device))
            _, p = torch.max(outputs, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(batch['label'].numpy())
    species_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda x: x[1])]
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'report': classification_report(labels, preds, labels=list(range(4)),
                                        target_names=species_names,
                                        digits=4, output_dict=True, zero_division=0),
    }


def eval_mtl(model, loader, device):
    model.eval()
    ps, ls_, pi, li_ = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            os_, oi = model(batch['image'].to(device))
            ps.extend(torch.max(os_, 1)[1].cpu().numpy())
            pi.extend(torch.max(oi, 1)[1].cpu().numpy())
            ls_.extend(batch['label'].numpy())
            li_.extend((batch['instar'] - 1).numpy())
    species_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda x: x[1])]
    instar_names = [f'Instar_{i}' for i in range(1, 7)]
    return {
        'species_accuracy': accuracy_score(ls_, ps),
        'species_macro_f1': f1_score(ls_, ps, average='macro', zero_division=0),
        'instar_accuracy': accuracy_score(li_, pi),
        'instar_macro_f1': f1_score(li_, pi, average='macro', zero_division=0),
        'species_report': classification_report(ls_, ps, labels=list(range(4)),
                                                target_names=species_names,
                                                digits=4, output_dict=True, zero_division=0),
        'instar_report': classification_report(li_, pi, labels=list(range(6)),
                                               target_names=instar_names,
                                               digits=4, output_dict=True, zero_division=0),
    }


def eval_mixed(model, loader, device):
    model.eval()
    ps, ls_, pd_, ld_ = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            os_, od = model(batch['image'].to(device))
            ps.extend(torch.max(os_, 1)[1].cpu().numpy())
            pd_.extend(od.cpu().numpy().flatten())
            ls_.extend(batch['label'].numpy())
            ld_.extend(batch['days'].numpy())
    species_names = [k for k, v in sorted(SPECIES_MAP.items(), key=lambda x: x[1])]
    return {
        'species_accuracy': accuracy_score(ls_, ps),
        'species_macro_f1': f1_score(ls_, ps, average='macro', zero_division=0),
        'mae': mean_absolute_error(ld_, pd_),
        'rmse': float(np.sqrt(mean_squared_error(ld_, pd_))),
        'r2': r2_score(ld_, pd_),
        'species_report': classification_report(ls_, ps, labels=list(range(4)),
                                                target_names=species_names,
                                                digits=4, output_dict=True, zero_division=0),
    }


# ================= Summary statistics =================
def summarize(metric_lists, out_path):
    """metric_lists: {metric: [fold values]}; outputs mean +/- SD and 95% CI (t-distribution)."""
    rows = []
    for m, vals in metric_lists.items():
        vals = np.array(vals, dtype=float)
        n = len(vals)
        mean, sd = vals.mean(), vals.std(ddof=1)
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n)) if n > 1 else (mean, mean)
        rows.append({'metric': m, 'n': n, 'mean': mean, 'std': sd,
                     'ci95_low': ci[0], 'ci95_high': ci[1]})
    sdf = pd.DataFrame(rows)
    sdf.to_csv(out_path, index=False)
    print(sdf.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return sdf


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=str)


# ================= The four benchmarks =================
def run_baseline(df, device, epochs):
    print('\n========== Benchmark 1: Baseline species classification (5-fold GroupKFold) ==========')
    fold_metrics = {'accuracy': [], 'macro_f1': []}
    details = []
    for fold, tr, va, te in iter_grouped_folds(df):
        print(f'\n--- Fold {fold+1}/5 ---')
        set_seed(42 + fold)
        loaders = get_grouped_fold_dataloaders(df, tr, va, te, BATCH_SIZE, NUM_WORKERS)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = model.to(device)
        state, _ = train_classification(model, loaders, device, epochs, 42 + fold)
        model.load_state_dict(state)
        model = model.to(device)
        res = eval_classification(model, loaders['test'], device)
        print(f'fold {fold+1}: acc={res["accuracy"]:.4f} macroF1={res["macro_f1"]:.4f}')
        fold_metrics['accuracy'].append(res['accuracy'])
        fold_metrics['macro_f1'].append(res['macro_f1'])
        details.append({'fold': fold, 'n_test': len(loaders['test'].dataset), **res})
        save_json(details, os.path.join(RESULTS_DIR, 'baseline_folds.json'))
    summarize(fold_metrics, os.path.join(RESULTS_DIR, 'baseline_summary.csv'))


def run_mtl(df, device, epochs):
    print('\n========== Benchmark 2: MTL species+instar (5-fold GroupKFold) ==========')
    fold_metrics = {'species_accuracy': [], 'species_macro_f1': [],
                    'instar_accuracy': [], 'instar_macro_f1': []}
    details = []
    for fold, tr, va, te in iter_grouped_folds(df):
        print(f'\n--- Fold {fold+1}/5 ---')
        set_seed(42 + fold)
        loaders = get_grouped_fold_dataloaders(df, tr, va, te, BATCH_SIZE, NUM_WORKERS)
        model = MultiTaskResNet().to(device)
        state = train_mtl(model, loaders, device, epochs, 42 + fold)
        model.load_state_dict(state)
        model = model.to(device)
        res = eval_mtl(model, loaders['test'], device)
        print(f'fold {fold+1}: sp_acc={res["species_accuracy"]:.4f} instar_acc={res["instar_accuracy"]:.4f}')
        for k in fold_metrics:
            fold_metrics[k].append(res[k])
        details.append({'fold': fold, 'n_test': len(loaders['test'].dataset), **res})
        save_json(details, os.path.join(RESULTS_DIR, 'mtl_folds.json'))
    summarize(fold_metrics, os.path.join(RESULTS_DIR, 'mtl_summary.csv'))


def run_mixed(df, device, epochs):
    print('\n========== Benchmark 3: Mixed-MTL classification+regression (5-fold GroupKFold) ==========')
    fold_metrics = {'species_accuracy': [], 'species_macro_f1': [],
                    'mae': [], 'rmse': [], 'r2': []}
    details = []
    for fold, tr, va, te in iter_grouped_folds(df):
        print(f'\n--- Fold {fold+1}/5 ---')
        set_seed(42 + fold)
        loaders = get_grouped_fold_dataloaders(df, tr, va, te, BATCH_SIZE, NUM_WORKERS)
        model = MixedMTLResNet().to(device)
        state = train_mixed(model, loaders, device, epochs, 42 + fold)
        model.load_state_dict(state)
        model = model.to(device)
        res = eval_mixed(model, loaders['test'], device)
        print(f'fold {fold+1}: sp_acc={res["species_accuracy"]:.4f} MAE={res["mae"]:.4f} R2={res["r2"]:.4f}')
        for k in fold_metrics:
            fold_metrics[k].append(res[k])
        details.append({'fold': fold, 'n_test': len(loaders['test'].dataset), **res})
        save_json(details, os.path.join(RESULTS_DIR, 'mixed_mtl_folds.json'))
    summarize(fold_metrics, os.path.join(RESULTS_DIR, 'mixed_mtl_summary.csv'))


def run_cross_instar(df, device, epochs):
    print('\n========== Benchmark 4: Cross-instar shift (3 leaf-grouped repetitions) ==========')
    rep_metrics = {'accuracy': [], 'macro_f1': []}
    details = []
    for rep, tr_df, va_df, te_df in iter_cross_instar_splits(df):
        print(f'\n--- Rep {rep+1}/3 ---')
        print(f'  source train {len(tr_df)}, source val {len(va_df)}, target test {len(te_df)}')
        set_seed(42 + rep)
        loaders = get_dataloaders_from_frames(tr_df, va_df, te_df, BATCH_SIZE, NUM_WORKERS)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = model.to(device)
        state, src_acc = train_classification(model, loaders, device, epochs, 42 + rep)
        model.load_state_dict(state)
        model = model.to(device)
        res = eval_classification(model, loaders['test'], device)
        print(f'rep {rep+1}: source_val_acc={src_acc:.4f} target_acc={res["accuracy"]:.4f}')
        rep_metrics['accuracy'].append(res['accuracy'])
        rep_metrics['macro_f1'].append(res['macro_f1'])
        details.append({'rep': rep, 'n_test': int(len(te_df)),
                        'source_val_acc': src_acc, **res})
        save_json(details, os.path.join(RESULTS_DIR, 'cross_instar_reps.json'))
    summarize(rep_metrics, os.path.join(RESULTS_DIR, 'cross_instar_summary.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['baseline', 'mtl', 'mixed_mtl',
                                               'cross_instar', 'all'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, epochs: {args.epochs}')

    df = load_manifest()
    print(f'manifest: {len(df)} images, {df["leaf_id"].nunique()} leaves')

    # Hyperparameter record (in response to reviewer R3.15)
    save_json({'batch_size': BATCH_SIZE, 'lr': LR, 'optimizer': 'Adam',
               'epochs': args.epochs, 'init': 'ImageNet1K_V1',
               'input': '224x224 resize (anisotropic, consistent with the original)',
               'augmentation': 'RandomHorizontalFlip + RandomVerticalFlip',
               'n_folds': 5, 'cross_instar_reps': 3, 'base_seed': 42,
               'mixed_mtl_reg_weight': 0.1},
              os.path.join(RESULTS_DIR, 'hyperparameters.json'))

    if args.experiment in ('baseline', 'all'):
        run_baseline(df, device, args.epochs)
    if args.experiment in ('mtl', 'all'):
        run_mtl(df, device, args.epochs)
    if args.experiment in ('mixed_mtl', 'all'):
        run_mixed(df, device, args.epochs)
    if args.experiment in ('cross_instar', 'all'):
        run_cross_instar(df, device, args.epochs)
    print('\nALL DONE')


if __name__ == '__main__':
    main()
