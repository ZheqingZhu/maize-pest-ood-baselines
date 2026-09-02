# Maize Leaf Herbivory Dataset: PyTorch Baselines

PyTorch code for the paper: *A Spatio-temporal Image Dataset of Maize Leaf Herbivory by Four Major Pests* (Scientific Data, revised).

**What's new in the revision**: corrected, head-capsule-validated day-to-instar mapping; leaf-level provenance (`leaf_id`) for every image; and biologically **grouped evaluation protocols** (5-fold GroupKFold by leaf, plus leaf-disjoint cross-instar splits) replacing the original random-split benchmarks.

## 1. Data Preparation

The image data and metadata are hosted on Zenodo (v2): DOI `10.5281/zenodo.20327492`.

1. Download and extract `Maize_Leaf_Herbivory_Dataset.zip` from Zenodo.
2. Place the extracted `data/` directory in the root of this repository. The layout is `data/<species>/<instar>/<day>/*.jpg`.
3. `metadata.csv` (included here; identical to the Zenodo version) provides `image_path, species, instar, days_post_hatching, leaf_id` for the 14,642 images.
4. **Optional** (needed only to reproduce the grouped benchmarks exactly): also download `supplementary_provenance_labeled_images.zip` from the same Zenodo record and extract its contents into `data/supplementary/`. `grouped_manifest.csv` (included) covers the resulting 14,579-image provenance-resolved evaluation set.

## 2. Dependencies

```
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy scipy opencv-python pytorch-grad-cam
```

## 3. Scripts

Run from the repository root.

### Grouped evaluation (revised benchmarks, used in the revised paper)

- `train_grouped.py baseline|mtl|mixed_mtl|cross_instar|all --epochs 10` — leaf-grouped 5-fold cross-validation (or 3 leaf-disjoint repetitions for cross-instar), reporting mean ± SD and 95% CI. Requires `grouped_manifest.csv` and the Zenodo `data/` tree.

### Original random-split baselines (superseded; kept for reference)

- `train.py` — standard 4-class species classification (random split).
- `train_mtl.py` — multi-task learning (species + discrete instar).
- `train_mtl_mixed.py` — mixed multi-task learning (species + continuous age regression).
- `train_cross_instar.py` — developmental distribution-shift evaluation (trains on instars 1–3, tests on unseen instars 4–6).

### Visualization

- `plot_tsne.py` — t-SNE projection of penultimate-layer embeddings (early vs. late instars per species).
- `plot_figure6_cam.py` — Grad-CAM attention heatmaps for representative crops of the four species (requires `pytorch-grad-cam`).

## 4. Results

`results/` contains the grouped-evaluation summaries (`*_summary.csv`), the full training configuration (`hyperparameters.json`), and the figure comparing random-split vs. leaf-grouped evaluation.

## License

MIT (code). The dataset is distributed via Zenodo under its own terms.
