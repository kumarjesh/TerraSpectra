"""
TerraSpectra - Week 2: Baseline Model Training & Spatial Evaluation
===================================================================
Module: ml.training.train_baseline
Architecture: Calibrated Baseline Classifier (Random Forest & 2D-CNN on PCA Bands)
Inputs: 16-component PCA spectral volume (derived from 220-band cube)
Target: 3-class classification
  - Class 0: Non-crop Soil / Access Tracks
  - Class 1: Healthy Crop Canopy
  - Class 2: Pre-Symptomatic Fungal Infection (High Risk)

Critical Geospatial ML Constraint:
  SPATIAL CROSS-VALIDATION / FIELD-LEVEL SPLIT:
  Standard random pixel splitting causes severe data leakage because adjacent pixels
  share spatial autocorrelation. We enforce spatially partitioned train/val splits
  (e.g., eastern half vs. western half / field blocks).
"""

import os
import sys
import json
import pickle
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ml.preprocessing.pca_engine import SpectralPCAEngine
CUBE_PATH = os.path.join(BASE_DIR, "data", "salinas_hyperspectral_cube.npy")
LABELS_PATH = os.path.join(BASE_DIR, "data", "ground_truth_labels.npy")
META_PATH = os.path.join(BASE_DIR, "data", "cube_metadata.json")
PCA_MODEL_PATH = os.path.join(BASE_DIR, "ml", "preprocessing", "pca_model.pkl")
PCA_META_PATH = os.path.join(BASE_DIR, "ml", "preprocessing", "pca_metadata.json")
BASELINE_SAVE_PATH = os.path.join(BASE_DIR, "ml", "training", "baseline_model.pkl")

# ---------------------------------------------------------------------------
# 1. Spatial Dataset Preparation (Field-Separated Train/Test Split)
# ---------------------------------------------------------------------------
def prepare_spatial_datasets(test_spatial_ratio: float = 0.25):
    """
    Loads the 220-band cube, applies PCA reduction to 16 bands, and partitions
    pixels using a spatial block holdout (preserving geographic independence).
    """
    print("[*] Loading hyperspectral cube and ground truth labels...")
    cube = np.load(CUBE_PATH, mmap_mode="r")  # (220, 512, 512)
    labels = np.load(LABELS_PATH)             # (512, 512)
    
    pca_engine = SpectralPCAEngine.load(PCA_MODEL_PATH, PCA_META_PATH)
    print(f"[*] Projecting 220 bands -> {pca_engine.n_components} PCA components...")
    
    # Transform full scene to (16, 512, 512)
    pca_volume = pca_engine.transform_patch(np.array(cube, dtype=np.float32))
    
    num_components, H, W = pca_volume.shape
    features = np.transpose(pca_volume, (1, 2, 0)) # (512, 512, 16)
    
    # Spatial Block Partitioning (Field-separated: hold out a spatial quad for validation)
    # Train: Rows 0..384; Test: Rows 384..512 (plus spatial strip around infection zone to test generalization)
    split_row = int(H * (1.0 - test_spatial_ratio))
    
    # Create spatial masks
    train_mask = np.zeros((H, W), dtype=bool)
    test_mask = np.zeros((H, W), dtype=bool)
    
    # Ensure infection cluster (centered at row=185, col=245) has representative spatial samples in both
    train_mask[:split_row, :] = True
    test_mask[split_row:, :] = True
    
    # Also hold out the right half of the infection cluster for pure spatial test
    test_mask[170:200, 245:270] = True
    train_mask[170:200, 245:270] = False
    
    X_train = features[train_mask]
    y_train = labels[train_mask]
    
    X_test = features[test_mask]
    y_test = labels[test_mask]
    
    print(f"[+] Spatial Split Complete:")
    print(f"    - Train Samples: {X_train.shape[0]} pixels across field blocks")
    print(f"    - Test Samples:  {X_test.shape[0]} pixels (spatially separated)")
    print(f"    - Train Class Counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"    - Test Class Counts:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_train, y_train, X_test, y_test, pca_volume

# ---------------------------------------------------------------------------
# 2. Baseline Model Architecture: Calibrated Random Forest & 2D-CNN
# ---------------------------------------------------------------------------
class Baseline2DCNN(nn.Module):
    """
    Lightweight 2D-CNN processing PCA spatial patches (16 channels x Patch_H x Patch_W).
    """
    def __init__(self, in_channels: int = 16, num_classes: int = 3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

def train_baseline_model():
    """
    Trains the Week 2 baseline classifier and saves model artifacts and performance metrics.
    """
    X_train, y_train, X_test, y_test, pca_volume = prepare_spatial_datasets()
    
    # Balanced Random Forest Baseline
    print("\n[*] Training Week 2 Baseline Random Forest Classifier on PCA bands...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    
    # Sub-sample train set if larger than 40k to maintain fast training (<10s)
    if len(X_train) > 40000:
        sub_idx = np.random.choice(len(X_train), size=40000, replace=False)
        X_fit, y_fit = X_train[sub_idx], y_train[sub_idx]
    else:
        X_fit, y_fit = X_train, y_train
        
    clf.fit(X_fit, y_fit)
    
    # Evaluate on Spatially Separated Test Set
    print("[*] Evaluating baseline on holdout spatial field blocks...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_disease = f1_score(y_test, y_pred, average=None)[2] if 2 in y_test else 0.0
    
    print("\n" + "="*60)
    print("TERRASPECTRA WEEK 2 BASELINE EVALUATION REPORT")
    print("="*60)
    target_names = ["0: Non-crop Soil", "1: Healthy Crop", "2: Pre-Symptomatic Stress"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_mat)
    print(f"\n[+] Key Performance Indicators (KPIs):")
    print(f"    - Micro F1-Score:             {f1_micro * 100:.2f}%")
    print(f"    - Macro F1-Score:             {f1_macro * 100:.2f}%")
    print(f"    - Pre-Symptomatic F1-Score:   {f1_disease * 100:.2f}%")
    print("="*60)
    
    # Save Baseline Model
    os.makedirs(os.path.dirname(BASELINE_SAVE_PATH), exist_ok=True)
    with open(BASELINE_SAVE_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[+] Serialized baseline classifier to {BASELINE_SAVE_PATH}")
    
    metrics = {
        "model_type": "Balanced Random Forest (100 estimators, max_depth=12)",
        "input_features": 16,
        "spatial_evaluation": "Field-Block Holdout",
        "f1_micro": round(float(f1_micro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_disease_class": round(float(f1_disease), 4),
        "confusion_matrix": conf_mat.tolist()
    }
    
    metrics_path = os.path.join(BASE_DIR, "ml", "training", "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved evaluation metrics to {metrics_path}")
    
    return clf, metrics

if __name__ == "__main__":
    train_baseline_model()
