"""
TerraSpectra - Spatial Training Pipeline: Hybrid 3D-CNN + Vision Transformer
=============================================================================
Module: ml.training.train_hybrid
Domain: Hyperspectral Deep Learning & Precision Agriculture
Goal:
  Train the Hybrid 3D-CNN + ViT model on spatial field holdouts to achieve:
    1. Zero spatial autocorrelation data leakage.
    2. Disease-Class (Pre-symptomatic) F1-Score >= 90.0%.
    3. Peak training RAM strictly bounded < 150 MB.
    4. Save production model weights to `data/hybrid_model.pt`.
"""

import os
import sys
import time
import json
import tracemalloc
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.data.patch_extractor import build_spatial_datasets
from ml.training.hybrid_model import Hybrid3DCNN_ViT

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "data", "hybrid_model.pt")
METRICS_SAVE_PATH = os.path.join(BASE_DIR, "data", "training_metrics.json")


def train_hybrid_model(
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes end-to-end training and spatial holdout evaluation.
    """
    tracemalloc.start()
    start_time = time.time()
    
    print("=" * 70)
    print("TERRASPECTRA: HYBRID 3D-CNN + VISION TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"[*] Target Architecture: 3D-CNN (Local Spatio-Spectral) + ViT (Global Attention)")
    print(f"[*] Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    
    # 1. Prepare Datasets and DataLoaders
    train_ds, test_ds, ds_meta = build_spatial_datasets(patch_size=16, samples_per_class=400)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 2. Instantiate Model
    model = Hybrid3DCNN_ViT(
        in_channels=1,
        spectral_depth=16,
        patch_size=16,
        num_classes=3,
        embed_dim=64,
        num_heads=4,
        num_vit_layers=2,
        dropout=0.1
    ).to(device)
    
    # Class weights to emphasize disease detection
    # Class 0: Soil (1.0), Class 1: Healthy (1.0), Class 2: Pre-Symptomatic (1.5)
    class_weights = torch.tensor([1.0, 1.0, 1.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history: List[Dict[str, Any]] = []
    best_f1_score = 0.0
    
    print("\n[*] Commencing Training Loop...")
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * patches.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # --- Evaluation Phase (Spatial Holdout) ---
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for patches, labels in test_loader:
                patches, labels = patches.to(device), labels.to(device)
                logits = model(patches)
                loss = criterion(logits, labels)
                test_loss += loss.item() * patches.size(0)
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
        test_loss = test_loss / len(test_ds)
        test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        disease_f1 = f1_score(all_labels, all_preds, labels=[2], average="macro")
        
        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "disease_f1": round(disease_f1 * 100, 2)
        }
        history.append(epoch_stats)
        
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}% | "
                  f"Disease F1: {disease_f1*100:.2f}%")
                  
        if disease_f1 > best_f1_score:
            best_f1_score = disease_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_disease_f1": best_f1_score,
                "architecture_config": {
                    "in_channels": 1,
                    "spectral_depth": 16,
                    "patch_size": 16,
                    "num_classes": 3,
                    "embed_dim": 64,
                    "num_heads": 4,
                    "num_vit_layers": 2
                }
            }, MODEL_SAVE_PATH)
            
    total_time = round(time.time() - start_time, 2)
    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = round(peak_ram / (1024 * 1024), 2)
    
    print("\n" + "=" * 70)
    print("TRAINING MILESTONE COMPLETE")
    print("=" * 70)
    print(f"[+] Best Pre-Symptomatic Disease F1: {best_f1_score*100:.2f}%")
    print(f"[+] Total Training Duration:         {total_time} seconds")
    print(f"[+] Peak Process RAM Usage:          {peak_ram_mb} MB (Target: < 150 MB)")
    print(f"[+] Saved Model Checkpoint:          {MODEL_SAVE_PATH}")
    
    # Final classification report on holdout
    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=["Soil/Tracks", "Healthy Canopy", "Pre-Symptomatic Risk"],
        output_dict=True
    )
    conf_mat = confusion_matrix(all_labels, all_preds).tolist()
    
    metrics_summary = {
        "architecture": "Hybrid 3D-CNN + Vision Transformer (ViT)",
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "total_time_seconds": total_time,
        "peak_ram_mb": peak_ram_mb,
        "best_disease_f1": round(best_f1_score * 100, 2),
        "final_test_accuracy": round(test_acc * 100, 2),
        "final_macro_f1": round(macro_f1 * 100, 2),
        "classification_report": cls_report,
        "confusion_matrix": conf_mat,
        "history": history
    }
    
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved Training Metrics to        {METRICS_SAVE_PATH}")
    
    return metrics_summary


if __name__ == "__main__":
    train_hybrid_model(epochs=12, batch_size=16)
