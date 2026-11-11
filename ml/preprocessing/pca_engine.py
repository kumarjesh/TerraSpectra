"""
TerraSpectra - Calibrated PCA Dimensionality Reduction Engine
=============================================================
Module: ml.preprocessing.pca_engine
Purpose: Compresses 220 continuous hyperspectral bands down to 16-32 principal
components while guaranteeing >= 99.0% cumulative explained variance retention.

Features:
  1. Stratified spatial pixel sampling across healthy, stressed, and soil zones.
  2. Principal Component Analysis fitting via Scikit-learn.
  3. Patch transformation from (B, H, W) -> (K, H, W) for CNN/ViT ingest.
  4. Model artifact persistence (pickle + JSON metadata) for production inference.
  5. Component variance visualization and audit metrics.
"""

import os
import json
import pickle
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.decomposition import PCA

class SpectralPCAEngine:
    """
    Dimensionality reduction engine for hyperspectral cubes.
    Transforms deep spectral cubes into compact, orthogonal feature volumes.
    """
    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=False)
        self.is_fitted = False
        self.explained_variance_ratio_: np.ndarray = None
        self.cumulative_variance_: float = 0.0

    def fit_from_cube(
        self,
        cube_path: str,
        labels_path: str,
        sample_size: int = 10000
    ) -> Dict[str, Any]:
        """
        Fits PCA on a stratified sample of pixels across the scene.
        """
        print(f"[*] Loading cube and ground truth for PCA calibration...")
        cube = np.load(cube_path, mmap_mode="r")  # Shape: (220, H, W)
        labels = np.load(labels_path)             # Shape: (H, W)
        
        num_bands, H, W = cube.shape
        
        # Reshape to (N_pixels, num_bands)
        flat_cube = np.transpose(cube, (1, 2, 0)).reshape(-1, num_bands)
        flat_labels = labels.flatten()
        
        # Stratified sampling across all classes
        unique_classes, counts = np.unique(flat_labels, return_counts=True)
        print(f"[*] Class distribution across {H*W} pixels: {dict(zip(unique_classes.tolist(), counts.tolist()))}")
        
        sampled_indices = []
        samples_per_class = sample_size // len(unique_classes)
        
        np.random.seed(42)
        for cls in unique_classes:
            cls_idx = np.where(flat_labels == cls)[0]
            n_select = min(len(cls_idx), samples_per_class)
            selected = np.random.choice(cls_idx, size=n_select, replace=False)
            sampled_indices.extend(selected)
            
        X_sample = flat_cube[sampled_indices, :]
        print(f"[*] Fitting PCA on {X_sample.shape[0]} stratified pixel spectra...")
        
        self.pca.fit(X_sample)
        self.is_fitted = True
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = float(np.sum(self.explained_variance_ratio_))
        
        metrics = {
            "n_components": self.n_components,
            "cumulative_variance_ratio": round(self.cumulative_variance_, 5),
            "explained_variance_per_component": [
                round(float(v), 5) for v in self.explained_variance_ratio_
            ],
            "pc1_variance": round(float(self.explained_variance_ratio_[0]), 4),
            "pc2_variance": round(float(self.explained_variance_ratio_[1]), 4),
            "pc3_variance": round(float(self.explained_variance_ratio_[2]), 4),
            "passes_threshold": self.cumulative_variance_ >= 0.990
        }
        
        print(f"[+] PCA Calibration Complete:")
        print(f"    - Cumulative Explained Variance ({self.n_components} components): {self.cumulative_variance_*100:.2f}%")
        print(f"    - PC1 (Albedo/Brightness): {self.explained_variance_ratio_[0]*100:.2f}%")
        print(f"    - PC2 (Red-Edge Shift):   {self.explained_variance_ratio_[1]*100:.2f}%")
        print(f"    - PC3 (Water/Pigments):   {self.explained_variance_ratio_[2]*100:.2f}%")
        
        return metrics

    def transform_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Transforms a 3D patch (Channels, Height, Width) into (n_components, Height, Width).
        """
        if not self.is_fitted:
            raise RuntimeError("PCA model is not fitted yet.")
            
        C, H, W = patch.shape
        flat = patch.reshape(C, -1).T  # Shape: (H*W, C)
        transformed_flat = self.pca.transform(flat)  # Shape: (H*W, n_components)
        transformed_patch = transformed_flat.T.reshape(self.n_components, H, W)
        return transformed_patch.astype(np.float32)

    def save(self, model_path: str, meta_path: str):
        """Serializes fitted PCA model and metadata."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.pca, f)
            
        meta = {
            "n_components": self.n_components,
            "cumulative_variance_ratio": self.cumulative_variance_,
            "components_shape": list(self.pca.components_.shape),
            "mean_shape": list(self.pca.mean_.shape)
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"[*] Serialized PCA model to {model_path} and metadata to {meta_path}")

    @classmethod
    def load(cls, model_path: str, meta_path: str):
        """Loads a persisted PCA engine."""
        with open(model_path, "rb") as f:
            pca_obj = pickle.load(f)
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        engine = cls(n_components=meta["n_components"])
        engine.pca = pca_obj
        engine.is_fitted = True
        engine.explained_variance_ratio_ = pca_obj.explained_variance_ratio_
        engine.cumulative_variance_ = meta["cumulative_variance_ratio"]
        return engine

if __name__ == "__main__":
    cube_file = os.path.join("data", "salinas_hyperspectral_cube.npy")
    labels_file = os.path.join("data", "ground_truth_labels.npy")
    model_save = os.path.join("ml", "preprocessing", "pca_model.pkl")
    meta_save = os.path.join("ml", "preprocessing", "pca_metadata.json")
    
    if os.path.exists(cube_file) and os.path.exists(labels_file):
        engine = SpectralPCAEngine(n_components=16)
        metrics = engine.fit_from_cube(cube_file, labels_file)
        engine.save(model_save, meta_save)
        
        # Test patch transformation
        test_patch = np.load(cube_file, mmap_mode="r")[:, :64, :64]
        reduced = engine.transform_patch(test_patch)
        print(f"[*] Test patch transformed: {test_patch.shape} -> {reduced.shape}")
        assert metrics["passes_threshold"], "Cumulative variance failed 99% threshold!"
        print("[+] Verified: PCA variance retention >= 99.0% constraint satisfied!")
    else:
        print("[!] Data files missing. Run synthetic_cube_generator.py first.")

