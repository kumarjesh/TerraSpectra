"""
TerraSpectra - Week 1 Automated Verification Suite
===================================================
Module: tests.test_week1_pipeline
Verifies:
  1. Hyperspectral cube dimension, band count, and spatial properties.
  2. Memory-safe windowed streaming benchmark (Peak RAM < 150 MB).
  3. Calibrated PCA variance retention (>= 99.0%).
  4. Affine geotransform coordinate roundtrip accuracy (EPSG:4326).
  5. FastAPI endpoint functionality and diagnostic index calculation.
"""

import os
import sys
import json
import tracemalloc
import numpy as np
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing.raster_reader import WindowedRasterReader
from ml.preprocessing.pca_engine import SpectralPCAEngine
from backend.app.main import startup_event, health_check, get_farm_aoi, get_spectral_curve, extract_tile_window, WindowRequest
CUBE_PATH = os.path.join(BASE_DIR, "data", "salinas_hyperspectral_cube.npy")
META_PATH = os.path.join(BASE_DIR, "data", "cube_metadata.json")
PCA_MODEL_PATH = os.path.join(BASE_DIR, "ml", "preprocessing", "pca_model.pkl")
PCA_META_PATH = os.path.join(BASE_DIR, "ml", "preprocessing", "pca_metadata.json")

def test_cube_metadata_and_dimensions():
    """Validates that the generated hyperspectral cube matches sensor specs."""
    assert os.path.exists(CUBE_PATH), "Cube data file missing!"
    assert os.path.exists(META_PATH), "Cube metadata missing!"
    
    with open(META_PATH, "r") as f:
        meta = json.load(f)
        
    assert meta["num_bands"] == 220, f"Expected 220 bands, got {meta['num_bands']}"
    assert meta["grid_shape"] == [220, 512, 512], f"Unexpected shape {meta['grid_shape']}"
    assert meta["crs"] == "EPSG:4326", f"Expected EPSG:4326, got {meta['crs']}"
    assert len(meta["affine_geotransform"]) == 6, "Invalid affine transform"

def test_memory_safe_windowing():
    """Benchmark test: Peak RAM must remain under 150 MB during a 256x256 window read."""
    tracemalloc.start()
    reader = WindowedRasterReader(CUBE_PATH, META_PATH)
    
    tracemalloc.reset_peak()
    tile, transform, meta = reader.read_window(col_off=100, row_off=100, width=256, height=256)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak_mem / (1024 ** 2)
    assert peak_mb < 150.0, f"Memory windowing failed: peak RAM {peak_mb:.2f} MB >= 150.0 MB threshold"
    assert tile.shape == (220, 256, 256), f"Unexpected tile shape {tile.shape}"

def test_pca_variance_retention():
    """Validates that PCA reduces 220 bands to 16 components while retaining >= 99.0% variance."""
    assert os.path.exists(PCA_MODEL_PATH), "PCA model missing!"
    assert os.path.exists(PCA_META_PATH), "PCA metadata missing!"
    
    engine = SpectralPCAEngine.load(PCA_MODEL_PATH, PCA_META_PATH)
    assert engine.n_components == 16, f"Expected 16 components, got {engine.n_components}"
    assert engine.cumulative_variance_ >= 0.990, f"Variance retention {engine.cumulative_variance_*100:.2f}% < 99.0%"
    
    # Test patch transformation
    dummy_patch = np.zeros((220, 32, 32), dtype=np.float32)
    transformed = engine.transform_patch(dummy_patch)
    assert transformed.shape == (16, 32, 32), f"Unexpected transformed shape {transformed.shape}"

def test_geospatial_coordinate_roundtrip():
    """Validates sub-pixel GPS-to-pixel and pixel-to-GPS conversion accuracy."""
    reader = WindowedRasterReader(CUBE_PATH, META_PATH)
    test_coords = [(0, 0), (185, 245), (511, 511), (256, 256)]
    
    for r_orig, c_orig in test_coords:
        lat, lon = reader.pixel_to_gps(r_orig, c_orig)
        r_rec, c_rec = reader.gps_to_pixel(lat, lon)
        assert abs(r_orig - r_rec) <= 1, f"Row mismatch: orig {r_orig} vs recovered {r_rec}"
        assert abs(c_orig - c_rec) <= 1, f"Col mismatch: orig {c_orig} vs recovered {c_rec}"

def test_fastapi_endpoints():
    """Validates core API endpoints and diagnostic classification."""
    startup_event()
    
    # 1. Health
    health = health_check()
    assert health["status"] == "healthy"
    assert health["dataset_ready"] is True
    assert health["pca_engine_ready"] is True
    
    # 2. AOI
    aoi_res = get_farm_aoi()
    assert aoi_res.status_code == 200
    
    # 3. Spectral Curve for Stressed Cluster (Row=185, Col=245)
    curve_res = get_spectral_curve(row=185, col=245)
    assert curve_res["risk_level"] == "High"
    assert "Pre-Symptomatic" in curve_res["diagnosis"]
    assert curve_res["num_bands"] == 220
    assert curve_res["indices"]["red_edge_position_nm"] < 720.0
    
    # 4. Tiled Window with PCA
    win_res = extract_tile_window(WindowRequest(col_offset=50, row_offset=50, width=128, height=128, apply_pca=True))
    assert win_res["success"] is True
    assert win_res["metadata"]["patch_shape"] == [16, 128, 128]
    assert win_res["metadata"]["pca_applied"] is True

if __name__ == "__main__":
    pytest.main(["-v", __file__])
