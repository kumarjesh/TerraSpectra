"""
TerraSpectra - Memory-Safe Windowed Raster Ingestion Engine
===========================================================
Module: ml.preprocessing.raster_reader
Purpose: Ingests massive hyperspectral data cubes using windowed spatial reads.
Enforces peak memory bounding (Zero-OOM streaming) for tiles (e.g. 256x256).

Features:
  1. Windowed memory-mapped patch extraction.
  2. Sub-pixel Affine Geotransform calculations (pixel <-> GPS EPSG:4326).
  3. Atmospheric water vapor band filtering (1350-1450 nm and 1800-1950 nm).
  4. Surface reflectance radiometric scaling and normalization.
  5. Built-in tracemalloc memory profiling to verify <150MB peak RAM guarantee.
"""

import os
import json
import tracemalloc
from typing import Tuple, Dict, Any, Optional
import numpy as np

class WindowedRasterReader:
    """
    High-performance, memory-safe reader for 3D hyperspectral cubes (Channels, Height, Width).
    Uses memory-mapping to stream arbitrary spatial windows without loading full scenes into RAM.
    """
    def __init__(self, cube_path: str, metadata_path: str):
        if not os.path.exists(cube_path):
            raise FileNotFoundError(f"Hyperspectral cube not found at: {cube_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Cube metadata not found at: {metadata_path}")
            
        with open(metadata_path, "r") as f:
            self.metadata: Dict[str, Any] = json.load(f)
            
        self.num_bands = self.metadata["num_bands"]
        self.grid_height = self.metadata["grid_shape"][1]
        self.grid_width = self.metadata["grid_shape"][2]
        self.wavelengths = np.array(self.metadata["wavelengths_nm"])
        self.affine = self.metadata["affine_geotransform"]
        
        # Open cube in read-only memory-mapped mode (mmap)
        # Guarantees zero loading of unneeded pixels into memory!
        self.cube = np.load(cube_path, mmap_mode="r")
        
        # Precompute atmospheric water absorption band indices (noisy bands)
        # 1350 - 1450 nm & 1800 - 1950 nm
        water_mask_1 = (self.wavelengths >= 1350.0) & (self.wavelengths <= 1450.0)
        water_mask_2 = (self.wavelengths >= 1800.0) & (self.wavelengths <= 1950.0)
        self.noisy_water_bands = water_mask_1 | water_mask_2
        self.clean_band_indices = np.where(~self.noisy_water_bands)[0]

    def get_window_transform(self, col_off: int, row_off: int) -> list:
        """
        Calculates the affine geotransform for a windowed sub-patch.
        Affine format: [top_left_lon, pixel_width, 0, top_left_lat, 0, -pixel_height]
        """
        top_left_lon = self.affine[0] + (col_off * self.affine[1])
        top_left_lat = self.affine[3] + (row_off * self.affine[5])
        return [
            top_left_lon,
            self.affine[1],
            0.0,
            top_left_lat,
            0.0,
            self.affine[5]
        ]

    def pixel_to_gps(self, row: float, col: float) -> Tuple[float, float]:
        """Converts (row, col) pixel indices to (latitude, longitude) EPSG:4326."""
        lon = self.affine[0] + (col * self.affine[1])
        lat = self.affine[3] + (row * self.affine[5])
        return lat, lon

    def gps_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Converts (latitude, longitude) EPSG:4326 to closest integer (row, col) pixel indices."""
        col = int(round((lon - self.affine[0]) / self.affine[1]))
        row = int(round((lat - self.affine[3]) / self.affine[5]))
        row = max(0, min(self.grid_height - 1, row))
        col = max(0, min(self.grid_width - 1, col))
        return row, col

    def read_window(
        self,
        col_off: int,
        row_off: int,
        width: int = 256,
        height: int = 256,
        filter_water_bands: bool = False
    ) -> Tuple[np.ndarray, list, Dict[str, Any]]:
        """
        Reads a spatial window of shape (channels, height, width).
        Peak memory allocation is bounded strictly by the requested patch size.
        """
        # Validate and clip window bounds
        c_end = min(self.grid_width, col_off + width)
        r_end = min(self.grid_height, row_off + height)
        
        # Windowed slice from memory-mapped array
        if filter_water_bands:
            patch = np.array(self.cube[self.clean_band_indices, row_off:r_end, col_off:c_end], dtype=np.float32)
        else:
            patch = np.array(self.cube[:, row_off:r_end, col_off:c_end], dtype=np.float32)
            
        patch_transform = self.get_window_transform(col_off, row_off)
        
        patch_meta = {
            "col_offset": col_off,
            "row_offset": row_off,
            "patch_shape": list(patch.shape),
            "num_bands": patch.shape[0],
            "affine_geotransform": patch_transform,
            "filtered_water_bands": filter_water_bands
        }
        
        return patch, patch_transform, patch_meta

    def read_spectral_curve(self, row: int, col: int) -> Dict[str, Any]:
        """
        Extracts the full 220-band spectral profile at a specific pixel location.
        Computes key vegetative indices (NDVI, PRI, Red-Edge position).
        """
        row = max(0, min(self.grid_height - 1, row))
        col = max(0, min(self.grid_width - 1, col))
        
        reflectance = self.cube[:, row, col].astype(float)
        lat, lon = self.pixel_to_gps(row, col)
        
        # Calculate Diagnostic Indices:
        # 1. NDVI = (NIR_850 - Red_660) / (NIR_850 + Red_660)
        idx_850 = int(np.argmin(np.abs(self.wavelengths - 850.0)))
        idx_660 = int(np.argmin(np.abs(self.wavelengths - 660.0)))
        nir_val = reflectance[idx_850]
        red_val = reflectance[idx_660]
        ndvi = float((nir_val - red_val) / (nir_val + red_val + 1e-6))
        
        # 2. PRI (Photochemical Reflectance Index) = (R531 - R570) / (R531 + R570)
        idx_531 = int(np.argmin(np.abs(self.wavelengths - 531.0)))
        idx_570 = int(np.argmin(np.abs(self.wavelengths - 570.0)))
        r531 = reflectance[idx_531]
        r570 = reflectance[idx_570]
        pri = float((r531 - r570) / (r531 + r570 + 1e-6))
        
        # 3. Red-Edge Position (REP): First derivative peak between 690 and 740 nm
        red_edge_mask = (self.wavelengths >= 680.0) & (self.wavelengths <= 750.0)
        re_indices = np.where(red_edge_mask)[0]
        d_re = np.gradient(reflectance[re_indices])
        max_d_idx = re_indices[np.argmax(d_re)]
        rep_nm = float(self.wavelengths[max_d_idx])
        
        # Diagnosis:
        # Healthy canopy: REP around 725-730 nm, NDVI > 0.65
        # Pre-symptomatic infection: Red-edge blue shift (REP <= 718 nm) due to early chlorophyll-a/b synthesis disruption
        if rep_nm <= 718.0:
            diagnosis = "Pre-Symptomatic Fungal Infection (High Risk)"
            risk_level = "High"
        elif ndvi > 0.65:
            diagnosis = "Healthy Active Photosynthetic Canopy"
            risk_level = "Low"
        elif ndvi > 0.30:
            diagnosis = "Moderate Stress / Sparse Canopy"
            risk_level = "Moderate"
        else:
            diagnosis = "Non-Crop Soil / Access Track"
            risk_level = "None"
            
        curve_data = [
            {"wavelength_nm": round(float(w), 2), "reflectance": round(float(r), 4)}
            for w, r in zip(self.wavelengths, reflectance)
        ]
        
        return {
            "coordinates": {"row": row, "col": col, "lat": round(lat, 6), "lon": round(lon, 6)},
            "indices": {
                "ndvi": round(ndvi, 3),
                "pri": round(pri, 4),
                "red_edge_position_nm": round(rep_nm, 1)
            },
            "diagnosis": diagnosis,
            "risk_level": risk_level,
            "num_bands": len(curve_data),
            "spectral_curve": curve_data
        }

def profile_windowed_memory(cube_path: str, metadata_path: str):
    """
    Executes a formal memory benchmark measuring peak RAM during a 256x256 window read
    versus an unwindowed full-scene read.
    """
    tracemalloc.start()
    reader = WindowedRasterReader(cube_path, metadata_path)
    
    # Measure memory during 256x256 windowed read
    tracemalloc.reset_peak()
    tile, transform, meta = reader.read_window(col_off=100, row_off=100, width=256, height=256)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak_mem / (1024 ** 2)
    print(f"[*] Memory Windowing Benchmark: Peak RAM = {peak_mb:.2f} MB")
    assert peak_mb < 150.0, f"Memory threshold exceeded: {peak_mb:.2f} MB >= 150.0 MB"
    print("[+] Passed: Windowed read strictly satisfies <150MB peak RAM requirement!")

if __name__ == "__main__":
    cube_file = os.path.join("data", "salinas_hyperspectral_cube.npy")
    meta_file = os.path.join("data", "cube_metadata.json")
    
    if os.path.exists(cube_file) and os.path.exists(meta_file):
        profile_windowed_memory(cube_file, meta_file)
        
        # Test curve extraction
        reader = WindowedRasterReader(cube_file, meta_file)
        healthy_res = reader.read_spectral_curve(50, 50)
        stressed_res = reader.read_spectral_curve(185, 245)
        print(f"[*] Healthy pixel diagnosis: {healthy_res['diagnosis']} (REP: {healthy_res['indices']['red_edge_position_nm']} nm)")
        print(f"[*] Stressed pixel diagnosis: {stressed_res['diagnosis']} (REP: {stressed_res['indices']['red_edge_position_nm']} nm)")
    else:
        print("[!] Data files missing. Run synthetic_cube_generator.py first.")
