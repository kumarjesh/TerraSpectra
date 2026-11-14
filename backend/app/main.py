"""
TerraSpectra - FastAPI High-Throughput Hyperspectral & GIS Service
==================================================================
Module: backend.app.main
Endpoints:
  - GET  /health                      : Hardware acceleration, PyTorch & system telemetry.
  - GET  /api/v1/farm/aoi             : GeoJSON FeatureCollection for Salinas 1,000-acre AOI.
  - GET  /api/v1/tiles/preview        : High-resolution false-color composite PNG image.
  - GET  /api/v1/spectral/curve       : 220-band reflectance profile at GPS coordinate.
  - POST /api/v1/tiles/window         : Windowed spatial patch extraction with optional PCA.
"""

import os
import io
import json
import psutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import torch

from ml.preprocessing.raster_reader import WindowedRasterReader
from ml.preprocessing.pca_engine import SpectralPCAEngine

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CUBE_PATH = os.path.join(BASE_DIR, "data", "salinas_hyperspectral_cube.npy")
META_PATH = os.path.join(BASE_DIR, "data", "cube_metadata.json")
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "farm_aoi.geojson")
PREVIEW_PATH = os.path.join(BASE_DIR, "data", "salinas_false_color_preview.png")
PCA_MODEL_PATH = os.path.join(BASE_DIR, "ml", "preprocessing", "pca_model.pkl")
PCA_META_PATH = os.path.join(BASE_DIR, "ml", "preprocessing", "pca_metadata.json")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reader, pca_engine
    if os.path.exists(CUBE_PATH) and os.path.exists(META_PATH):
        reader = WindowedRasterReader(CUBE_PATH, META_PATH)
        print(f"[+] Initialized WindowedRasterReader ({reader.num_bands} bands, {reader.grid_height}x{reader.grid_width})")
    else:
        print("[!] Warning: Hyperspectral cube files not found. Generate via synthetic_cube_generator.py first.")
        
    if os.path.exists(PCA_MODEL_PATH) and os.path.exists(PCA_META_PATH):
        pca_engine = SpectralPCAEngine.load(PCA_MODEL_PATH, PCA_META_PATH)
        print(f"[+] Loaded Calibrated PCA Engine ({pca_engine.n_components} components, {pca_engine.cumulative_variance_*100:.2f}% var)")
    yield

app = FastAPI(
    title="TerraSpectra Precision Agriculture Hyperspectral API",
    description="High-throughput geospatial API serving memory-safe hyperspectral cubes and pre-symptomatic disease analytics.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def startup_event():
    """Synchronous initialization helper for test suites."""
    global reader, pca_engine
    if os.path.exists(CUBE_PATH) and os.path.exists(META_PATH):
        reader = WindowedRasterReader(CUBE_PATH, META_PATH)
    if os.path.exists(PCA_MODEL_PATH) and os.path.exists(PCA_META_PATH):
        pca_engine = SpectralPCAEngine.load(PCA_MODEL_PATH, PCA_META_PATH)

# Pydantic Schemas
class SpectralPoint(BaseModel):
    wavelength_nm: float
    reflectance: float

class DiagnosticIndices(BaseModel):
    ndvi: float
    pri: float
    red_edge_position_nm: float

class SpectralCurveResponse(BaseModel):
    coordinates: Dict[str, Any]
    indices: DiagnosticIndices
    diagnosis: str
    risk_level: str
    num_bands: int
    spectral_curve: List[SpectralPoint]

class WindowRequest(BaseModel):
    col_offset: int = Field(0, ge=0)
    row_offset: int = Field(0, ge=0)
    width: int = Field(256, ge=16, le=512)
    height: int = Field(256, ge=16, le=512)
    apply_pca: bool = False
    filter_water_bands: bool = False

@app.get("/health", tags=["System"])
def health_check():
    """Returns real-time service health, GPU status, memory telemetry, and dataset readiness."""
    mem = psutil.virtual_memory()
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Host Acceleration)"
    
    return {
        "status": "healthy",
        "service": "TerraSpectra Hyperspectral Engine",
        "version": "1.0.0",
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "device": device_name
        },
        "dataset_ready": reader is not None,
        "pca_engine_ready": pca_engine is not None,
        "system_ram": {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent
        }
    }

@app.get("/api/v1/farm/aoi", tags=["GIS Data"])
def get_farm_aoi():
    """Returns the GeoJSON FeatureCollection defining the 1,000-acre farm boundary and 5-acre zones."""
    if not os.path.exists(GEOJSON_PATH):
        raise HTTPException(status_code=404, detail="Farm AOI GeoJSON not found on server.")
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/v1/tiles/preview", tags=["Imagery"])
def get_tile_preview(mode: str = Query("false_color", pattern="^(false_color|ndvi|rgb)$")):
    """
    Returns an optimized PNG preview of the hyperspectral scene.
    mode: 'false_color' (NIR=850nm, Red=660nm, Green=550nm).
    """
    if not os.path.exists(PREVIEW_PATH):
        raise HTTPException(status_code=404, detail="Preview image not found.")
    return FileResponse(PREVIEW_PATH, media_type="image/png")

@app.get("/api/v1/spectral/curve", response_model=SpectralCurveResponse, tags=["Analytics"])
def get_spectral_curve(
    lat: Optional[float] = Query(None, description="Latitude in EPSG:4326"),
    lon: Optional[float] = Query(None, description="Longitude in EPSG:4326"),
    row: Optional[int] = Query(None, description="Pixel row index"),
    col: Optional[int] = Query(None, description="Pixel column index")
):
    """
    Extracts the full 220-band physical reflectance curve at specific GPS coordinates or pixel indices.
    Returns NDVI, PRI, and Red-Edge Position (REP) with pre-symptomatic disease diagnosis.
    """
    if reader is None:
        raise HTTPException(status_code=503, detail="Hyperspectral reader is not initialized.")
        
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        r, c = reader.gps_to_pixel(lat, lon)
    elif isinstance(row, int) and isinstance(col, int):
        r, c = row, col
    else:
        # Default to the center of the pre-symptomatic infection hotspot
        r, c = 185, 245
        
    result = reader.read_spectral_curve(r, c)
    return result

@app.post("/api/v1/tiles/window", tags=["Tiling Engine"])
def extract_tile_window(req: WindowRequest):
    """
    Extracts a spatial sub-window (e.g. 256x256) on-demand using memory-safe windowed streaming.
    Optionally projects the 220 spectral bands to 16 PCA components.
    """
    if reader is None:
        raise HTTPException(status_code=503, detail="Hyperspectral reader is not initialized.")
        
    patch, transform, meta = reader.read_window(
        col_off=req.col_offset,
        row_off=req.row_offset,
        width=req.width,
        height=req.height,
        filter_water_bands=req.filter_water_bands
    )
    
    if req.apply_pca:
        if pca_engine is None:
            raise HTTPException(status_code=503, detail="PCA engine is not loaded.")
        patch = pca_engine.transform_patch(patch)
        meta["pca_applied"] = True
        meta["output_components"] = pca_engine.n_components
        meta["cumulative_variance_ratio"] = pca_engine.cumulative_variance_
    else:
        meta["pca_applied"] = False
        
    meta["patch_shape"] = list(patch.shape)
    meta["mean_reflectance"] = round(float(np.mean(patch)), 4)
    meta["min_reflectance"] = round(float(np.min(patch)), 4)
    meta["max_reflectance"] = round(float(np.max(patch)), 4)
    
    return {
        "success": True,
        "metadata": meta
    }

@app.get("/api/v1/inference/risk-map", tags=["Inference & Risk"])
def get_risk_heatmap():
    """Returns the transparent RGBA predicted disease-risk heatmap overlay PNG for Deck.gl."""
    heatmap_path = os.path.join(BASE_DIR, "data", "salinas_risk_heatmap.png")
    if not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail="Risk heatmap not found. Run inference pipeline first.")
    return FileResponse(heatmap_path, media_type="image/png")

@app.get("/api/v1/inference/kpis", tags=["Inference & Risk"])
def get_inference_kpis():
    """Returns real-time agronomic KPIs (acreage at risk, mean risk score, priority zone)."""
    kpi_path = os.path.join(BASE_DIR, "data", "inference_kpis.json")
    if not os.path.exists(kpi_path):
        raise HTTPException(status_code=404, detail="KPIs not found. Run inference pipeline first.")
    with open(kpi_path, "r") as f:
        return JSONResponse(content=json.load(f))

@app.get("/api/v1/inference/zones", tags=["Inference & Risk"])
def get_risk_zones():
    """Returns vectorized GeoJSON management zones with risk classifications and coordinates."""
    zones_path = os.path.join(BASE_DIR, "data", "predicted_risk_zones.geojson")
    if not os.path.exists(zones_path):
        raise HTTPException(status_code=404, detail="Risk zones GeoJSON not found.")
    with open(zones_path, "r") as f:
        return JSONResponse(content=json.load(f))

@app.get("/api/v1/inference/prescription", tags=["Inference & Risk"])
def get_spray_prescription():
    """Generates an actionable hyper-targeted spray prescription report."""
    kpi_path = os.path.join(BASE_DIR, "data", "inference_kpis.json")
    if not os.path.exists(kpi_path):
        raise HTTPException(status_code=404, detail="Inference data unavailable.")
    with open(kpi_path, "r") as f:
        kpis = json.load(f)
        
    return {
        "prescription_id": "RX-SALINAS-2026-B4",
        "farm_id": kpis["farm_id"],
        "target_acreage": kpis["acreage_at_risk"],
        "chemical_savings": kpis["target_zone"]["estimated_chemical_savings"],
        "lead_time": kpis["projected_lead_time"],
        "treatment_protocol": {
            "active_ingredient": "Bacillus subtilis strain QST 713",
            "application_rate": "3.5 L / acre",
            "dilution_ratio": "1:50 in non-chlorinated water",
            "spray_window": "Next 72-96 hours (Optimal early dawn application)"
        },
        "target_coordinates": kpis["target_zone"]["center_gps"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

