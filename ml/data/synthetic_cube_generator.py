"""
TerraSpectra - Synthetic Hyperspectral Cube Generator
=====================================================
Domain: Precision Agriculture & Hyperspectral Computer Vision
Sensor Specs: Replicates NASA Hyperion (220 contiguous bands from 400 nm to 2500 nm)
Spatial Context: 1,000-acre agricultural farm in Salinas Valley, California (EPSG:4326)

This module implements biophysically grounded spectral reflectance modeling:
  1. Healthy vegetation: Low blue/red reflectance (chlorophyll absorption), green peak (550 nm),
     steep red-edge transition (690-740 nm), high NIR plateau (750-1100 nm), and atmospheric water dips.
  2. Pre-symptomatic fungal infection: Early cellular disruption causing a 'blue shift' in the
     red-edge inflection point (702 nm vs 725 nm), a 15% drop in NIR plateau, and altered PRI (531/570 nm),
     while RGB reflectance remains virtually identical to healthy crops (visually indistinguishable).
  3. Non-crop background: Bare soil, farm access roads, and irrigation channels.
"""

import os
import json
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Physical Constants & Hyperspectral Wavelength Setup
# ---------------------------------------------------------------------------
NUM_BANDS = 220
WAVELENGTHS = np.linspace(400.0, 2500.0, NUM_BANDS)  # ~9.54 nm spectral resolution

# Farm Area of Interest (AOI): Salinas Valley, CA
# 1,000-acre farm (~4.047 km^2) -> approx 2.01 km x 2.01 km bounding box
LAT_CENTER = 36.6777
LON_CENTER = -121.6555
DELTA_LAT = 0.0182  # ~2.02 km in latitude
DELTA_LON = 0.0225  # ~2.01 km in longitude (at 36.68 deg N)

GRID_HEIGHT = 512
GRID_WIDTH = 512

# Pixel spatial resolution: ~3.9 meters/pixel
PIXEL_SIZE_Y = DELTA_LAT / GRID_HEIGHT
PIXEL_SIZE_X = DELTA_LON / GRID_WIDTH

TOP_LEFT_LON = LON_CENTER - (DELTA_LON / 2.0)
TOP_LEFT_LAT = LAT_CENTER + (DELTA_LAT / 2.0)

# Affine Geotransform Matrix (GDAL standard):
# [top_left_lon, pixel_width, 0, top_left_lat, 0, -pixel_height]
AFFINE_TRANSFORM = [
    TOP_LEFT_LON,
    PIXEL_SIZE_X,
    0.0,
    TOP_LEFT_LAT,
    0.0,
    -PIXEL_SIZE_Y
]

def generate_vegetation_spectrum(stress_level: float = 0.0) -> np.ndarray:
    """
    Computes a physically grounded surface reflectance spectrum (0.0 to 1.0)
    across all 220 bands (400 nm to 2500 nm).
    
    stress_level: float between 0.0 (completely healthy) and 1.0 (advanced pre-symptomatic stress)
    """
    # 1. Base Chlorophyll Absorption (Visible Spectrum: 400 - 700 nm)
    # Strong absorption at 430-450 nm (blue) and 660-680 nm (red)
    blue_absorption = 0.35 * np.exp(-((WAVELENGTHS - 440) ** 2) / (2 * (30 ** 2)))
    red_absorption = 0.40 * np.exp(-((WAVELENGTHS - 670) ** 2) / (2 * (25 ** 2)))
    green_peak = 0.12 * np.exp(-((WAVELENGTHS - 550) ** 2) / (2 * (40 ** 2)))
    
    vis_reflectance = 0.05 + green_peak - (blue_absorption + red_absorption) * 0.04
    vis_reflectance = np.clip(vis_reflectance, 0.02, 0.20)
    
    # 2. Red-Edge Sigmoidal Transition (690 - 750 nm)
    # Healthy inflection: 725 nm. Pre-symptomatic stress blue-shifts inflection to 702 nm!
    red_edge_center = 725.0 - (23.0 * stress_level)
    red_edge_slope = 0.08 - (0.02 * stress_level)
    red_edge_rise = 1.0 / (1.0 + np.exp(-red_edge_slope * (WAVELENGTHS - red_edge_center)))
    
    # 3. Near-Infrared (NIR) Plateau (750 - 1100 nm)
    # Leaf mesophyll scattering produces high reflectance. Stress reduces this by up to 20%.
    nir_plateau_base = 0.52 - (0.12 * stress_level)
    
    # 4. Shortwave Infrared (SWIR) Water Absorption Bands
    # Deep water absorption valleys at 1400 nm and 1900 nm, plus minor dips at 970 nm & 1200 nm
    water_dip_970 = 0.05 * np.exp(-((WAVELENGTHS - 970) ** 2) / (2 * (40 ** 2)))
    water_dip_1200 = 0.08 * np.exp(-((WAVELENGTHS - 1200) ** 2) / (2 * (50 ** 2)))
    water_dip_1400 = 0.38 * np.exp(-((WAVELENGTHS - 1420) ** 2) / (2 * (60 ** 2)))
    water_dip_1900 = 0.42 * np.exp(-((WAVELENGTHS - 1940) ** 2) / (2 * (70 ** 2)))
    
    # Combine components
    reflectance = vis_reflectance + (nir_plateau_base - vis_reflectance) * red_edge_rise
    reflectance = reflectance - (water_dip_970 + water_dip_1200 + water_dip_1400 + water_dip_1900)
    
    # Pre-symptomatic Photochemical Reflectance Index (PRI) adjustment (531 nm vs 570 nm)
    # Carotenoid / xanthophyll cycle alters the 531 nm reflectance slightly
    pri_band_mask = np.exp(-((WAVELENGTHS - 531) ** 2) / (2 * (10 ** 2)))
    reflectance = reflectance - (0.015 * stress_level * pri_band_mask)
    
    return np.clip(reflectance, 0.01, 0.85)

def generate_soil_spectrum() -> np.ndarray:
    """Bare soil reflectance: monotonic gradual increase from visible to SWIR."""
    base = 0.10 + 0.35 * ((WAVELENGTHS - 400.0) / 2100.0)
    # Minor atmospheric water dips
    water_dip_1400 = 0.06 * np.exp(-((WAVELENGTHS - 1420) ** 2) / (2 * (60 ** 2)))
    water_dip_1900 = 0.07 * np.exp(-((WAVELENGTHS - 1940) ** 2) / (2 * (70 ** 2)))
    return np.clip(base - water_dip_1400 - water_dip_1900, 0.05, 0.55)

def generate_synthetic_hyperspectral_scene(output_dir: str = "data"):
    """
    Synthesizes the 1,000-acre agricultural hyperspectral cube (512 x 512 x 220).
    Injects an early-stage 5-acre fungal infection zone invisible in standard RGB.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    print(f"[*] Generating synthetic hyperspectral cube ({GRID_HEIGHT}x{GRID_WIDTH}x{NUM_BANDS})...")
    
    # Allocate full 3D array: Shape (channels, height, width) for GDAL/Rasterio convention
    cube = np.zeros((NUM_BANDS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    labels = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)  # 0: soil/road, 1: healthy crop, 2: pre-symptomatic stress
    
    healthy_spec = generate_vegetation_spectrum(stress_level=0.0)
    stressed_spec = generate_vegetation_spectrum(stress_level=0.85)
    soil_spec = generate_soil_spectrum()
    
    # 1. Fill base with healthy crop canopy
    # Physically correlated spatial canopy texture modulates the spectrum across all bands
    canopy_texture = 1.0 + np.random.normal(0.0, 0.04, (GRID_HEIGHT, GRID_WIDTH)).astype(np.float32)
    canopy_texture = np.clip(canopy_texture, 0.85, 1.15)
    
    for b in range(NUM_BANDS):
        sensor_noise = np.random.normal(0.0, 0.001, (GRID_HEIGHT, GRID_WIDTH)).astype(np.float32)
        cube[b, :, :] = (healthy_spec[b] * canopy_texture) + sensor_noise
    labels[:, :] = 1  # All healthy crop initially
    
    # 2. Add farm access roads and field boundary buffers (grid lines)
    road_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    # Outer buffer
    road_mask[:12, :] = True
    road_mask[-12:, :] = True
    road_mask[:, :12] = True
    road_mask[:, -12:] = True
    # Internal service tracks dividing the 1,000 acres into fields
    road_mask[120:126, :] = True
    road_mask[250:256, :] = True
    road_mask[380:386, :] = True
    road_mask[:, 165:171] = True
    road_mask[:, 340:346] = True
    
    soil_texture = 1.0 + np.random.normal(0.0, 0.03, (GRID_HEIGHT, GRID_WIDTH)).astype(np.float32)
    soil_texture = np.clip(soil_texture, 0.90, 1.10)
    for b in range(NUM_BANDS):
        sensor_noise = np.random.normal(0.0, 0.001, np.sum(road_mask)).astype(np.float32)
        cube[b, road_mask] = (soil_spec[b] * soil_texture[road_mask]) + sensor_noise
    labels[road_mask] = 0  # Soil / roads
    
    # 3. Inject Pre-Symptomatic Fungal Infection Zone (5-acre cluster)
    # 5 acres at ~15.2 m^2/pixel is approximately 1,332 pixels (radius ~21 pixels)
    # Centered at field zone 3: row=185, col=245
    center_r, center_c = 185, 245
    radius = 22.0
    
    y, x = np.ogrid[:GRID_HEIGHT, :GRID_WIDTH]
    dist_from_center = np.sqrt((y - center_r) ** 2 + (x - center_c) ** 2)
    infection_mask = (dist_from_center <= radius) & (~road_mask)
    
    # Radial Gaussian severity (highest stress in center, fading to edges)
    severity_map = np.clip(1.0 - (dist_from_center / radius), 0.0, 1.0) * infection_mask
    
    for b in range(NUM_BANDS):
        band_healthy = healthy_spec[b]
        band_stressed = stressed_spec[b]
        delta = (band_stressed - band_healthy)
        cube[b, :, :] += (delta * severity_map).astype(np.float32)
        
    labels[infection_mask] = 2  # Pre-symptomatic stress
    
    # Clip reflectance to valid physical domain [0.0, 1.0]
    cube = np.clip(cube, 0.001, 0.999)
    
    # 4. Save metadata and serialized artifacts
    cube_path = os.path.join(output_dir, "salinas_hyperspectral_cube.npy")
    labels_path = os.path.join(output_dir, "ground_truth_labels.npy")
    metadata_path = os.path.join(output_dir, "cube_metadata.json")
    
    print(f"[*] Saving hyperspectral cube array to {cube_path} ({cube.nbytes / (1024**2):.1f} MB)...")
    np.save(cube_path, cube)
    np.save(labels_path, labels)
    
    metadata = {
        "dataset_name": "TerraSpectra Salinas Valley Hyperspectral Simulation",
        "sensor": "NASA Hyperion Analog",
        "crs": "EPSG:4326",
        "num_bands": NUM_BANDS,
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "grid_shape": [NUM_BANDS, GRID_HEIGHT, GRID_WIDTH],
        "pixel_resolution_deg": [PIXEL_SIZE_X, PIXEL_SIZE_Y],
        "approx_ground_resolution_meters": 3.93,
        "affine_geotransform": AFFINE_TRANSFORM,
        "center_coordinates": {"lat": LAT_CENTER, "lon": LON_CENTER},
        "bounding_box": {
            "min_lon": TOP_LEFT_LON,
            "max_lon": TOP_LEFT_LON + DELTA_LON,
            "min_lat": TOP_LEFT_LAT - DELTA_LAT,
            "max_lat": TOP_LEFT_LAT
        },
        "classes": {
            "0": "Non-crop / Access Roads",
            "1": "Healthy Crop Canopy",
            "2": "Pre-Symptomatic Fungal Infection (High Risk)"
        },
        "target_infection_zone": {
            "acres": 5.2,
            "center_pixel": [center_r, center_c],
            "center_gps": [
                TOP_LEFT_LAT - (center_r * PIXEL_SIZE_Y),
                TOP_LEFT_LON + (center_c * PIXEL_SIZE_X)
            ]
        }
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Saved metadata to {metadata_path}")
    
    # 5. Generate False-Color Composite Image (NIR = 850 nm, Red = 660 nm, Green = 550 nm)
    # Band indices corresponding closest to target wavelengths:
    nir_idx = np.argmin(np.abs(WAVELENGTHS - 850.0))
    red_idx = np.argmin(np.abs(WAVELENGTHS - 660.0))
    green_idx = np.argmin(np.abs(WAVELENGTHS - 550.0))
    
    fc_composite = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
    for c_idx, b_idx in enumerate([nir_idx, red_idx, green_idx]):
        band = cube[b_idx, :, :]
        p2, p98 = np.percentile(band, (2, 98))
        norm_band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        fc_composite[:, :, c_idx] = (norm_band * 255.0).astype(np.uint8)
        
    preview_img_path = os.path.join(output_dir, "salinas_false_color_preview.png")
    Image.fromarray(fc_composite).save(preview_img_path)
    print(f"[*] Saved false-color composite preview to {preview_img_path}")
    
    # 6. Generate GeoJSON for Farm Boundary and Management Zones
    geojson_path = os.path.join(output_dir, "farm_aoi.geojson")
    generate_farm_geojson(geojson_path, metadata)
    print(f"[*] Saved farm AOI GeoJSON to {geojson_path}")
    
    return cube, metadata

def generate_farm_geojson(filepath: str, meta: dict):
    """Creates a GeoJSON representation of the 1,000-acre farm boundary and 5-acre zones."""
    bbox = meta["bounding_box"]
    infection_gps = meta["target_infection_zone"]["center_gps"]
    
    # Farm perimeter polygon
    farm_coords = [
        [bbox["min_lon"], bbox["max_lat"]],
        [bbox["max_lon"], bbox["max_lat"]],
        [bbox["max_lon"], bbox["min_lat"]],
        [bbox["min_lon"], bbox["min_lat"]],
        [bbox["min_lon"], bbox["max_lat"]]
    ]
    
    # 5-acre infection hotspot buffer (~75 meter radius)
    inf_lat, inf_lon = infection_gps[0], infection_gps[1]
    d_lat = 0.00068
    d_lon = 0.00085
    zone_coords = [
        [inf_lon - d_lon, inf_lat + d_lat],
        [inf_lon + d_lon, inf_lat + d_lat],
        [inf_lon + d_lon, inf_lat - d_lat],
        [inf_lon - d_lon, inf_lat - d_lat],
        [inf_lon - d_lon, inf_lat + d_lat]
    ]
    
    geojson = {
        "type": "FeatureCollection",
        "name": "Salinas Valley Farm 1000-Acre AOI",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "feature_id": "farm_boundary_salinas_01",
                    "name": "Rancho Salinas Agricultural Parcel 04",
                    "total_acreage": 1000.0,
                    "crop_type": "Lettuce / Brassica",
                    "status": "Monitored",
                    "crs": "EPSG:4326"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [farm_coords]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "feature_id": "infection_zone_05a",
                    "name": "Predicted Pre-Symptomatic Fungal Cluster",
                    "acreage_at_risk": 5.2,
                    "severity": "High (Pre-Symptomatic)",
                    "lead_time": "3 Weeks Before Visible Symptoms",
                    "recommended_action": "Hyper-targeted preventative bio-fungicide spray"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [zone_coords]
                }
            }
        ]
    }
    
    with open(filepath, "w") as f:
        json.dump(geojson, f, indent=2)

if __name__ == "__main__":
    generate_synthetic_hyperspectral_scene()
