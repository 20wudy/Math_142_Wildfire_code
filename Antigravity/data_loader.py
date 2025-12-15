import numpy as np
import os
import requests
import gzip
import io
import math
import rasterio
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import transform_bounds

# --- Constants ---
# Major Victorian Cities & Towns
TOWNS = {
    "Melbourne": (144.963, -37.814),
    "Geelong": (144.361, -38.149),
    "Ballarat": (143.850, -37.562),
    "Bendigo": (144.279, -36.757),
    "Shepparton": (145.316, -36.384),
    "Mildura": (142.159, -34.208),
    "Warrnambool": (142.484, -38.382),
    "Traralgon": (146.533, -38.196),
    "Bairnsdale": (147.629, -37.825),
    "Horsham": (142.202, -36.711),
    "Wodonga": (146.883, -36.124),
    "Echuca": (144.755, -36.140),
    "Healesville": (145.518, -37.656), # Keep original
    "Mansfield": (146.083, -37.050)
}

WATER_BODIES = [
    # Major Bays & Lakes
    {"name": "Port Phillip", "lat": -38.0, "lon": 144.8, "rad": 0.25}, 
    {"name": "Western Port", "lat": -38.35, "lon": 145.35, "rad": 0.15},
    {"name": "L. Eildon", "lat": -37.15, "lon": 145.92, "rad": 0.1},
    {"name": "L. Hume", "lat": -36.10, "lon": 147.03, "rad": 0.08}
]

def get_srtm_tile_name(lat, lon):
    """
    Returns the SRTM tile name for a given lat/lon.
    SRTM naming: NxxEyyy.hgt (Lower Left Corner)
    e.g. 34.5 N, 119.2 W -> N34W120 ?? No.
    Grid is 1x1 degree.
    Lat 34.5 -> Floor 34 -> N34.
    Lon -119.2 -> Floor -120 -> W120.
    
    Yarra Ranges: -37.7, 145.5.
    Lat -37.7 -> Floor -38 -> S38.
    Lon 145.5 -> Floor 145 -> E145.
    """
    ns = 'S' if lat < 0 else 'N'
    ew = 'W' if lon < 0 else 'E'
    
    # SRTM names are based on lower-left corner
    # If lat is -37.7, the tile covering [-38, -37] starts at -38.
    lat_floor = int(math.floor(lat))
    lon_floor = int(math.floor(lon))
    
    lat_val = abs(lat_floor)
    lon_val = abs(lon_floor)
    
    return f"{ns}{lat_val:02d}{ew}{lon_val:03d}"

def download_tile(tile_name, cache_dir="cache"):
    """
    Downloads and unzips SRTM tile from AWS S3.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    local_path = os.path.join(cache_dir, f"{tile_name}.hgt")
    
    if os.path.exists(local_path):
        return local_path
        
    # URL structure: https://s3.amazonaws.com/elevation-tiles-prod/skadi/S38/S38E145.hgt.gz
    # Group by Latitude band (e.g. S38)
    lat_band = tile_name[:3] 
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_band}/{tile_name}.hgt.gz"
    
    print(f"Downloading {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download {tile_name}: {response.status_code}")
        return None
        
    print("Decompressing...")
    # Decompress into file
    with gzip.open(io.BytesIO(response.content), 'rb') as f_in:
        with open(local_path, 'wb') as f_out:
            f_out.write(f_in.read())
            
    return local_path

def get_data(bounds, downsample=1):
    """
    Downloads and merges SRTM tiles for the given bounds.
    Returns (elevation_map, fuel_map).
    """
    print(f"Loading data for bounds: {bounds}")
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Identify required tiles
    # Simple loop covering the integer degrees
    tiles = []
    
    start_lat = int(math.floor(min_lat))
    end_lat = int(math.floor(max_lat))
    start_lon = int(math.floor(min_lon))
    end_lon = int(math.floor(max_lon))
    
    src_files_to_close = []
    to_merge = []
    
    try:
        for lat in range(start_lat, end_lat + 1):
            for lon in range(start_lon, end_lon + 1):
                # Use slightly offset point to ensure we get the tile containing meaningful data
                # Actually passing integer lat/lon to get_tile_name is fine if logic holds.
                # Center of the degree square: lat+0.5, lon+0.5
                tname = get_srtm_tile_name(lat + 0.5, lon + 0.5)
                fpath = download_tile(tname)
                if fpath:
                    import rasterio
                    src = rasterio.open(fpath)
                    to_merge.append(src)
                    src_files_to_close.append(src)
        
        if not to_merge:
            raise Exception("No tiles downloaded.")
            
        print("Merging tiles...")
        merged, out_trans = merge(to_merge, bounds=bounds)
        
        # Downsample if requested
        if downsample > 1:
            print(f"Downsampling by factor of {downsample}...")
            # Slice the array: [band, row, col] -> merged is (1, rows, cols)
            merged = merged[:, ::downsample, ::downsample]
            
            # Update transform
            # Affine(a, b, c, d, e, f)
            # a = pixel width, e = pixel height (neg)
            # c, f = origin
            # If we skip pixels, width/height multiply by downsample
            out_trans = out_trans * out_trans.scale(downsample, downsample)

        print("Successfully loaded real elevation data & generated fuel map.")
        
        # Generate Fuel Map
        elevation_map = merged[0]
        fuel_map = generate_fuel_map(elevation_map, out_trans)
        
        return elevation_map, fuel_map
        
    except Exception as e:
        print(f"Real data loading failed: {e}. Falling back to synthetic.")
        return get_synthetic_data(800 // downsample, 800 // downsample) # Adjust synthetic size too
    finally:
        for src in src_files_to_close:
            src.close()

def generate_fuel_map(elevation, transform):
    """
    Generates a fuel layer based on heuristics.
    0: Urban (concrete/gardens)
    1: Grassland (valleys/farms)
    2: Forest (hills/ridges)
    3: Water (reservoirs/dams)
    """
    rows, cols = elevation.shape
    fuel_map = np.full((rows, cols), 2, dtype=int) # Default to Forest (2)
    
    # Calculate Slope
    dy, dx = np.gradient(elevation)
    slope = np.sqrt(dy**2 + dx**2)
    
    # 1. Grassland (1): Low elevation (< 150m) OR Low slope (< 0.1 rise/run) & Moderate Elevation (< 400m)
    # Adjust thresholds based on Yarra Valley geography
    # Yarra Glen / Healesville valley floor is approx 50-150m.
    is_valley = (elevation < 200)
    is_flat_plateau = (slope < 2.0) & (elevation < 400) # Valid for farming areas
    fuel_map[is_valley | is_flat_plateau] = 1
    
    # 2. Urban (0): Proximity to towns
    
    cols_grid, rows_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    xs, ys = rasterio.transform.xy(transform, rows_grid, cols_grid)
    xs = np.array(xs).reshape(rows, cols)
    ys = np.array(ys).reshape(rows, cols)
    
    for name, (tx, ty) in TOWNS.items():
        dist_sq = (xs - tx)**2 + (ys - ty)**2
        
        # Melbourne is huge, others are smaller
        if name == "Melbourne":
            rad = 0.25 # ~25km radius
        elif name in ["Geelong", "Ballarat", "Bendigo"]:
            rad = 0.08 # ~8km
        else:
            rad = 0.03 # ~3km
            
        fuel_map[dist_sq < rad**2] = 0
        
    # 3. Water (3): Major Bays & Lakes
    for wb in WATER_BODIES:
         wx, wy, rad = wb["lon"], wb["lat"], wb["rad"]
         dist_sq = (xs - wx)**2 + (ys - wy)**2
         fuel_map[dist_sq < rad**2] = 3
    
    # 4. Mallee Scrub (North West) - Treat as dry Grassland/Scrub (Type 1)
    # North of -36, West of 144 roughly
    is_mallee = (ys > -36.0) & (xs < 144.0)
    # Maybe introduce a new fuel type? For now, Grass is fine (Fast spread).
    fuel_map[is_mallee] = 1
    
    # 5. The Alps (High Country) - Force Forest (Type 2)
    is_alps = (xs > 146.0) & (ys > -37.5) & (elevation > 400)
    fuel_map[is_alps] = 2

    # 6. Ocean/Sea (3) - Elevation <= 0
    # Also catch very low elevations near coast
    fuel_map[elevation <= 0] = 3

    return fuel_map

def get_synthetic_data(width, height):
    print("Generating synthetic terrain...")
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    elevation_map = 300 * (np.sin(X/3) + np.cos(Y/2)) + 400
    elevation_map += np.random.normal(0, 10, (height, width))
    elevation_map = np.maximum(elevation_map, 0)
    
    # Simple synthetic fuel
    fuel_map = np.full((height, width), 2, dtype=int) # Forest
    fuel_map[elevation_map < 300] = 1 # Grass in valleys
    fuel_map[80:120, 80:120] = 0 # Town
    
    return elevation_map, fuel_map


if __name__ == "__main__":
    bounds = (145.30, -37.80, 145.70, -37.50) # Expanded bounds test 
    dem, fuel = get_data(bounds)
    print(f"Shape: {dem.shape}, Range: {dem.min()} - {dem.max()}")
    print(f"Fuel Types present: {np.unique(fuel)}")

