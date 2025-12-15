import numpy as np
import data_loader
import fire_model
import visualizer
import sys

def main():
    print("=== Victoria Fire Spread Model ===")
    
    print("Starting Fire Spread Simulation...")
    
    # 1. Load Data
    # Victoria State Bounds: (140.9, -39.2, 150.5, -34.0) -- Extended East to 150.5
    bounds = (140.9, -39.2, 150.5, -34.0) 
    downsample = 10
    
    elevation_map, fuel_map = data_loader.get_data(bounds, downsample=downsample)
    rows, cols = elevation_map.shape
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Calculate Cell Size for Physics & Area
    # d_lat (deg) -> d_lat_m = d_lat * 111000
    # cell_size = (d_lat_m + d_lon_m) / 2
    avg_lat = (min_lat + max_lat) / 2
    d_lat_deg = (max_lat - min_lat) / rows
    d_lat_m = abs(d_lat_deg) * 111320
    
    d_lon_deg = (max_lon - min_lon) / cols
    d_lon_m = abs(d_lon_deg) * 111320 * np.cos(np.radians(avg_lat))
    
    cell_size = (d_lat_m + d_lon_m) / 2
    print(f"Calculated Cell Size: {cell_size:.2f} meters (Downsampled {downsample}x)")
    
    # 2. Initialize Model
    # Simulation: Strong Wind, Random Direction
    wind_speed = 20.0 # m/s (72 km/h) - Extreme
    wind_dir = np.random.randint(0, 360) # Random Direction
    print(f"Wind Conditions: {wind_speed} m/s from {wind_dir} degrees")
    
    model = fire_model.FireModel(elevation_map, fuel_map, wind_speed=wind_speed, wind_dir=wind_dir, cell_size=cell_size)
    
    # 3. Ignite
    # Try different ignition points until we hit burnable fuel (Grass=1 or Forest=2)
    # Avoid Urban(0) or Water(3)
    rows, cols = elevation_map.shape
    start_point_found = False
    for _ in range(100):
        r = np.random.randint(rows//4, rows//2) # Top/Central area
        c = np.random.randint(cols//3, 2*cols//3)
        if fuel_map[r, c] in [1, 2]: # Grass or Forest
            print(f"Ignition at ({r}, {c}) - Fuel Type: {fuel_map[r,c]}, Elev: {elevation_map[r,c]:.1f}m")
            model.ignite(r, c)
            start_point_found = True
            break
            
    if not start_point_found:
        print("Could not find valid ignition point. Igniting center.")
        model.ignite(rows//2, cols//2)
        
    # 4. Run Simulation
    print("Running simulation...")
    # Simulate 14 Days (168 hours)
    # User requested "specific hour mark".
    sim_duration = 14 * 24 * 60 
    print(f"Simulating {sim_duration/60} Hours...")
    ignition_times = model.run_simulation(max_time=sim_duration)
    
    # 5. Visualize
    print("Generating animation...")
    
    # Calculate Landmark Pixels
    # data_loader.get_data creates a merged dataset. We need the transform.
    # Uh oh, get_data returns arrays but not the transform. 
    # We should have returned the transform from get_data or re-calculate it roughly.
    # Re-calculating from bounds and shape is possible.
    # bounds = (min_lon, min_lat, max_lon, max_lat)
    # shape = (rows, cols)
    # d_lon = (max_lon - min_lon) / cols
    # d_lat = (max_lat - min_lat) / rows
    # pixel_x = (lon - min_lon) / d_lon
    # pixel_y = (max_lat - lat) / d_lat (Note: Y is inverted, 0 at Top/MaxLat)
    
    min_lon, min_lat, max_lon, max_lat = bounds
    rows, cols = elevation_map.shape
    d_lon = (max_lon - min_lon) / cols
    d_lat = (max_lat - min_lat) / rows
    
    landmarks = []
    
    # Process Towns - ONLY MELBOURNE
    for name, (lon, lat) in data_loader.TOWNS.items():
        if name == "Melbourne":
            px = int((lon - min_lon) / d_lon)
            py = int((max_lat - lat) / d_lat)
            landmarks.append((px, py, name))
            
    # Process Lakes - None requested? "Remove all... apart from Melbourne"
    # So skipping lakes.
    # for lake in data_loader.WATER_BODIES:
    #     px = int((lake["lon"] - min_lon) / d_lon)
    #     py = int((max_lat - lake["lat"]) / d_lat)
    #     landmarks.append((px, py, lake["name"]))
    
    visualizer.create_animation(
        elevation_map, 
        fuel_map, 
        ignition_times, 
        output_filename="simulation_fuel.gif",
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        landmarks=landmarks,
        cell_size=cell_size,
        time_step_hours=6,
        max_duration_hours=sim_duration/60
    )
    print("Done! Check 'simulation_fuel.gif'.")

if __name__ == "__main__":
    main()
