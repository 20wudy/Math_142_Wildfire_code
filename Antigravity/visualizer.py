import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LightSource, ListedColormap

def create_animation(elevation_map, fuel_map, ignition_times, output_filename="fire_spread.gif", frames=60, wind_speed=0, wind_dir=0, landmarks=None, cell_size=30.0, time_step_hours=None, max_duration_hours=None):
    rows, cols = elevation_map.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Victoria Fire Spread\nWind: {wind_speed*3.6:.0f} km/h (From {wind_dir}°)")
    ax.axis('off')
    
    # 1. Base Layer: Fuel Type
    # Colors: Urban=Grey, Grass=LightGreen, Forest=DarkGreen, Water=Blue
    cmap_fuel = ListedColormap(['#808080', '#90EE90', '#006400', '#4682B4'])
    ax.imshow(fuel_map, cmap=cmap_fuel, extent=(0, cols, rows, 0), alpha=0.6)
    
    # 2. Hillshade Overlay (Texture)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb_hill = ls.shade(elevation_map, cmap=plt.cm.gray, vert_exag=0.5, blend_mode='overlay')
    ax.imshow(rgb_hill, extent=(0, cols, rows, 0), alpha=0.3)
    
    # 3. Add Landmarks (Towns & Lakes)
    if landmarks:
        for x, y, name in landmarks:
            if 0 <= x < cols and 0 <= y < rows:
                ax.text(x, y, name, color='white', fontsize=9, fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    # 4. Wind Arrow
    arrow_len = cols * 0.1
    # Wind from dir -> flows towards dir + 180
    rad = np.radians(wind_dir)
    flow_dx = -np.sin(rad)
    flow_dy = np.cos(rad) 
    arrow_x = cols * 0.95
    arrow_y = rows * 0.1
    ax.arrow(arrow_x, arrow_y, flow_dx * arrow_len, flow_dy * arrow_len, 
             head_width=cols*0.02, head_length=cols*0.02, fc='white', ec='black', width=cols*0.005)
    ax.text(arrow_x, arrow_y - rows*0.05, "Wind", color='white', ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
    
    # Add Legend (Top Left)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', label='Urban'),
        Patch(facecolor='#90EE90', label='Grass'),
        Patch(facecolor='#006400', label='Forest'),
        Patch(facecolor='#4682B4', label='Water'),
        Patch(facecolor='red', label='Active'),
        Patch(facecolor='black', label='Burnt')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize='small')
    
    # --- Scale Bar ---
    # Pixels = 50000 / cell_size.
    bar_len_px = 50000.0 / cell_size
    bar_x = cols * 0.7 # Bottom Right
    bar_y = rows * 0.93
    ax.plot([bar_x, bar_x + bar_len_px], [bar_y, bar_y], color='white', linewidth=4)
    ax.plot([bar_x, bar_x + bar_len_px], [bar_y, bar_y], color='black', linewidth=1)
    ax.text(bar_x + bar_len_px/2, bar_y - rows*0.015, "50 km", color='white', ha='center', fontweight='bold', fontsize=10, 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    
    # 5. Fire Layer
    fire_rgba = np.zeros((rows, cols, 4))
    fire_img = ax.imshow(fire_rgba, extent=(0, cols, rows, 0))

    # Text for Stats (Bottom Left)
    stats_text = ax.text(cols*0.02, rows*0.98, "", color='white', ha='left', va='bottom', fontsize=12, fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

    # Calc Max Time logic
    max_time_data = np.max(ignition_times[ignition_times != np.inf])
    if max_time_data == 0: max_time_data = 1.0
    
    max_time = max_time_data
    if max_duration_hours:
        max_time = min(max_time_data, max_duration_hours * 60.0)

    if time_step_hours:
        # Fixed time steps
        step_min = time_step_hours * 60
        time_steps = np.arange(0, max_time + 0.1, step_min) # +0.1 to include end if exact
        if time_steps[-1] > max_time: time_steps = time_steps[:-1] # Trim overshoot
        frames = len(time_steps)
    else:
        time_steps = np.linspace(0, max_time * 1.0, frames)
    
    def update(frame_idx):
        current_time = time_steps[frame_idx]
        burn_duration = 5 * 60 # Let's say active front is 5 hours.
        
        is_ignited = ignition_times <= current_time
        is_burnt_out = ignition_times <= (current_time - burn_duration)
        is_active = is_ignited & (~is_burnt_out)
        
        # Calculate Stats
        # Cell Area in Hectares
        # cell_size (m) -> cell_size^2 (m^2) / 10000 (m^2/ha)
        cell_area_ha = (cell_size ** 2) / 10000.0
        
        burnt_count = np.sum(is_ignited)
        active_count = np.sum(is_active)
        
        total_ha = burnt_count * cell_area_ha
        active_ha = active_count * cell_area_ha
        
        # Calculate Perimeter
        # Pad mask to detect edges at Image borders
        padded_mask = np.pad(is_ignited.astype(int), 1)
        # Edges along rows (Y)
        dy = np.abs(np.diff(padded_mask, axis=0))
        # Edges along cols (X)
        dx = np.abs(np.diff(padded_mask, axis=1))
        # Total edge segments
        perimeter_segments = np.sum(dy) + np.sum(dx)
        # Perimeter in km
        perimeter_km = perimeter_segments * cell_size / 1000.0
        
        # Format time: Days Hours
        days = int(current_time // (24 * 60))
        hours = int((current_time % (24 * 60)) / 60)
        
        stats_text.set_text(f"Time: {days}d {hours}h\nTotal Burnt: {total_ha:,.0f} ha\nActive Fire: {active_ha:,.0f} ha\nPerimeter: {perimeter_km:,.0f} km")
        
        overlay = np.zeros((rows, cols, 4))
        overlay[is_ignited] = [0.0, 0.0, 0.0, 0.6] # Burnt
        overlay[is_active] = [1.0, 0.0, 0.0, 1.0] # Fire
        
        fire_img.set_data(overlay)
        return [fire_img, stats_text]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

    
    print(f"Saving animation to {output_filename}...")
    try:
        # Try imagemagick or pillow
        ani.save(output_filename, writer='pillow', fps=15)
        print("Success.")
    except Exception as e:
        print(f"Failed to save GIF: {e}. Trying simple plot.")
        # Fallback: Just save final state
        fig.savefig(output_filename.replace(".gif", ".png"))

if __name__ == "__main__":
    # Test
    elev = np.random.rand(100, 100)
    urban = np.zeros((100, 100))
    times = np.random.rand(100, 100) * 100
    create_animation(elev, urban, times)
