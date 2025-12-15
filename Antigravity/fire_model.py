import numpy as np
import heapq

class FireModel:
    def __init__(self, elevation_map, fuel_map, wind_speed=0.0, wind_dir=0.0, cell_size=30.0):
        """
        elevation_map: 2D array (meters)
        fuel_map: 2D array (0=Urban, 1=Grass, 2=Forest, 3=Water)
        cell_size: Model resolution in meters (default 30m).
        """
        self.elevation = elevation_map
        self.fuel = fuel_map
        self.rows, self.cols = elevation_map.shape
        self.cell_size = cell_size
        
        self.ignition_time = np.full((self.rows, self.cols), np.inf)
        self.burnt = np.zeros((self.rows, self.cols), dtype=bool)
        
        self.wind_speed = wind_speed # m/s
        self.wind_dir = np.radians(wind_dir)
        
        # --- Research-Based Parameters ---
        # 1. Slope Effect: Noble et al. (1980)
        # Relationship: R = R0 * exp(0.069 * slope_degrees)
        self.k_slope = 0.0693 / (np.pi/180) # Converted for radians

        # 2. Wind Sensitivity & Base Rates
        # Derived from McArthur (1966, 1967) & Cheney et al. (1998) CSIRO
        # - Grassland: RoS is approx 40-50% of 10m open wind speed (Cheney).
        # - Forest: RoS is approx 5-10% of 10m open wind speed (McArthur Mk5).
        
        # Our Model Logic: Rate = Base * (1 + Sensitivity * Wind)
        # At high wind (e.g. 20m/s):
        # - Grass: 1.5 * (1 + 0.25*20) = 9.0 m/s (45% of 20). Matches Cheney.
        # - Forest: 0.5 * (1 + 0.10*20) = 1.5 m/s (7.5% of 20). Matches McArthur.
        
        self.base_rates = {
            0: 0.1,  # Urban (Suppression/Obstacles)
            1: 1.5,  # Grass (CSIRO Grassland Model: High potential spread)
            2: 0.5,  # Forest (McArthur Forest Model: High intensity, slower rate)
            3: 0.0   # Water
        }
        
        self.wind_sensitivity = {
            0: 0.05,
            1: 0.25, # Grass is highly wind driven
            2: 0.10, # Forest is wind driven but drag is higher
            3: 0.0
        } 

    def ignite(self, r, c, start_time=0.0):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.fuel[r,c] != 3: # Can't ignite water
                self.ignition_time[r, c] = start_time

    def run_simulation(self, max_time=1000):
        pq = []
        start_cells = np.argwhere(self.ignition_time < np.inf)
        for r, c in start_cells:
            heapq.heappush(pq, (self.ignition_time[r, c], r, c))
            
        visited = np.zeros_like(self.burnt, dtype=bool)
        
        # Neighbors
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), 
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]
        
        # cell_size is now self.cell_size
        
        while pq:
            curr_time, r, c = heapq.heappop(pq)
            
            if curr_time > max_time:
                break
                
            if visited[r, c]:
                continue
            visited[r, c] = True
            
            current_elev = self.elevation[r, c]
            
            for dr, dc, dist_mult in neighbors:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.rows and 0 <= nc < self.cols and not visited[nr, nc]:
                    fuel_type = self.fuel[nr, nc]
                    
                    if fuel_type == 3: # Water barrier
                        continue
                        
                    base_rate = self.base_rates.get(fuel_type, 0.5)
                    
                    # 1. Slope Effect
                    dist = self.cell_size * dist_mult
                    elev_diff = self.elevation[nr, nc] - current_elev
                    slope_angle = np.arctan2(elev_diff, dist) 
                    slope_factor = np.exp(self.k_slope * slope_angle) # Note: angle in radians, k_slope handles conversion
                    slope_factor = np.clip(slope_factor, 0.1, 8.0)
                    
                    # 2. Wind Effect
                    sensitivity = self.wind_sensitivity.get(fuel_type, 0.1)
                    wind_factor = 1.0
                    
                    if self.wind_speed > 0:
                        # Vector alignment
                        # Wind: (sin(dir), cos(dir)) ? No, check coord system.
                        # Standard math: 0 deg = East (x+). 90 deg = North (y+).
                        # Let's stick to simple: wind_dir is input angle.
                        # Flow vector: (cos(wind_dir), sin(wind_dir))
                        # Spread vector: (dc, -dr) ??
                        # Let's standardize:
                        # Row increases DOWN (y- in math, or just y). Col increases RIGHT (x+).
                        # Spread vector d_vec = (dc, dr).
                        # If wind is "From North" (0 deg), it blows SOUTH. Vector (0, 1).
                        # If wind is "From West" (270 deg), it blows EAST. Vector (1, 0).
                        
                        # Let's interpret the input `wind_dir` as "Blowing FROM angle (Compass)".
                        # 0=N, 90=E, 180=S, 270=W.
                        # Flow vector X component: -sin(wind_dir)
                        # Flow vector Y component: -cos(wind_dir)
                        # e.g. North Wind (0): vel_x = 0, vel_y = -1 (Up? No, South is Y+).
                        # Wait. Rows increase downwards.
                        # North Wind blows DOWN. Vector (0, +1).
                        # West Wind blows RIGHT. Vector (+1, 0).
                        
                        # Compass to Vector (r, c):
                        # North (0): (1, 0) [dr=1, dc=0]
                        # East (90): (0, -1) [dr=0, dc=-1] ?? No, East wind blows West.
                        
                        wx = -np.sin(self.wind_dir) # East-West component
                        wy = -np.cos(self.wind_dir) # North-South component (Cartesian Y)
                        
                        # In grid: y is inverted (row).
                        # If wind blows NORTH (to Y+ in cartesian, Up), it is (-1, 0) in grid.
                        # Let's just use a simpler alignment match.
                        
                        # Spread vector relative to grid
                        dx = dc
                        dy = -dr # Invert row to match Cartesian Y up
                        
                        # Wind vector in Cartesian (blowing TO)
                        # From North (0) -> Blows South. Cartesian Y-.
                        # From West (270) -> Blows East. Cartesian X+.
                        # angle_to = angle_from + 180.
                        # rad_to = self.wind_dir + np.pi
                        # w_vec_x = sin(rad_to) ? No standard compass (0=N, CW).
                        # Math: 0=E, CCW.
                        # Let's just assume User gives "From" compass degrees.
                        # Flow Direction = (From + 180).
                        flow_dir = self.wind_dir + np.pi
                        
                        # Compass 0=N (Y+), 90=E (X+).
                        # Grid: R+ is South (Y-).
                        
                        # Let's map spread (dr, dc) to Compass Heading.
                        # North spread: dr=-1, dc=0.
                        # East spread: dr=0, dc=1.
                        # South: dr=1, dc=0.
                        
                        # Dot product approach is safest.
                        # Wind Vector (dr_wind, dc_wind):
                        # Dot product approach is safest.
                        # Wind Vector (dr_wind, dc_wind):
                        # North Wind (Blows South): dr=1 (Down), dc=0.
                        # East Wind (Blows West): dr=0, dc=-1 (Left).
                        
                        # cos(0)=1, sin(0)=0.
                        # wd_r = cos(dir). 
                        # wd_c = -sin(dir).
                        
                        wd_r = np.cos(self.wind_dir) 
                        wd_c = -np.sin(self.wind_dir)
                        
                        # Dot product of unit vectors
                        dot = (wd_r * dr + wd_c * dc) / dist_mult
                        
                        # Factor
                        wind_factor = 1.0 + sensitivity * self.wind_speed * dot
                        wind_factor = max(0.1, wind_factor)

                    final_rate = base_rate * slope_factor * wind_factor
                    travel_time = dist / final_rate
                    new_time = curr_time + travel_time
                    
                    if new_time < self.ignition_time[nr, nc]:
                        self.ignition_time[nr, nc] = new_time
                        heapq.heappush(pq, (new_time, nr, nc))
            
            # --- Ember Spotting ---
            # Random chance to throw embers downwind
            # Only Forest (2) throws significant embers. Grass (1) is short range.
            fuel_here = self.fuel[r, c]
            if fuel_here == 2 and self.wind_speed > 5.0: # Only strong wind + forest
                # Probability
                if np.random.random() < 0.02: # 2% chance per cell per step
                    # Distance: Proportional to wind.
                    spot_dist_px = np.random.randint(2, int(self.wind_speed * 0.5)) + 1
                    
                    # Direction: Wind Dir +/- 20 degrees
                    # Wind Dir 0=North (Blows South)
                    spot_angle = self.wind_dir + np.random.normal(0, 0.3)
                    
                    # Target Calculation (Downwind)
                    # Blows South: +Row.
                    # sp_r = r + cos(angle) * dist
                    # sp_c = c - sin(angle) * dist
                    
                    sp_r = int(r + np.cos(spot_angle) * spot_dist_px)
                    sp_c = int(c - np.sin(spot_angle) * spot_dist_px)
                    
                    # Determine landing time
                    flight_time = 10.0 + np.random.random() * 20.0 # Minutes
                    spot_time = curr_time + flight_time
                    
                    if 0 <= sp_r < self.rows and 0 <= sp_c < self.cols:
                         if self.fuel[sp_r, sp_c] in [1, 2]: # Land on fuel
                             if spot_time < self.ignition_time[sp_r, sp_c]:
                                 # Spot fire!
                                 self.ignition_time[sp_r, sp_c] = spot_time
                                 heapq.heappush(pq, (spot_time, sp_r, sp_c))
        return self.ignition_time

if __name__ == "__main__":
    # Smoke test
    dem = np.zeros((10, 10))
    urban = np.zeros((10, 10), dtype=int)
    model = FireModel(dem, urban, wind_speed=5.0, wind_dir=0)
    model.ignite(5, 5)
    res = model.run_simulation(100)
    print("Simulation finished.")
