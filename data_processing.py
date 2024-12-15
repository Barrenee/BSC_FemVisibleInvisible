import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import netCDF4
from mpl_toolkits.basemap import Basemap

# Step 1: Read NO₂ data from NetCDF file
data_dir = "./es-ftp.bsc.es:8021/"
fp = f'{data_dir}CALIOPE/no2/NO2/sconcno2_2023010100.nc'
nc = netCDF4.Dataset(fp)  # Open the NetCDF file

# Extract variables
no2_levels = nc["sconcno2"][:]  # NO₂ levels
latitudes = nc["lat"][:]   # Latitude array
longitudes = nc["lon"][:] # Longitude array

# Step 2: Ensure lat/lon and NO₂ values align properly
# Assume that no2_levels, latitudes, and longitudes are 2D arrays of the same size
# Flatten the arrays for easier handling
latitudes_flat = latitudes.flatten()
longitudes_flat = longitudes.flatten()
no2_levels_flat = no2_levels[0].flatten()  # First time slice

# Step 3: Create the grid using the same method
cell_size = 0.01  # Approx. 1km resolution in degrees
polygons = []
centers = []
no2_data = []

for lon, lat, no2 in zip(longitudes_flat, latitudes_flat, no2_levels_flat):
    # Define the grid cell centered on the (lon, lat)
    poly = Polygon([
        (lon - cell_size / 2, lat - cell_size / 2),  # Bottom-left
        (lon + cell_size / 2, lat - cell_size / 2),  # Bottom-right
        (lon + cell_size / 2, lat + cell_size / 2),  # Top-right
        (lon - cell_size / 2, lat + cell_size / 2),  # Top-left
        (lon - cell_size / 2, lat - cell_size / 2)   # Closing point
    ])
    polygons.append(poly)
    centers.append((lon, lat))  # Store the center for clarity
    no2_data.append(no2)  # Assign NO₂ value

# Step 4: Create a GeoDataFrame
grid = gpd.GeoDataFrame({
    'geometry': polygons,
    'center_lon': [c[0] for c in centers],
    'center_lat': [c[1] for c in centers],
    'no2_level': no2_data  # Add NO₂ data
})

# Step 5: Plot the grid
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
grid.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot grid boundaries
grid.plot(column='no2_level', ax=ax, cmap='coolwarm', legend=True, legend_kwds={'label': "NO₂ Levels (µg/m³)"})
ax.set_title("1km x 1km Grid of Catalonia with NO₂ Levels")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Step 6: Save dataset to GeoJSON
grid.to_file("catalonia_no2_grid.geojson", driver="GeoJSON")
