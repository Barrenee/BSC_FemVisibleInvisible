import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load your grid data
grid = gpd.read_file("catalonia_no2_grid.geojson")  # Load the GeoJSON with NO₂ data

# Open shapefile
data_dir = "./es-ftp.bsc.es:8021/"
shapefile = gpd.read_file(f"{data_dir}/LimitsAdministratius/Catalunya/divisions-administratives-v2r1-catalunya-5000-20240705.shp")

# Ensure CRS consistency
if grid.crs != shapefile.crs:
    shapefile = shapefile.to_crs(grid.crs)  # Transform shapefile to match grid CRS

# Perform spatial join to combine datasets
grid = gpd.sjoin(grid, shapefile, how='inner', predicate='intersects')  # Spatial join

# Step 1: Setup the world map with Cartopy
fig, ax = plt.subplots(
    1, 1, figsize=(15, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# Add world features
ax.set_global()
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Step 2: Plot shapefile boundaries
for _, row in shapefile.iterrows():
    ax.add_geometries(
        [row['geometry']],
        crs=ccrs.PlateCarree(),
        edgecolor='black',
        facecolor='none',
        linewidth=1.0,
    )

# Step 3: Overlay Catalonia NO₂ data
grid.plot(
    ax=ax,
    column='no2_level',          # Data column for color
    cmap='coolwarm',             # Color map
    legend=True,                 # Add legend for NO₂ levels
    legend_kwds={'label': "NO₂ Levels (µg/m³)"}
)

# Step 4: Zoom in to Catalonia
ax.set_extent([-1, 5, 39, 44], crs=ccrs.PlateCarree())  # Catalonia's bounding box

# Add title
ax.set_title("Catalonia NO₂ Levels with Administrative Boundaries", fontsize=16)

# Show plot
plt.show()
