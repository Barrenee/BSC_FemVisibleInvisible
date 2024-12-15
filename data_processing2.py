import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import netCDF4
from PIL import Image
from pyproj import Proj, transform
from math import inf

Image.MAX_IMAGE_PIXELS = None   # disables the warning

utm_proj = Proj(proj="utm", zone=31, ellps="WGS84", datum="WGS84", north=True)
wgs84_proj = Proj(proj="latlong", datum="WGS84")


# Step 1: Read NO₂ data from NetCDF file
data_dir = "./es-ftp.bsc.es:8021/"
fp = f'{data_dir}CALIOPE/no2/NO2/sconcno2_2023010100.nc'
nc = netCDF4.Dataset(fp)  # Open the NetCDF file



# Extract variables
no2_levels = nc["sconcno2"][:]  # NO₂ levels
latitudes_no2 = nc["lat"][:]   # Latitude array
min_lat = np.min(latitudes_no2)
max_lat = np.max(latitudes_no2)
longitudes_no2 = nc["lon"][:] # Longitude array
min_lon = np.min(longitudes_no2)
max_lon = np.max(longitudes_no2)

# Read the land use data. It will be a .tif file:



# LAND USE DATA PROCESSING

land_use_fp = f"{data_dir}/LandUse/MUCSC_2022_10_m_v_3.tif"

land_use_image = Image.open(land_use_fp)  
land_use = np.array(land_use_image)

# UTM square parameters
center_easting = 260160.000
center_northing = 4747980.000
width = 10.00
height = 10.00

# Calculate half dimensions
half_width = width / 2
half_height = height / 2

# Calculate starting coordinates (southwest corner)
start_easting = center_easting - half_width
start_northing = center_northing - half_height

end_easting = center_easting + half_width
end_northing = center_northing + half_height

lon_start_land_use, lat_start_land_use = transform(utm_proj, wgs84_proj, start_easting, start_northing)
lon_end_land_use, lat_end_land_use = transform(utm_proj, wgs84_proj, end_easting, end_northing)

# Output the result
print(f"Starting coordinates (southwest corner): UTM ({start_easting:.3f}, {start_northing:.3f})")
print(f"Ending coordinates (northeast corner): UTM ({end_easting:.3f}, {end_northing:.3f})")
print()
print(f"Starting coordinates (southwest corner): WGS84 ({lon_start_land_use:.6f}, {lat_start_land_use:.6f})")
print(f"Ending coordinates (northeast corner): WGS84 ({lon_end_land_use:.6f}, {lat_end_land_use:.6f})")
print()
print(f"Minimum longitude: {min_lon}")
print(f"Minimum latitude: {min_lat}")
print(f"Maximum longitude: {max_lon}")
print(f"Maximum latitude: {max_lat}")





numpy_transform_matrix = np.array([[30, 0, 0], 
                                   [-30, 260160, 4747980], 
                                   [0,0,1]]) 
print(numpy_transform_matrix)

pixels = []
np.zeros

pixels = np.array(pixels)
x, y, _ = np.dot(numpy_transform_matrix, pixels)
print(x,y)

longitude, latitude = transform(utm_proj, wgs84_proj, x, y)

# Get the coordinates for the land use data
land_use_coords = land_use_image.getbbox()
land_use_coords = list(land_use_coords)

prev = inf
prev2 = inf
# Iterate over the longitudes and latitudes of the NO2 data to find the most similar coordinates to the land use starting longitude
for i in range(min(longitudes_no2.shape)):
    dist = abs(longitudes_no2[i,i] - lon_start_land_use)
    if dist < prev:
        prev = dist
    if dist > prev:
        print(i)
        most_similar_lon_start = longitudes_no2[i,i]
        break








print(most_similar_lon_start, lon_start_land_use, most_similar_lon_start-lon_start_land_use)


print(land_use.shape)
print(latitudes_no2.shape)
print(longitudes_no2.shape)
print(no2_levels.shape)