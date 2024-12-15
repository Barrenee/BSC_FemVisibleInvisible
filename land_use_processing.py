import numpy as np
import rasterio
from netCDF4 import Dataset
from pyproj import Proj, Transformer
from PIL import Image
import gc
# File paths
data_dir = "./es-ftp.bsc.es:8021/"
land_use_fp = f"{data_dir}/LandUse/MUCSC_2022_10_m_v_3.tif"
Image.MAX_IMAGE_PIXELS = None   # disables the warning


# Open the GeoTIFF file

stride = 5000
land_use_image = Image.open(land_use_fp)
image_data = np.array(land_use_image)
total_row, total_col = image_data.shape

for i in range(0, total_row, stride):
    for j in range(0, total_col, stride):
        with rasterio.open(land_use_fp) as src:
            # Read the image data into a NumPy array (first band)
            land_use_image = Image.open(land_use_fp)  
            image_data = np.array(land_use_image)
            # Get the affine transformation and CRS from the metadata
            transform = src.transform
            crs = src.crs

            # Get the number of rows and columns of the raster
            image_data=image_data[i:i+stride, j:j+stride]
            rows, cols = image_data.shape
            print(rows, cols, "|", i, i+stride, "|" ,j, j+stride)
            # Create arrays for latitude and longitude
            lons = []
            lats = []

            # Create the transformer (WGS84 is EPSG:4326)
            transformer = Transformer.from_crs("ETRS89", "epsg:4326", always_xy=True)

            # Iterate over the raster's pixel coordinates and convert to latitude/longitude
            
            rows, cols = image_data.shape
            for row in range(rows):
                for col in range(cols):
                    # Get the geographic coordinates (longitude, latitude)
                    lon, lat = transform * (col, row)  # Convert pixel to map coordinates
                    lons.append(lon)
                    lats.append(lat)

            # Convert the lists to NumPy arrays
            lons = np.array(lons).reshape((rows, cols))
            lats = np.array(lats).reshape((rows, cols))

            # Now we need to create a NetCDF file to save this data
            # Create the NetCDF file
            nc_file = f"./data/LandUse/land_use_data{i}_{j}.nc"
            with Dataset(nc_file, "w", format="NETCDF4") as nc:
                # Create dimensions
                nc.createDimension("lat", rows)
                nc.createDimension("lon", cols)

                # Create variables for latitude, longitude, and land use values
                latitudes = nc.createVariable("latitude", np.float32, ("lat", "lon"))
                longitudes = nc.createVariable("longitude", np.float32, ("lat", "lon"))
                land_use = nc.createVariable("land_use", np.int32, ("lat", "lon"))

                # Assign attributes
                latitudes.units = "degrees_north"
                longitudes.units = "degrees_east"
                land_use.units = "classification_code"  # Adjust this based on your data

                # Assign the data to the variables
                latitudes[:] = lats
                longitudes[:] = lons
                
                land_use[:] = image_data[:rows, :cols]


            print(f"NetCDF file '{nc_file}' has been created successfully.")
            gc.collect()
            del lons
            del lats
            del latitudes
            del longitudes
            del land_use
            gc.collect()


import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Open the NetCDF file
nc_file = "land_use_data.nc"
with Dataset(nc_file, "r") as nc:
    # Extract the variables from the NetCDF file
    latitudes = nc.variables["latitude"][:]
    longitudes = nc.variables["longitude"][:]
    land_use = nc.variables["land_use"][:]

# Check the shape and data types
print("Latitude shape:", latitudes.shape)
print("Longitude shape:", longitudes.shape)
print("Land use shape:", land_use.shape)


import matplotlib.pyplot as plt
import numpy as np

# Plot the land use data on a map
plt.figure(figsize=(10, 8))
plt.pcolormesh(longitudes, latitudes, land_use, cmap='viridis', shading='auto')
plt.colorbar(label='Land Use Classification')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Land Use Map')

# Add gridlines
plt.grid(True)

# Show the plot
plt.show()

