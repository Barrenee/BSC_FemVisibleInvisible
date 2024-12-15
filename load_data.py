import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import folium
from scipy.spatial import cKDTree
from tqdm import tqdm
import rtree
import matplotlib.pyplot as plt
import pyreadr
# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(data_dir, date_str, barcelona_bounds):
    """Loads and preprocesses the required data."""
    # File paths
    sconcno2_file = os.path.join(data_dir, 'NO2', f'sconcno2_{date_str}00.nc') # we use only hour 00
    caliope_urban_file = os.path.join(data_dir, 'CALIOPE-Urban', f'UK_{date_str}.rds')
    roads_file = os.path.join(data_dir, 'road_transport_interurban_CAT.geojson')
    population_file = os.path.join(data_dir, 'Poblacio', 'gridpoblacio01012022.shp')
    
    # Load datasets
    sconcno2_data = xr.open_dataset(sconcno2_file, engine='netcdf4')

    caliope_urban_df = pyreadr.read_r(caliope_urban_file)[None]

    roads = gpd.read_file(roads_file)
    population = gpd.read_file(population_file)

    # --- 1.1  Filter and transform sconcno2 (reference caliope) to the barcelona region ---
    # Filter and transform sconcno2
    sconcno2_data = sconcno2_data.isel(time=0).squeeze()
    sconcno2_lats = sconcno2_data['lat'].values  # 2D array
    sconcno2_lons = sconcno2_data['lon'].values  # 2D array
    sconcno2_values = sconcno2_data['sconcno2'].values # 2D array
    
    # Get coordinates from the sconcno2 grid
    coords_sconcno2 = np.stack([sconcno2_lats.flatten(), sconcno2_lons.flatten()], axis=-1)

    # Create a mask for the bounding box, first transforming lat/lon to the sconcno2 grid projection, and then select it
    lons_sconcno2, lats_sconcno2 = np.meshgrid(sconcno2_data['lon'].values[0,:], sconcno2_data['lat'].values[:,0])
    mask_bounds = (
        (lats_sconcno2 >= barcelona_bounds[0]) & (lats_sconcno2 <= barcelona_bounds[2]) &
        (lons_sconcno2 >= barcelona_bounds[1]) & (lons_sconcno2 <= barcelona_bounds[3])
    )

    # Apply mask to filter data
    sconcno2_values = sconcno2_values[mask_bounds]
    lats_sconcno2_filtered = lats_sconcno2[mask_bounds]
    lons_sconcno2_filtered = lons_sconcno2[mask_bounds]
    coords_sconcno2_filtered = np.stack([lats_sconcno2_filtered.flatten(), lons_sconcno2_filtered.flatten()], axis=-1)

    # --- 1.2 Filter and transform CALIOPE-Urban to barcelona area ---
    # Filter and transform Caliope Urban
    # Transform coordinates
    caliope_urban_df['geometry'] = gpd.points_from_xy(caliope_urban_df['X'], caliope_urban_df['Y'])
    caliope_urban_gdf = gpd.GeoDataFrame(caliope_urban_df, crs="EPSG:25831")
    caliope_urban_gdf = caliope_urban_gdf.to_crs(epsg=4326)
    caliope_urban_gdf['lon'] = caliope_urban_gdf.geometry.x
    caliope_urban_gdf['lat'] = caliope_urban_gdf.geometry.y
    
    # Filter data based on the same mask applied to sconcno2 (it is a subset)
    mask_bounds = (
        (caliope_urban_gdf['lat'] >= barcelona_bounds[0]) & (caliope_urban_gdf['lat'] <= barcelona_bounds[2]) &
        (caliope_urban_gdf['lon'] >= barcelona_bounds[1]) & (caliope_urban_gdf['lon'] <= barcelona_bounds[3])
    )
    caliope_urban_gdf = caliope_urban_gdf[mask_bounds].copy()
    
    # --- 1.3 Filter and Transform Roads to Barcelona ---
    # Reproject roads to the target CRS (4326) and filter by the bounding box
    roads = roads.to_crs(epsg=4326)
    roads_filtered = gpd.clip(roads, Polygon([(barcelona_bounds[1], barcelona_bounds[0]), (barcelona_bounds[3], barcelona_bounds[0]), (barcelona_bounds[3], barcelona_bounds[2]), (barcelona_bounds[1], barcelona_bounds[2])]))
    
    # --- 1.4 Filter and Transform Population to Barcelona
    # Reproject population to the target CRS (4326) and filter by the bounding box
    population = population.to_crs(epsg=4326)
    population_filtered = gpd.clip(population, Polygon([(barcelona_bounds[1], barcelona_bounds[0]), (barcelona_bounds[3], barcelona_bounds[0]), (barcelona_bounds[3], barcelona_bounds[2]), (barcelona_bounds[1], barcelona_bounds[2])]))
    
    return  sconcno2_values, coords_sconcno2_filtered, caliope_urban_gdf, roads_filtered, population_filtered, lats_sconcno2, lons_sconcno2


def create_grid(lats, lons, grid_resolution=25):
    """Creates a grid with the given resolution in meters, covering the span of lat/lons in the original grid."""
    
    # Convert the bounding box to meters using an approximate conversion for Barcelona's latitude
    meters_per_degree_lat = 111132.954 # Approx. meters for lat
    meters_per_degree_lon = 111132.954 * np.cos(np.deg2rad(np.mean(lats))) # Approx. meters for lon

    # Get the bounding box in lat/lon coordinates
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)
    
    # Calculate the grid bounds in meters
    min_x_m = min_lon * meters_per_degree_lon
    max_x_m = max_lon * meters_per_degree_lon
    min_y_m = min_lat * meters_per_degree_lat
    max_y_m = max_lat * meters_per_degree_lat

    # Create x and y coordinates for the grid
    x_coords_m = np.arange(min_x_m, max_x_m, grid_resolution)
    y_coords_m = np.arange(min_y_m, max_y_m, grid_resolution)

    # Create the 2D grid in meters
    x_grid, y_grid = np.meshgrid(x_coords_m, y_coords_m)

    # Convert the x/y grid back to lat/lon
    lon_grid = x_grid / meters_per_degree_lon
    lat_grid = y_grid / meters_per_degree_lat

    return lat_grid, lon_grid

def features_from_grid(roads_filtered, population_filtered, target_lat_grid, target_lon_grid):
    """Creates the grid features (road density, population density, distance to roads and Gaussian Plume)"""
    
    # --- Create the grid to do the spatial join ---
    target_coords_flat = np.stack([target_lat_grid.flatten(), target_lon_grid.flatten()], axis=-1)
    target_grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(target_coords_flat[:, 1], target_coords_flat[:, 0]), crs="EPSG:4326")

    # --- 1. Road Density ---
    # Spatial join between roads and the grid
    roads_joined = gpd.sjoin(target_grid_gdf, roads_filtered, how="left")
    
    # Count number of roads inside each cell
    road_counts = roads_joined.groupby(roads_joined.index).size()

    # Populate the density map
    road_density_map = np.zeros_like(target_lat_grid.flatten(), dtype=float)
    road_density_map[road_counts.index] = road_counts.values
    road_density_map = road_density_map.reshape(target_lat_grid.shape)
    
    # --- 2. Distance to Major Roads ---
    # Create a R-tree index for road geometries
    idx = rtree.index.Index()
    for i, row in roads_filtered.iterrows():
        idx.insert(i, row.geometry.bounds)

    # Function to find nearest road for a given point
    def nearest_road(point):
        nearest_ids = list(idx.nearest(point.coords[0], 1))
        if nearest_ids:
            nearest_road = roads_filtered.iloc[nearest_ids[0]].geometry
            return point.distance(nearest_road)
        else:
            return np.nan

    # Calculate distances to the nearest road
    distances = target_grid_gdf.apply(nearest_road, axis=1)
    distances_map = distances.values.reshape(target_lat_grid.shape)
    
    # --- 3. Population Density ---
    # Spatial join between population and the grid
    population_joined = gpd.sjoin(target_grid_gdf, population_filtered, how="left")

    # Sum population in each cell
    population_counts = population_joined.groupby(population_joined.index)['TOTAL'].sum()
    
    # Populate the population map
    population_map = np.zeros_like(target_lat_grid.flatten(), dtype=float)
    population_map[population_counts.index] = population_counts.values
    population_map = population_map.reshape(target_lat_grid.shape)
    
    return road_density_map, distances_map, population_map


def gaussian_plume(source_lat, source_lon, target_lat_grid, target_lon_grid, dispersion_factor=0.1):
    """Approximates a Gaussian plume dispersion factor relative to a source point."""
    
    # Convert source lat/lon to a 2D array that matches the target grid
    source_lat_grid = np.full_like(target_lat_grid, source_lat)
    source_lon_grid = np.full_like(target_lon_grid, source_lon)

    # Calculate distances
    distances = np.sqrt(((target_lon_grid - source_lon_grid) * 111132.954 * np.cos(np.deg2rad(source_lat)))**2 + ((target_lat_grid - source_lat_grid) * 111132.954)**2)

    # Calculate the Gaussian dispersion factor. The distance needs to be divided by 2 (as it is a linear dispersion) and 
    # use a standard deviation that makes it significant at 1000m (around 250).
    dispersion = np.exp(-0.5 * (distances / 250)**2) * (1/np.sqrt(2 * np.pi * 250**2))
    
    # Normalize to 0-1
    dispersion = (dispersion - np.min(dispersion)) / (np.max(dispersion) - np.min(dispersion))
    return dispersion


def create_dataset(sconcno2_values, coords_sconcno2_filtered, caliope_urban_gdf, roads_filtered, population_filtered,  lats_sconcno2, lons_sconcno2, batch_size, image_size, seed):
    """Creates the custom dataset for Pytorch training"""
    
    # Create grid at the higher resolution (25x25m)
    target_lat_grid, target_lon_grid = create_grid(lats_sconcno2, lons_sconcno2, grid_resolution=25)

    # Create the road density, distance to road and population density maps for this grid
    road_density_map, distances_map, population_map = features_from_grid(roads_filtered, population_filtered, target_lat_grid, target_lon_grid)

    # Create a gaussian plume model for each cell of the sconcno2 grid
    gaussian_maps = []
    for lat_sconcno2, lon_sconcno2 in coords_sconcno2_filtered:
        gaussian_maps.append(gaussian_plume(lat_sconcno2, lon_sconcno2, target_lat_grid, target_lon_grid))
    gaussian_maps = np.array(gaussian_maps)

    # --- Interpolate the Sconcno2 map (1000x1000m) to the 25x25 grid and scale values ---
    # Scale the values between 0-1
    sconcno2_values = (sconcno2_values - np.min(sconcno2_values)) / (np.max(sconcno2_values) - np.min(sconcno2_values))
    sconcno2_values_reshaped = sconcno2_values.reshape(-1, 1, 1) # Prepare data for broadcasting (add channels = 1)
    sconcno2_maps = np.repeat(sconcno2_values_reshaped, target_lat_grid.flatten().shape[0], axis=1)  # Broadcast values
    sconcno2_maps = sconcno2_maps.reshape(len(sconcno2_values), target_lat_grid.shape[0], target_lat_grid.shape[1]) # Reshape to correct grid
        
    # --- Filter the Caliope Urban Data to this Grid ---
    # Get the lat/lon from the high res grid, and select the ones that are contained in a point in Caliope Urban data
    grid_points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(target_lon_grid.flatten(), target_lat_grid.flatten()), crs="EPSG:4326")
    caliope_urban_values_map = np.zeros_like(target_lat_grid.flatten()) # Initialize an array with zeros for the urban map
    caliope_points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(caliope_urban_gdf['lon'], caliope_urban_gdf['lat']), crs="EPSG:4326")

    # Spatial Join
    points_joined = gpd.sjoin(grid_points_gdf, caliope_points_gdf, how="left")

    # Fill the value with the pollutant_uk, if it exists, and 0 if not
    caliope_urban_values_map = np.array([caliope_urban_gdf['pollutant_uk'].values[idx] if not pd.isna(idx) else 0 for idx in points_joined['index_right'].values])

    # Scale the values between 0-1
    caliope_urban_values_map_scaled = (caliope_urban_values_map - np.min(caliope_urban_values_map)) / (np.max(caliope_urban_values_map) - np.min(caliope_urban_values_map))
    caliope_urban_values_map_scaled = caliope_urban_values_map_scaled.reshape(target_lat_grid.shape)

    # Stack all the input features (sconcno2, road density, distance to road, population density and gaussian dispersion)
    input_features = np.stack([sconcno2_maps, road_density_map, distances_map, population_map, gaussian_maps], axis=-1)
    
    # Reshape it to be (number of 1000m grid cells, image size, image size, number of channels)
    output_images = []
    input_images = []
    for input_image, output_image in zip(input_features, caliope_urban_values_map_scaled):
        # split the high res image into smaller patches
        for y in range(0, target_lat_grid.shape[0]-image_size, image_size):
           for x in range(0, target_lon_grid.shape[1]-image_size, image_size):
                output_images.append(output_image[y:y + image_size, x:x + image_size])
                input_images.append(input_image[y:y + image_size, x:x + image_size,:]) #all channels

    output_images = np.array(output_images)
    input_images = np.array(input_images)
    
    return  SuperResolutionDataset(input_images, output_images, seed=seed)

class SuperResolutionDataset(Dataset):
    """Pytorch Dataset class for the super resolution problem."""
    def __init__(self, input_data, output_data, seed=42):
        self.input_data = input_data
        self.output_data = output_data
        self.seed = seed
        # Scale the values between 0-1
        self.scaler_x = MinMaxScaler()
        self.input_data_scaled = self.scaler_x.fit_transform(self.input_data.reshape(-1, self.input_data.shape[-1])).reshape(self.input_data.shape)
        self.scaler_y = MinMaxScaler()
        self.output_data_scaled = self.scaler_y.fit_transform(self.output_data.reshape(-1, 1)).reshape(self.output_data.shape)
       
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return torch.tensor(self.input_data_scaled[idx], dtype=torch.float).permute(2, 0, 1), torch.tensor(self.output_data_scaled[idx], dtype=torch.float).unsqueeze(0)


# --- 2. Model Definition ---
class SuperResolutionModel(nn.Module):
    """Defines the model with CNN layers and Skip-connections"""
    def __init__(self, in_channels=5, base_channels=64):
        super(SuperResolutionModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2)

        self.deconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(base_channels * 2 * 2, base_channels, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(base_channels * 2, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        
        # Decoder
        x = self.relu(self.deconv1(x3))
        # Skip connection
        x = torch.cat([x, x2], dim=1) 
        x = self.relu(self.deconv2(x))

        # Skip connection
        x = torch.cat([x, x1], dim=1)
        x = self.conv_out(x)

        return x


# --- 3. Training Loop ---
def train_model(model, train_loader, test_loader, learning_rate, num_epochs, device, save_path):
    """Defines the training loop for the model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.shape[0]

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}/{num_epochs}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item() * inputs.shape[0]

        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}')
    
    torch.save(model.state_dict(), save_path)

    return model, train_losses, test_losses


# --- 4. Inference ---
def predict_whole_catalonia(model, data_dir, date_str, catalonia_bounds, device, image_size, seed):
    """Predicts the caliope urban values for the whole region of catalonia"""
     # Load datasets
    sconcno2_file = os.path.join(data_dir, 'NO2', f'sconcno2_{date_str}00.nc')
    roads_file = os.path.join(data_dir, 'road_transport_interurban_CAT.geojson')
    population_file = os.path.join(data_dir, 'Poblacio', 'gridpoblacio01012022.shp')
    
    # Load datasets
    sconcno2_data = xr.open_dataset(sconcno2_file, engine='netcdf4')
    roads = gpd.read_file(roads_file)
    population = gpd.read_file(population_file)

    # --- 1.1  Filter and transform sconcno2 (reference caliope) to the whole catalonia region ---
    sconcno2_data = sconcno2_data.isel(time=0).squeeze()
    sconcno2_lats = sconcno2_data['lat'].values  # 2D array
    sconcno2_lons = sconcno2_data['lon'].values  # 2D array
    sconcno2_values = sconcno2_data['sconcno2'].values # 2D array
    
    # Filter data based on the whole catalonia region
    lons_sconcno2, lats_sconcno2 = np.meshgrid(sconcno2_data['lon'].values[0,:], sconcno2_data['lat'].values[:,0])
    mask_bounds = (
        (lats_sconcno2 >= catalonia_bounds[0]) & (lats_sconcno2 <= catalonia_bounds[2]) &
        (lons_sconcno2 >= catalonia_bounds[1]) & (lons_sconcno2 <= catalonia_bounds[3])
    )

    # Apply mask to filter data
    sconcno2_values = sconcno2_values[mask_bounds]
    lats_sconcno2_filtered = lats_sconcno2[mask_bounds]
    lons_sconcno2_filtered = lons_sconcno2[mask_bounds]
    coords_sconcno2_filtered = np.stack([lats_sconcno2_filtered.flatten(), lons_sconcno2_filtered.flatten()], axis=-1)

    # --- 1.2 Filter and Transform Roads to whole catalonia ---
    # Reproject roads to the target CRS (4326) and filter by the bounding box
    roads = roads.to_crs(epsg=4326)
    roads_filtered = gpd.clip(roads, Polygon([(catalonia_bounds[1], catalonia_bounds[0]), (catalonia_bounds[3], catalonia_bounds[0]), (catalonia_bounds[3], catalonia_bounds[2]), (catalonia_bounds[1], catalonia_bounds[2])]))

    # --- 1.3 Filter and Transform Population to whole catalonia ---
    # Reproject population to the target CRS (4326) and filter by the bounding box
    population = population.to_crs(epsg=4326)
    population_filtered = gpd.clip(population, Polygon([(catalonia_bounds[1], catalonia_bounds[0]), (catalonia_bounds[3], catalonia_bounds[0]), (catalonia_bounds[3], catalonia_bounds[2]), (catalonia_bounds[1], catalonia_bounds[2])]))


    # Create grid at the higher resolution (25x25m)
    target_lat_grid, target_lon_grid = create_grid(lats_sconcno2, lons_sconcno2, grid_resolution=25)

    # Create the road density, distance to road and population density maps for this grid
    road_density_map, distances_map, population_map = features_from_grid(roads_filtered, population_filtered, target_lat_grid, target_lon_grid)
    
    # Create a gaussian plume model for each cell of the sconcno2 grid
    gaussian_maps = []
    for lat_sconcno2, lon_sconcno2 in coords_sconcno2_filtered:
        gaussian_maps.append(gaussian_plume(lat_sconcno2, lon_sconcno2, target_lat_grid, target_lon_grid))
    gaussian_maps = np.array(gaussian_maps)

    # --- Interpolate the Sconcno2 map (1000x1000m) to the 25x25 grid and scale values ---
    # Scale the values between 0-1
    sconcno2_values = (sconcno2_values - np.min(sconcno2_values)) / (np.max(sconcno2_values) - np.min(sconcno2_values))
    sconcno2_values_reshaped = sconcno2_values.reshape(-1, 1, 1) # Prepare data for broadcasting (add channels = 1)
    sconcno2_maps = np.repeat(sconcno2_values_reshaped, target_lat_grid.flatten().shape[0], axis=1)  # Broadcast values
    sconcno2_maps = sconcno2_maps.reshape(len(sconcno2_values), target_lat_grid.shape[0], target_lat_grid.shape[1]) # Reshape to correct grid
    
    # Stack all the input features (sconcno2, road density, distance to road, population density and gaussian dispersion)
    input_features = np.stack([sconcno2_maps, road_density_map, distances_map, population_map, gaussian_maps], axis=-1)

    # Make predictions
    predicted_images = []
    with torch.no_grad():
        for input_image in input_features:
           # split the high res image into smaller patches
            for y in range(0, target_lat_grid.shape[0]-image_size, image_size):
                for x in range(0, target_lon_grid.shape[1]-image_size, image_size):
                    input_patch = input_image[y:y + image_size, x:x + image_size, :]
                    input_patch_scaled = MinMaxScaler().fit_transform(input_patch.reshape(-1, input_patch.shape[-1])).reshape(input_patch.shape) # Scale the inputs
                    input_tensor = torch.tensor(input_patch_scaled, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)
                    prediction = model(input_tensor).squeeze().cpu().numpy()
                    predicted_images.append(prediction)

    # Put the predicted images together in a single array and return
    predicted_images = np.array(predicted_images)
    final_prediction = np.zeros_like(target_lat_grid, dtype=float)
    idx=0
    for y in range(0, target_lat_grid.shape[0]-image_size, image_size):
        for x in range(0, target_lon_grid.shape[1]-image_size, image_size):
            final_prediction[y:y + image_size, x:x + image_size] = predicted_images[idx]
            idx+=1
    
    return final_prediction, target_lat_grid, target_lon_grid

# --- Main execution ---
if __name__ == '__main__':
    # --- 0. Configuration ---
    data_dir = ''  # Path to the folder with the data
    date_str = '20230101'  # Date of the data to use
    barcelona_bounds = (41.26, 2.00, 41.58, 2.38) # Latitude/longitude bounds of Barcelona (min_lat, min_lon, max_lat, max_lon)
    catalonia_bounds = (40.4, 0.05, 42.9, 3.5) # Latitude/longitude bounds of Catalonia (min_lat, min_lon, max_lat, max_lon)
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 128
    image_size = 128 # Image size for training
    seed = 42
    save_path = 'super_resolution_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set to GPU if available

    # --- 1. Load and preprocess the data for the city of Barcelona ---
    sconcno2_values, coords_sconcno2_filtered, caliope_urban_gdf, roads_filtered, population_filtered, lats_sconcno2, lons_sconcno2 = load_and_preprocess_data(data_dir, date_str, barcelona_bounds)

    # --- 2. Create the dataset and data loaders ---
    dataset = create_dataset(sconcno2_values, coords_sconcno2_filtered, caliope_urban_gdf, roads_filtered, population_filtered,  lats_sconcno2, lons_sconcno2, batch_size, image_size, seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 3. Initialize the model and train it ---
    model = SuperResolutionModel().to(device)
    trained_model, train_losses, test_losses = train_model(model, train_loader, test_loader, learning_rate, num_epochs, device, save_path)

    # --- 4. Inference for whole Catalonia ---
    final_prediction, target_lat_grid, target_lon_grid = predict_whole_catalonia(trained_model, data_dir, date_str, catalonia_bounds, device, image_size, seed)

    # --- 5. Visualization ---
    # Create a Folium map centered on the mean latitude and longitude
    m = folium.Map(location=[np.mean(target_lat_grid), np.mean(target_lon_grid)], zoom_start=7)

    # Ensure that the latitudes and longitudes are correctly ordered and not flattened
    lats = target_lat_grid
    lons = target_lon_grid
    
    # Add the image overlay
    folium.raster_layers.ImageOverlay(
        image=final_prediction,  # Transpose if necessary for correct orientation
        bounds=[[np.min(lats), np.min(lons)], [np.max(lats), np.max(lons)]],
        opacity=0.7,
        colormap=lambda x: plt.cm.viridis(x) if x is not None else (0, 0, 0, 0) # Set colormap and transparency for nan values
    ).add_to(m)

    # Save the map to an HTML file
    m.save("super_resolution_map.html")
    print("Super-resolution map generated in super_resolution_map.html")