import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
from shapely.geometry import Point, box
import xarray as xr
from glob import glob
from tqdm import tqdm
from pyproj import Transformer
# --- Model Definition (same as before) ---

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Center crop x2 to match the size of x1
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = x2[:, :, diffY // 2:x2.size()[2] - diffY // 2, diffX // 2:x2.size()[3] - diffX // 2]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet40x40(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = DownConv(in_channels, 32)
        self.down2 = DownConv(32, 64)
        self.down3 = DownConv(64, 128)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = UpConv(256, 128)
        self.up3 = UpConv(128, 64)
        self.up4 = UpConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1_pool, x1 = self.down1(x)  # 40 -> 20
        x2_pool, x2 = self.down2(x1_pool)  # 20 -> 10
        x3_pool, x3 = self.down3(x2_pool)  # 10 -> 5
        x4 = self.bottleneck(x3_pool)  # 5
        x = self.up2(x4, x3)  # 5 -> 10 
        x = self.up3(x, x2)  # 10 -> 20
        x = self.up4(x, x1)  # 20 -> 40
        return self.out_conv(x)

# --- Prediction Data Preparation ---

def geojson_to_matrix(geojson, variable, grid_size=40):
    """
    Transforms GeoJSON into a 40x40 matrix of `mean_imd_tot` values.

    Parameters:
        geojson (dict): Input GeoJSON object.
        grid_size (int): Size of the output matrix (default is 40x40).

    Returns:
        np.ndarray: 40x40 matrix of `mean_imd_tot` values.
    """
    # Extract features
    features = geojson.get("features", [])

    # Take the first `grid_size * grid_size` features
    max_features = grid_size * grid_size
    selected_features = features[:max_features]

    # Extract `mean_imd_tot` values, default to 0 if not present
    mean_imd_tot_values = [
        feature["properties"].get(variable, 0.0) for feature in selected_features
    ]

    # Pad with zeros if less than required number of values
    if len(mean_imd_tot_values) < max_features:
        mean_imd_tot_values.extend([0.0] * (max_features - len(mean_imd_tot_values)))

    # Convert to 40x40 matrix
    matrix = np.array(mean_imd_tot_values).reshape(grid_size, grid_size)

    return matrix



def extract_value_with_crs_transformation(predictions, target_lon, target_lat, grid_xs, grid_ys, grid_size_m, sub_resolution_m, target_crs, grid_crs):
    """
    Extract the predicted value for a specific lon/lat from the predictions, handling CRS transformation.
    """
    # Transform target coordinates to the grid CRS
    transformer = Transformer.from_crs(target_crs, grid_crs, always_xy=True)
    target_x, target_y = transformer.transform(target_lon, target_lat)

    print(f"Transformed coordinates: ({target_x}, {target_y}) in CRS {grid_crs}")


    # 392867.50915097236 393867.50915097236


    # Determine the bounding box of the grid cell containing the target point
    for bbox_x in grid_xs:
        for bbox_y in grid_ys:
            bbox_min_x, bbox_min_y = bbox_x, bbox_y
            bbox_max_x, bbox_max_y = bbox_x + grid_size_m, bbox_y + grid_size_m
            
            # Check if target_point is within this grid cell
            if bbox_min_x <= target_x < bbox_max_x and bbox_min_y <= target_y < bbox_max_y:
                bbox_key = str((bbox_x, bbox_y))
                print(f"Target point is in grid cell with bbox {bbox_key}")

                # Retrieve the corresponding prediction matrix
                prediction_matrix = predictions.get(bbox_key)
                if prediction_matrix is None:
                    raise ValueError(f"No prediction data available for bbox {bbox_key}")

                # Map target_point to sub-resolution grid within the cell
                relative_x = (target_x - bbox_min_x) / sub_resolution_m
                relative_y = (target_y - bbox_min_y) / sub_resolution_m

                # Convert to matrix indices
                idx_x = int(relative_x)
                idx_y = int(relative_y)

                # Extract the predicted value
                predicted_value = prediction_matrix[idx_y][idx_x]  # Note: Y comes first in matrix indexing
                print(f"Predicted value at target coordinates: {predicted_value}")

                return predicted_value

    # If the point is not found in any grid cell
    raise ValueError("Target coordinates are not within the prediction grid.")


target_lon = 2.009802
target_lat = 41.39216
target_lon = 1.191975
target_lat = 41.11588

#1.191975	41.11588
#2.237875	41.44398
#2.082141	41.32177



target_crs = "EPSG:4326"  # WGS84
grid_crs = "EPSG:32631"


def create_prediction_data(no2_file, roads_grid, elevation_grid, grid_size_m=1000, sub_resolution_m=25, hour=0):
    """
    Prepares input data for prediction from NO2 data and GeoDataFrames.
    """

    # Load NO2 data for the specified hour
    no2_data = xr.open_dataset(no2_file, engine='netcdf4')
    no2_data_hour = no2_data.isel(time=hour)
    no2df = no2_data_hour[["sconcno2", "lat", "lon"]].to_dataframe().reset_index()

    # Create GeoDataFrame from NO2 data
    no2_gdf = gpd.GeoDataFrame(
        no2df,
        geometry=gpd.points_from_xy(no2df['lon'], no2df['lat']),
        crs='EPSG:4326'
    )

    # Ensure consistent CRS
    if no2_gdf.crs != roads_grid.crs:
        no2_gdf = no2_gdf.to_crs(roads_grid.crs)

    # Spatial indexing for faster operations
    roads_grid.sindex
    no2_gdf.sindex
    elevation_grid.sindex

    # Grid creation
    min_x, min_y, max_x, max_y = no2_gdf.total_bounds
    grid_xs = np.arange(min_x, max_x + grid_size_m, grid_size_m)
    grid_ys = np.arange(min_y, max_y + grid_size_m, grid_size_m)
    cell_xs = np.arange(0, grid_size_m, sub_resolution_m)
    cell_ys = np.arange(0, grid_size_m, sub_resolution_m)

    max_no2 = no2_gdf['sconcno2'].max()

    features = {
        'has_road': geojson_to_matrix(roads_grid, 'has_road'),
        'elevation': geojson_to_matrix(elevation_grid, 'elevation_mean'),
        'imd_total': geojson_to_matrix(roads_grid, 'mean_imd_tot'),
        'no2_conc': np.full((40, 40), max_no2)
    }

    prediction_data = []

    prediction_data.append({
        'bbox_x': grid_xs[1],
        'bbox_y': grid_ys[1],
        'features': features
    })


    predicted_value = extract_value_with_crs_transformation(
        predictions=prediction_data,
        target_lon=target_lon,
        target_lat=target_lat,
        grid_xs=grid_xs,
        grid_ys=grid_ys,
        grid_size_m=1000,
        sub_resolution_m=25,
        target_crs=target_crs,
        grid_crs=grid_crs
    )

    return predicted_value

# --- Prediction Function ---

def predict(model, prediction_data, device):
    """
    Makes predictions using the trained model.
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for region in tqdm(prediction_data, desc="Predicting"):
            # Create input tensor
            input_tensor = torch.stack([
                torch.tensor(region['features']['has_road'], dtype=torch.float32),
                torch.tensor(region['features']['elevation'], dtype=torch.float32),
                torch.tensor(region['features']['imd_total'], dtype=torch.float32),  # Assuming you have this
                torch.tensor(region['features']['no2_conc'], dtype=torch.float32)
            ], dim=0).unsqueeze(0).to(device)  # Add batch dimension

            # Make prediction
            output = model(input_tensor)
            output = output.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

            # Store prediction
            bbox_key = str((region['bbox_x'], region['bbox_y']))
            predictions[bbox_key] = output.tolist()
            print("KEEYYY", bbox_key)
            break

    return predictions

# --- Main Function ---

def main():
    # Paths and parameters
    no2_files = sorted(glob('../NO2/sconcno2_*.nc'))  # Update with your path
    roads_grid_file = 'pred1roads.geojson'
    elevation_grid_file = 'pred1_with_elevation.geojson'
    model_path = 'trained_model.pth'
    output_dir = 'predictions'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GeoDataFrames
    roads_grid = gpd.read_file(roads_grid_file)
    elevation_grid = gpd.read_file(elevation_grid_file)

    # Load the trained model
    in_channels = 4
    out_channels = 1
    model = UNet40x40(in_channels, out_channels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Make predictions for each hour of each day
    os.makedirs(output_dir, exist_ok=True)

    for no2_file in no2_files:
        date_str = os.path.basename(no2_file).split('_')[1] 
        print(f"Processing {no2_file}...")

        for hour in range(24):
            print(f"  Hour: {hour}")

            # Create input data for prediction
            prediction_data = create_prediction_data(no2_file, roads_grid, elevation_grid, hour=hour)

            # Make predictions
            predictions = predict(model, prediction_data, device)
            print(predictions)

            # Save predictions (you might want to customize the format)
            output_file = os.path.join(output_dir, f"predictions_{date_str}_{hour}.json")
            with open(output_file, 'w') as f:
                json.dump(predictions, f)

if __name__ == "__main__":
    main()