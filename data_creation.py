import pyreadr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import xarray as xr
from glob import glob
import os
import json
from concurrent.futures import ProcessPoolExecutor

def extract_spatial_regions(no2_df, uk_gdf, grid_size_m=1000, sub_resolution_m=25):
    """
    Extract 1km x 1km regions with 25m resolution features efficiently.

    Parameters:
    -----------
    no2_df : pandas.DataFrame or geopandas.GeoDataFrame
        Input NO2 concentration dataframe
    uk_gdf : geopandas.GeoDataFrame
        Input UK pollution and contextual data
    grid_size_m : int, optional
        Size of grid regions in meters (default 1000m = 1km)
    sub_resolution_m : int, optional
        Resolution of sub-grids (default 25m)

    Returns:
    --------
    list : List of dictionaries, each representing a 1km x 1km region
    """

    # 1. Efficient CRS Handling and GeoDataFrame Creation:
    if not isinstance(no2_df, gpd.GeoDataFrame):
        no2_gdf = gpd.GeoDataFrame(
            no2_df,
            geometry=gpd.points_from_xy(no2_df['lon'], no2_df['lat']),
            crs='EPSG:4326'
        )
    else:
        no2_gdf = no2_df

    # Ensure consistent CRS upfront to avoid repeated transformations
    if no2_gdf.crs != uk_gdf.crs:
        no2_gdf = no2_gdf.to_crs(uk_gdf.crs)

    # 2. Spatial Indexing for Faster Spatial Operations:
    uk_gdf.sindex  # Build spatial index for uk_gdf (if not already built)
    no2_gdf.sindex

    # 3. Optimized Grid Creation and Iteration:
    min_x, min_y, max_x, max_y = no2_gdf.total_bounds

    # Pre-compute grid coordinates
    grid_xs = np.arange(min_x, max_x + grid_size_m, grid_size_m)
    grid_ys = np.arange(min_y, max_y + grid_size_m, grid_size_m)

    # Pre-compute 25m cell coordinates
    cell_xs = np.arange(0, grid_size_m, sub_resolution_m)
    cell_ys = np.arange(0, grid_size_m, sub_resolution_m)

    regions = []

    # 4. Vectorized Operations within Grid Cells:
    for grid_x in grid_xs:
        for grid_y in grid_ys:
            grid_bbox = box(grid_x, grid_y, grid_x + grid_size_m, grid_y + grid_size_m)

            # Efficiently filter using spatial index
            possible_uk_matches_index = list(uk_gdf.sindex.intersection(grid_bbox.bounds))
            possible_uk_matches = uk_gdf.iloc[possible_uk_matches_index]
            uk_subset = possible_uk_matches[possible_uk_matches.intersects(grid_bbox)]

            possible_no2_matches_index = list(no2_gdf.sindex.intersection(grid_bbox.bounds))
            possible_no2_matches = no2_gdf.iloc[possible_no2_matches_index]
            no2_subset = possible_no2_matches[possible_no2_matches.intersects(grid_bbox)]

            if not no2_subset.empty and not uk_subset.empty:
                # Initialize feature arrays

                max_no2 = no2_subset['sconcno2'].max()
                features = {
                    'has_road': np.zeros((40, 40), dtype=np.int8),
                    'elevation': np.zeros((40, 40)),
                    'imd_total': np.zeros((40, 40)),
                    'no2_conc': np.full((40, 40), max_no2),
                    'pollutant_uk': np.zeros((40, 40)),
                }

                # Vectorized creation of cell geometries
                cell_boxes = [
                    box(grid_x + cell_x, grid_y + cell_y, grid_x + cell_x + sub_resolution_m,
                        grid_y + cell_y + sub_resolution_m)
                    for cell_y in cell_ys
                    for cell_x in cell_xs
                ]
                cell_boxes = gpd.GeoSeries(cell_boxes, crs=uk_gdf.crs)

                # Use spatial join for faster association
                uk_cell_matches = gpd.sjoin(cell_boxes.to_frame('geometry'), uk_subset, how='left',
                                            predicate='intersects')

                # Vectorized feature assignment using numpy indexing
                uk_cell_matches_grouped = uk_cell_matches.groupby(uk_cell_matches.index)

                # Assign values only if group exists
                if not uk_cell_matches.empty:
                    for index, group in uk_cell_matches_grouped:
                        i, j = divmod(index, 40)  # Reverse engineer i, j from flattened index
                        features['has_road'][i, j] = group['has_road'].iloc[0] if not np.isnan(
                            group['has_road'].iloc[0]) else 0
                        features['elevation'][i, j] = group['elevation_mean'].iloc[0] if not np.isnan(
                            group['elevation_mean'].iloc[0]) else 0
                        features['imd_total'][i, j] = group['mean_imd_tot'].iloc[0] if not np.isnan(
                            group['mean_imd_tot'].iloc[0]) else 0
                        features['pollutant_uk'][i, j] = group['pollutant_uk'].iloc[0] if not np.isnan(
                            group['pollutant_uk'].iloc[0]) else 0

                regions.append({
                    'bbox_x': grid_x,
                    'bbox_y': grid_y,
                    'features': {k: v.tolist() for k,v in features.items()}
                })

    return regions

def process_data_pair(urban_file, no2_file, roads_grid, output_dir):
    """
    Processes a pair of urban and NO2 data files, extracts spatial regions,
    and saves the results to a JSON file.
    """
    try:


        for hour in range(24):
            # Extract date from filenames for naming the output file
            date_str = os.path.basename(urban_file).split('.')[0].split('_')[1]

            # Load urban data
            urban = pyreadr.read_r(urban_file)
            df_urban = urban[None]
            df_urban['date'] = pd.to_datetime(df_urban['date'])
            df_urban_selected = df_urban[df_urban['date'] == df_urban['date'].unique()[hour]]
            gdf_urban = gpd.GeoDataFrame(df_urban_selected,
                                        geometry=gpd.points_from_xy(df_urban_selected.X, df_urban_selected.Y),
                                        crs="EPSG:32631")
            
            if roads_grid.crs != gdf_urban.crs:
                roads_grid = roads_grid.to_crs(gdf_urban.crs)

            # Step 3: Perform the spatial join to combine the data based on location
            # The join will attach columns from 'roads_grid' to the points in 'gdf_urban' that fall within the grid cells
            print(roads_grid.columns)

            
            gdf_joined = gpd.sjoin(gdf_urban, roads_grid, how="right", op='within')

            gdf_joined = gdf_joined.drop(['index_left'], axis=1)

            df = gdf_joined.copy()
            df['pollutant_uk'] = pd.to_numeric(df['pollutant_uk'], errors='coerce')
            df = df.reset_index(drop=True)
            df['pollutant_uk'] = df['pollutant_uk'].interpolate(method='linear')
            df['elevation_mean'] = df['elevation_mean'].fillna(0)
            df = df.to_crs(crs="EPSG:32631")
            df['centroid'] = df['geometry'].centroid
            df['lon'] = df['centroid'].x
            df['lat'] = df['centroid'].y

            # Load NO2 data
            no2_data = xr.open_dataset(no2_file, engine='netcdf4')
            first_time = no2_data.isel(time=hour)
            no2df = first_time[["sconcno2", "lat", "lon"]].to_dataframe().reset_index()

            # Ensure CRS consistency
            if gdf_urban.crs != "EPSG:32631":
                gdf_urban = gdf_urban.to_crs("EPSG:32631")

            # Extract regions
            regions = extract_spatial_regions(no2df, df)

            # Save regions to a JSON file
            output_file = os.path.join(output_dir, f"regions_{date_str}_{hour}.json")
            with open(output_file, 'w') as f:
                json.dump(regions, f)

            print(f"Processed and saved regions for {date_str} hour {hour}")
        return f"Successfully processed {date_str}"

    except Exception as e:
        print(f"Error processing {date_str}: {e}")
        return f"Failed to process {date_str}: {e}"

def main():
    urban_files = sorted(glob('../CALIOPE-Urban/UK_*.rds'))
    no2_files = sorted(glob('../NO2/sconcno2_*.nc'))
    output_dir = 'test_regions'  # Directory to save output files

    updated_grid = gpd.read_file('pred1_with_elevation.geojson')
    roads_grid = gpd.read_file('pred1roads.geojson')
    roads_grid['elevation_mean'] = updated_grid['elevation_mean']

    os.makedirs(output_dir, exist_ok=True)

    # Create a list of tuples, pairing urban and NO2 files
    file_pairs = list(zip(urban_files, no2_files))

    # Parallel processing
    for pair in file_pairs:
        process_data_pair(pair[0], pair[1], roads_grid, output_dir)

if __name__ == "__main__":
    main()