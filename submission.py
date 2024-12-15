
import pandas as pd
import json
from datetime import datetime, timedelta

json_files = ["./predictions/predictions_complete1.json", "./predictions/predictions_complete2.json", "./predictions/predictions_complete3.json", "./predictions/predictions_complete4.json"]

# Latitude and Longitude values
lats = [41.39216, 41.11588, 41.44398, 41.32177]
lons = [2.009802, 1.191975, 2.237875, 2.082141]

# Start date
start_date = datetime(2023, 1, 1, 0, 0)

# Initialize an empty list to store rows
rows = []

# ID counter
row_id = 1

# Process each JSON file and associate with lat/lon
for file_idx, file in enumerate(json_files):
    with open(file, 'r') as f:
        data = json.load(f)
        # Repeat the 24 values for each hour of the year (365 days * 24 hours)
        data = data * 365

        # Generate rows for the current JSON
        for hour_idx, value in enumerate(data):
            # Calculate the timestamp for the prediction
            timestamp = start_date + timedelta(hours=hour_idx)

            # Append a row with id, date, lat, lon, and concentration
            rows.append({
                "id": row_id,
                "date": timestamp.strftime("%Y-%m-%d %H:%M"),
                "lat": lats[file_idx],
                "lon": lons[file_idx],
                "concentration": value
            })
            row_id += 1

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
output_path = "predictions_formatted.csv"
df.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
