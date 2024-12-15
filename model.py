import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Data Loading and Preprocessing ---

class RegionsDataset(Dataset):
    def __init__(self, data_dir, static_features_file=None):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        
        # Load static features from the first file if not provided
        if static_features_file is None:
            static_features_file = os.path.join(data_dir, self.file_list[0])

        with open(static_features_file, 'r') as f:
            first_file_data = json.load(f)
        
        self.static_features = {}
        for region in first_file_data:
            bbox_key = (region['bbox_x'], region['bbox_y'])
            self.static_features[bbox_key] = {
                'has_road': torch.tensor(region['features']['has_road'], dtype=torch.float32),
                'elevation': torch.tensor(region['features']['elevation'], dtype=torch.float32),
                'imd_total': torch.tensor(region['features']['imd_total'], dtype=torch.float32)
            }

        # Determine the number of unique regions based on static features
        self.unique_regions = list(self.static_features.keys())
        self.num_regions = len(self.unique_regions)
        self.num_files = len(self.file_list)

    def __len__(self):
      
        return self.num_regions * self.num_files

    def __getitem__(self, idx):
        # Determine the file index and the region index from the overall index
        file_idx = idx // self.num_regions
        region_idx = idx % self.num_regions
        
        # Get the bbox_key for the current region
        bbox_key = self.unique_regions[region_idx]

        # Load data from the corresponding file
        file_name = self.file_list[file_idx]
        with open(os.path.join(self.data_dir, file_name), 'r') as f:
            data = json.load(f)

        # Find the region with the matching bbox_key
        region_data = None
        for region in data:
            if (region['bbox_x'], region['bbox_y']) == bbox_key:
                region_data = region
                break
        
        if region_data is None:
            raise ValueError(f"Region with bbox_key {bbox_key} not found in file {file_name}")

        # Get static features from the pre-loaded dictionary
        static_features = self.static_features[bbox_key]

        # Combine features
        no2_conc = torch.tensor(region_data['features']['no2_conc'], dtype=torch.float32) * 100
        pollutant_uk = torch.tensor(region_data['features']['pollutant_uk'], dtype=torch.float32)

        input_features = torch.stack([
            static_features['has_road'],
            static_features['elevation'],
            static_features['imd_total'],
            no2_conc
        ], dim=0)  # Shape: [4, 40, 40]
        

        return input_features, pollutant_uk

# --- Model Definition (UNet-like Architecture) ---
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

class UNet(nn.Module):
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
    
# --- Training Loop ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses

# --- Main Function ---

def main():
    data_dir = 'processed_regions'
    static_features_file = os.path.join(data_dir, 'regions_20230101_0.json')
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and split into train and validation sets
    full_dataset = RegionsDataset(data_dir, static_features_file=static_features_file)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    in_channels = 4  # has_road, elevation, imd_total, no2_conc
    out_channels = 1  # pollutant_uk
    model = UNet(in_channels, out_channels)
    criterion = nn.MSELoss()  # You might want to consider other loss functions like L1Loss or a combination
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

    # (Optional) Plot training and validation losses
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()