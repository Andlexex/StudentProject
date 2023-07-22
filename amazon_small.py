import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
import torch_geometric.utils as utils

data_path = "csv_files/All_Beauty.csv"
df = pd.read_csv(data_path, header=None, usecols=[0, 1, 2])
print("finish reading")
df.columns = ["user_id", "product_id", "rating"]
user_col = df['user_id'].values
product_col = df['product_id'].values
rating_col = df['rating'].values

user_to_index = {user_id: index for index, user_id in enumerate(set(user_col))}
product_to_index = {product_id: index for index, product_id in enumerate(set(product_col))}

user_index_col = [user_to_index[user_id] for user_id in user_col]
product_index_col = [product_to_index[product_id] for product_id in product_col]

rating_tensor = torch.tensor(rating_col, dtype=torch.float)

# Create separate tensors for source nodes (users) and target nodes (products)
user_tensor = torch.tensor(user_index_col, dtype=torch.long)
product_tensor = torch.tensor(product_index_col, dtype=torch.long)

# For bipartite graphs, the edge_index tensor should have two rows
edge_index = torch.stack([user_tensor, product_tensor], dim=0)

# Update the num_nodes to consider both users and products
num_nodes = max(max(user_index_col), max(product_index_col)) + 1
print("Value of num_nodes", num_nodes)

data = Data(edge_index=edge_index, y=rating_tensor, num_nodes=num_nodes)

indices = list(range(data.edge_index.size(1)))
device = torch.device('cpu')

# Use PyTorch's random_split to split the dataset into train, test, and validation sets

train_indices, test_indices = train_test_split(indices, train_size=0.1,test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, train_size=0.8, test_size=0.2, random_state=42)

train_data = data.__class__()
test_data = data.__class__()
val_data = data.__class__()

# Train data
train_data.edge_index = data.edge_index[:, train_indices]
train_data.y = data.y[train_indices]

# Test data
test_data.edge_index = data.edge_index[:, test_indices]
test_data.y = data.y[test_indices]

# Validation data
val_data.edge_index = data.edge_index[:, val_indices]
val_data.y = data.y[val_indices]


class GATModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GATModel, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels)
        self.conv4 = GATv2Conv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x




torch.manual_seed(42)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------------------------------------------------

hidden_channels = 8
lr = 0.01
epochs = 100
batch_size = 4

# Early Stopping
patience = 15
min_delta = 0.1

model = GATModel(num_features=1, hidden_channels=hidden_channels, num_classes=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Model training
model.train()

train_losses = []
val_losses = []
best_val_loss = float('inf')
early_stop_counter = 0

import matplotlib.pyplot as plt

for epoch in range(epochs):
    # Training
    train_loss = 0.0
    
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.y.unsqueeze(1), batch.edge_index)
        loss = criterion(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    
    # Calculate average training loss
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    torch.cuda.empty_cache()
    # Validation
    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        out = model(batch.y.unsqueeze(1), batch.edge_index)
        loss = criterion(out, batch.y)
        val_loss += loss.item() * batch.num_graphs

    # Calculate average validation loss
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Print training and validation loss for monitoring
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    # Check for early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_epoch = epoch
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Set the model back to training mode
    model.train()

# Plotting training and validation curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot as an image file
plt.savefig('loss_plot.png')

# Show the plot

# Evaluation on the test set
model.eval()
with torch.no_grad():
    predictions = []
    targets = []

    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.y.unsqueeze(1), batch.edge_index)
        test_loss = criterion(out, batch.y)
        print(f'Test Loss: {test_loss.item()}')
        predictions.extend(out.cpu().numpy().flatten())
        targets.extend(batch.y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = np.mean(np.abs(predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
