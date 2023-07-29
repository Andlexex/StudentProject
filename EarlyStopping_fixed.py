 #!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GINConv
from sklearn.model_selection import train_test_split

# Step 1: Load the MovieLens dataset manually
data_path = 'u.data'  # Set the path to the dataset file
df = pd.read_csv(data_path, delimiter='\t')

# Step 2: Preprocess the dataset
# Extract the user, movie, and rating columns
user_col = df['user_id'].values
movie_col = df['item_id'].values
rating_col = df['rating'].values

# Create a dictionary to map unique user and movie IDs to continuous indices
#user_id -> index  (where index is continously numerated!) and movie_id -> index
# you could leave this out, but we had issues without that so we added it into our code
user_to_index = {user_id: index for index, user_id in enumerate(set(user_col))}
movie_to_index = {movie_id: index for index, movie_id in enumerate(set(movie_col))}


# Map the user and movie columns to their corresponding indices
# das ist KEIN dictionary mehr, sondern eine Liste  [index_zeile1, index_zeile2, index_zeile3, ..., index_zeile_n]
user_index_col = [user_to_index[user_id] for user_id in user_col]
movie_index_col = [movie_to_index[movie_id] for movie_id in movie_col]

# Convert the rating column to a PyTorch tensor
rating_tensor = torch.tensor(rating_col, dtype=torch.float)

# Step 3: Create the graph data
# Create edge indices for user-movie interactions
edge_index = torch.tensor([user_index_col, movie_index_col], dtype=torch.long)

# Set the number of nodes (users and movies)
num_nodes = len(user_to_index) + len(movie_to_index)

# Create the data object for the entire dataset
data = Data(edge_index=edge_index, y=rating_tensor, num_nodes=num_nodes)
print(data)


# Step 4: Split the data into training and test sets
indices = list(range(data.edge_index.size(1)))

#das hier klein, damit der Speicher nicht überdreht wird. Aber nicht zu klein, weil sonst kommt es zu problemen!
train_indices, test_indices = train_test_split(indices, train_size=0.1, test_size=0.2)
train_indices, val_indices = train_test_split(train_indices, train_size=0.8, test_size=0.2, random_state=42)


#irgendeine syntax
train_data = data.__class__()
test_data = data.__class__()
val_data = data.__class__()


#setzt die Parameter von train_data und test_data

#soweit ich es verstehe, sind alle 2.500 nodes im training und testset vorhanden. gesplittet werden nur die edges, d.h. 
#es ist nur ein subset der 100.000 edges im training set sowie im test set vorhanden
# also 10% der Bewertungen 

train_data.edge_index = data.edge_index[:, train_indices]
train_data.y = data.y[train_indices]
train_data.num_nodes = data.num_nodes

test_data.edge_index = data.edge_index[:, test_indices]
test_data.y = data.y[test_indices]
test_data.num_nodes = data.num_nodes

val_data.edge_index = data.edge_index[:, val_indices]
val_data.y = data.y[val_indices]
val_data.num_nodes = data.num_nodes

'''
Step 5: Define the Graph Convolutional Network (GCN) model. Hier könnte man dann die zusätzlichen Dinge einbauen
num_features = Anzahl input layer.  hidden_channels = Anzahl "output" layer, d.h. anzahl Parameter, die das Model versucht
sich herzuleiten und zu lernen. dadurch kann komplexität des models kontrolliert werden. wenn zB hidden_channels zu groß ist, 
dann passiert overfitting.
Soweit ich es verstehe, hat dieses Neural network nur 3 Layer: self.conv1, dann RELU, dann self.conv2. 
RELU wird hier genutzt, um non-linearity einzuführen. es ginge auch ohne. 
Actually ist der Loss ohne RELU sogar kleiner!

'''

class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        #first graph convolution
        self.conv1 = GCNConv(num_features, hidden_channels)
        #second graph convolution
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GATModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads):
        super(GATModel, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads)
        self.conv4 = GATv2Conv(hidden_channels, num_classes, heads)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
       # x = self.conv2(x, edge_index)
       # x = F.relu(x)
       # x = self.conv3(x, edge_index)
       # x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x
    
class GINModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
# Step 6: Train and evaluate the GCN model
# Set seed for reproducibility
torch.manual_seed(42)

# Set the device --> aktiviere GPU falls vorhanden
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#------------------------------------------------------

#hidden channels und epochs tunen
hidden_channels=8 #8 und 16
lr = 0.01  #0.01 vs 0.001 
epochs = 100  #100 vs 200
batch_size = 64

 #1, 16, 32 ,64, 128, 256, 512

#Early Stopping
patience = 15  # Number of epochs to wait for improvement
min_delta = 0.1  # Minimum improvement required to consider as improvement

best_val_loss = np.inf
best_epoch = 0
early_stop_counter = 0

# Define the GCNModel
#model = GATModel(num_features = 1, hidden_channels = hidden_channels, num_classes = 1, heads = 1).to(device)
#model = GINModel(num_features = 1, num_classes = 1).to(device)

model = GCN(num_features=1, hidden_channels=hidden_channels, num_classes=1).to(device)
#------------------------------------------------------
#loss function, and optimizer, MSE = Metrik für Loss 
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

# Create data loaders for training and test sets
train_loader = DataLoader([train_data], batch_size=batch_size)
test_loader = DataLoader([test_data], batch_size=batch_size)
val_loader = DataLoader([val_data], batch_size=batch_size)

# Model training
model.train()

train_losses = []
val_losses = []
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

    # Validation
    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        batch = batch.to(device)
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
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot as an image file
plt.savefig('loss_plot.png')

# Show the plot

# Evaluation on test set
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

    GAT_results = open("GAT_results_schwankungen.txt", "a")
    '''
    GAT_results.write(f"Hidden_layers: {4}, " + f"lr: {lr}, " + 
                            f"batch_size: {batch_size}, " + f"epoch: {epoch}, " + 
                            f"MSE: {mse}, " + f"RMSE: {rmse} " + "\n")
    '''
    GAT_results.close()

