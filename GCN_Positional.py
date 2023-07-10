#!/usr/bin/env python
# coding: utf-8

# In[56]:


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
from torch_geometric.transforms import NormalizeFeatures


from gnn_lspe.layers.gatedgcn_lspe_layer import GatedGCNLSPELayer


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
train_indices, test_indices = train_test_split(indices, train_size=0.1, test_size=0.1)

#irgendeine syntax
train_data = data.__class__()
test_data = data.__class__()

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






class GCNPositionalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GatedGCNLSPELayer(input_dim, hidden_dim, dropout=0.5, batch_norm=True)
        self.layer2 = GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout=0.5, batch_norm=True)
        self.layer3 = GatedGCNLSPELayer(hidden_dim, output_dim, dropout=0.5, batch_norm=True)
        
    def forward(self, g, features):
        h, p, e = features, torch.zeros_like(features), torch.zeros((g.number_of_edges(), 1))  # initialize p and e
        
        h, p, e = self.layer1(g, h, p, e, snorm_n=None)  # Set snorm_n to None for simplicity
        h, p, e = self.layer2(g, h, p, e, snorm_n=None)
        h, p, e = self.layer3(g, h, p, e, snorm_n=None)
        
        return h

    
    
    
    
# Set up the training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNPositionalModel(1, 16, 1).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()


train_loader = DataLoader([train_data], batch_size=1)
test_loader = DataLoader([test_data], batch_size=1)


# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(data, features)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
    
    epoch_loss = total_loss / len(train_data)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

