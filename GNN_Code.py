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
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads)
        self.conv4 = GATv2Conv(hidden_channels, num_classes, heads)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x
    
    
class GINModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        ))
        #self.model = GIN(self.conv1, self.conv2, num_layers=2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        batch = utils.to_dense_batch(edge_index)[0]

        # Compute global_add_pool
        x = global_add_pool(x, batch)
        x = global_add_pool(x, edge_index)
        return x
        '''
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
        '''
   
# Set the number of input features (number of nodes) and output classes (1 in this case)

# Step 6: Train and evaluate the GCN model
# Set seed for reproducibility
torch.manual_seed(42)

# Set the device --> aktiviere GPU falls vorhanden
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#------------------------------------------------------
# Define the GCNModel
#model = GCN(num_features=1, hidden_channels=16, num_classes=1).to(device)
#print(model)

# Define the GATModel
model = GATModel(num_features = 1, hidden_channels = 16, num_classes = 1, heads = 1)

# Define the GINModel
#model = GINModel(num_features = 1, num_classes = 1)

print(model)
#------------------------------------------------------
#loss function, and optimizer, MSE = Metrik für Loss 
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create data loaders for training and test sets
train_loader = DataLoader([train_data], batch_size=1)
test_loader = DataLoader([test_data], batch_size=1)

# Model trainieren
model.train()

#das ist die standard metrik

#100x durchs komplette Trainingsset durchgehen, um Parameter zu updaten
for epoch in range(200):
  #alle batches durchiterieren (Batch = Anzahl Samples, die gleichzeitig betrachtet werden)
    for batch in train_loader:
        #Batch zum Device schieben, das ist hauptsächlich für Optimierungszwecke da!
        batch = batch.to(device)
        # Forward pass durch das Model. Input: Features (batch.y.unsqueeze(1)) und Edge indicies (batch.edge_index). out = output des Models mit diesem Input
        out = model(batch.y.unsqueeze(1), batch.edge_index)
        #Loss berechnen
        loss = criterion(out, batch.y)
        #This line zeroes out the gradients of all the model parameters. It is necessary before computing the gradients for the current batch, as PyTorch accumulates gradients by default.
        optimizer.zero_grad()
        #Backward propagation vom Loss
        loss.backward()
        #Parameter des Models optimieren
        optimizer.step()

    # Print training loss for each epoch
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
'''
WICHTIG: 
Hier haben wir Logistic Regression und keine Classification, d.h. man versucht, Ratings von Movies zu predicten! (zwischen 1 und 5)
--> für Classification müsste man das model ändern (z.B. 0 bei Rating > 3 sonst 1, d.h. wenn "gutes" Rating, dann wird user das wahrscheinlich schauen)
--> RSE und MRSE sind auch quasi die "ABweichungen", d.h. RSE = 1 heißt, im SChnitt ist unser Rating um 1 daneben
'''
# Evaluation
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

    # Convert predictions and targets to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate evaluation metrics
    mse = np.mean(np.abs(predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    # Print performance measures
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

