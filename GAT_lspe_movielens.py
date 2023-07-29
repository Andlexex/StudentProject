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
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp
import numpy as np
import networkx as nx

import numpy as np
from scipy.sparse import csr_matrix



def calculatePosEncodings(edge_index, num_nodes):
    edge_index = edge_index.t().tolist()
    edges = [(src, dst) for src, dst in edge_index]

# Create the adjacency matrix in CSR format -> das wird dann für die encodings benutzt!
    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    ''' this code computes the in_degrees matrix from the edge list. it can later be adapted to compute the in-degrees matrix from the adjacency matrix (however, then, we should
    do some tests with small sample graphs to ensure everything is correct
    '''
    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    # Calculate the in-degrees for each node
    for edge in edges:
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i] for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    in_degrees = np.power(in_degrees, -0.5)  # Take the element-wise inverse square root

    # Create the sparse diagonal matrix N
    N = sp.diags(in_degrees, dtype=float)

    L = sp.eye(num_nodes) - N * A * N

    #calc eigvals and eigVecs, equivalent to the original code
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])

    #pos_enc_dim = hyperparameter!
    pos_enc_dim = 1
    RESULT_POS_ENCODING = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    return RESULT_POS_ENCODING


def precision(predictions, targets, threshold):
    # Apply a threshold to the predictions
    binary_predictions = (predictions >= threshold).astype(int)
    binary_targets = (targets >= threshold).astype(int)

    # Calculate the true positive (TP) and false positive (FP) counts
    TP = np.sum((binary_predictions == 1) & (binary_targets == 1))
    FP = np.sum((binary_predictions == 1) & (binary_targets == 0))

    print("Negative: ")
    print(np.sum(binary_targets == 1))
    print("Positive: ")
    print(np.sum(binary_targets == 0))
    # Calculate precision
    precision_value = TP / (TP + FP)
    return precision_value

def recall(predictions, targets, threshold):
    # Apply a threshold to the predictions
    binary_predictions = (predictions >= threshold).astype(int)
    binary_targets = (targets >= threshold).astype(int)
    # Calculate the true positive (TP) and false negative (FN) counts
    TP = np.sum((binary_predictions == 1) & (binary_targets == 1))
    FN = np.sum((binary_predictions == 0) & (binary_targets == 1))
    # Calculate recall
    recall_value = TP / (TP + FN)
    return recall_value

# Step 1: Load the MovieLens dataset manually
data_path = 'u.data'  # Set the path to the dataset file
df = pd.read_csv(data_path, delimiter='\t')
#print(df)

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

#####TOOOOOOOOODDDDDDDDDDDDDDDDOOOOOOOOOOOOOOOOOOOOOOOOOOO
positional_encodings = calculatePosEncodings(edge_index, num_nodes)

# Create the data object for the entire dataset. IMPORTANT: this is not according to the documentation, because y are edge features here! 
data = Data(edge_index=edge_index, y=rating_tensor, num_nodes=num_nodes, positional_encodings = positional_encodings)






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
train_data.positional_encodings = data.positional_encodings




test_data.edge_index = data.edge_index[:, test_indices]
test_data.y = data.y[test_indices]
test_data.num_nodes = data.num_nodes
test_data.positional_encodings = data.positional_encodings


val_data.edge_index = data.edge_index[:, val_indices]
val_data.y = data.y[val_indices]
val_data.num_nodes = data.num_nodes
val_data.positional_encodings = data.positional_encodings





class GATModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads):
        super(GATModel, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads)
        self.conv4 = GATv2Conv(hidden_channels, num_classes, heads)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x

''' später: einfach edge features anstatt x!
WICHTIG: x sind edge features!!!'''
class LSPEGAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads):
        super(LSPEGAT, self).__init__()
        self.gat = GATModel(num_features, hidden_channels, num_classes, heads)

    def forward(self, x, edge_index, pos_embeddings):

        #edge features: concatenate node features and edge features (TODO when adding node features)
        x = self.gat.conv1(x, edge_index)
        x = F.relu(x)
        x = self.gat.conv4(x, edge_index)

        print(pos_embeddings.shape)
        pos_embeddings_init = pos_embeddings.view(-1, pos_embeddings.size(2))
        pos_embeddings = self.gat.conv1(pos_embeddings_init, edge_index)
        pos_embeddings = F.relu(pos_embeddings)
        pos_embeddings  = self.gat.conv4(pos_embeddings, edge_index)

        return x, pos_embeddings


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
min_delta = 0.001  # Minimum improvement required to consider as improvement

best_val_loss = np.inf
best_epoch = 0
early_stop_counter = 0

# Define the GCNModel
#model = GATModel(num_features = 1, hidden_channels = hidden_channels, num_classes = 1, heads = 1).to(device)

#model = GINModel(num_features = 1, num_classes = 1).to(device)
#model = GCN(num_features=1, hidden_channels=hidden_channels, num_classes=1).to(device)

#------------------------------------------------------
#loss function, and optimizer, MSE = Metrik für Loss 
criterion = nn.MSELoss()
model = LSPEGAT(num_features = 1, hidden_channels = hidden_channels, num_classes = 1, heads = 1)

optimizer = optim.Adam(model.parameters(), lr=lr)

# Create data loaders for training and test sets
train_loader = DataLoader([train_data], batch_size=batch_size)
test_loader = DataLoader([test_data], batch_size=batch_size)
val_loader = DataLoader([val_data], batch_size=batch_size)

# Model training

train_losses = []
val_losses = []
predictions =[]

import matplotlib.pyplot as plt

for epoch in range(epochs):
    # Training
    train_loss = 0.0
    
    for batch in train_loader:
        batch = batch.to(device)
        out, out_pos_embeddings = model(batch.y.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1))
        loss = criterion(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
        predictions = out.detach().cpu().numpy()
        #print(predictions)

    # Calculate average training loss
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        batch = batch.to(device)
        out, out_pos_embeddings = model(batch.y.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1))
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


# Show the plot

# Evaluation on test set
model.eval()
with torch.no_grad():
    predictions = []
    targets = []

    for batch in test_loader:
        batch = batch.to(device)
        out, out_pos_embeddings = model(batch.y.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1))
        test_loss = criterion(out, batch.y)
        print(f'Test Loss: {test_loss.item()}')
        predictions.extend(out.cpu().numpy().flatten())
        targets.extend(batch.y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rounded_predictions = np.round(predictions, decimals=0)  # Round the predicted ratings

    # Plotting the distribution
    plt.hist(rounded_predictions, bins=15, edgecolor='black')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Ratings')
    plt.xticks(range(1, 16))
    #plt.show()
    plt.savefig('predicted_rankings.png')


    mse = np.mean(np.abs(predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    k = 5  # Define the value of k

    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print   (f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    
    threshold = 3.5 # Define the threshold to convert predictions into binary values
    np.set_printoptions(threshold=sys.maxsize)  # Set the threshold to print the whole array

    #print(rounded_predictions)
    #print(predictions)

    GAT_results = open("predictions_GAT.txt", "a")

    GAT_results.write(str(predictions))
    
    GAT_results.close()

    precision_value = precision(predictions, targets, threshold)
    recall_value = recall(predictions, targets, threshold)

    print(f"Precision: {precision_value}")
    print(f"Recall: {recall_value}")


