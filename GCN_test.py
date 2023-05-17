#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# In[2]:


def graphInfo(data):
    print(data.num_nodes)
    print(data.num_edges)
    print(data.num_node_features)
    print(data.has_isolated_nodes())
    print(data.has_self_loops())
    print(data.is_directed())


# In[3]:


edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                           [0, 1, 0, 2, 1, 4]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [1], [-1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
graphInfo(data)


# In[4]:


dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

print(len(dataset))
print("------")
graphInfo(dataset[0])


# In[5]:




# Step 1: Load the MovieLens dataset manually
# Set the path to the dataset file
data_path = 'u.data'  
df = pd.read_csv(data_path, delimiter='\t')


print(df)


# Extract features, labels, and adjacency matrix from the dataset
num_nodes = df['user_id'].max()  # Maximum user ID
num_movies = df['item_id'].max()  # Maximum movie ID
print(num_nodes)
print(num_movies)
for i in df['rating']:
    if i == 0:
        print(i)


# In[6]:


features = torch.eye(num_nodes + num_movies)  # Identity matrix as node features
real_featurematrix = []
for i in range (2624):
    real_featurematrix.append(features[i][0])
# Combine user and movie IDs into a single ID range
df['user_id'] += num_movies

# Create the adjacency matrix
adj = torch.tensor(df[['user_id', 'item_id']].values.T, dtype=torch.long)
for i in range(5):
    print(adj[1][i])

# Extract the labels (ratings)
torch.set_printoptions(edgeitems=-1, threshold=float('inf'))
labels = torch.tensor(df['rating'].values, dtype=torch.float)

for i in range(5):
    print(labels[i])


# In[7]:


# Step 2: Import the GCN network
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    


# In[8]:


# Step 3: Prepare the data for GCN training
data = Data(x=real_featurematrix, edge_index=adj, edge_attr = labels)
graphInfo(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Step 4: Run the GCN network on MovieLens
model = GCN(data.num_nodes, 16, 5).to(device)  # Assuming 5 classes for ratings
print(model)


# In[9]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

"""
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y.long())
    loss.backward()
    optimizer.step()
"""

for epoch in range(100):
    train_running_loss = 0.0
    train_acc = 0.0
    
    ## forward + backprop + loss
    logits = model(data.x.to(device) ,data.edge_index)
    loss = criterion(logits, data.y.to(device))
    optimizer.zero_grad()
    loss.backward()

    ## update model params
    optimizer.step()

    train_running_loss += loss.detach().item()
    train_acc += (torch.argmax(logits, 1).flatten() == labels).type(torch.float).mean().item()
    
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f'           %(epoch, train_running_loss / i, train_acc/i))    
    
"""    
def test():
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask].long()
    return test_correct.sum().item() / data.test_mask.sum().item()

# Step 5: Train and test the GCN network
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - num_movies:] = True

for epoch in range(1, 201):
    train()
    if epoch % 10 == 0:
        accuracy = test()
        print(f'Epoch: {epoch:03d}, Test Accuracy: {accuracy:.4f}')
"""


# In[ ]:




