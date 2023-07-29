#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Here we are using the Lastfm_subset as the test set and the Lastfm_test set as the train set duo to to much information.
#Please put those two sets in the same directory as this file and adjust the path where needed.
#If this programm seems to run for years, please adjust the amount of data in the test and train set to a minumum. As we are
#deleting all edges that lead to nodes which are not in the dataset it takes quiet a while(for me it took with that amount
#of data aprox 2h ;)). When taking a look into the structure of the data you will see that they are all JSON files(damn).
#So you can adjust the amount of data by changing the alphabet in the onlySomeLetters and above (maybe 
#not the best implementation).

import json
import pandas
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split


# In[2]:


#Reading in the data into arrays, careful indexes are important!

LettersForA="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

LettersForBOutter="ABCDEFGH"
LettersForBInner = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lettersForBInnerLast="ABCDEFGHIJ"

artists = []
timestamps = []
similars = []
tags = []
track_ids = []
titles = []

for outterLetter in LettersForA:
    for innerLetter in LettersForA:
        directory_path ="C:/Users/andre/anaconda3/Python3Projects/ForschungsProjekt/lastfm_subset/A/"+outterLetter+"/"+innerLetter
        json_files = [file for file in os.listdir(directory_path)]
        for file in json_files:
            with open(os.path.join(directory_path, file), "r") as f:
                data = json.load(f)
                artists.append(data["artist"])
                timestamps.append(data["timestamp"])
                similars.append(data["similars"])
                tags.append(data["tags"])
                track_ids.append(data["track_id"])
                titles.append(data["title"])

for outterLetter in LettersForBOutter:
    for innerLetter in LettersForBInner:
        directory_path ="C:/Users/andre/anaconda3/Python3Projects/ForschungsProjekt/lastfm_subset/B/"+outterLetter+"/"+innerLetter
        json_files = [file for file in os.listdir(directory_path)]
        for file in json_files:
            with open(os.path.join(directory_path, file), "r") as f:
                data = json.load(f)
                artists.append(data["artist"])
                timestamps.append(data["timestamp"])
                similars.append(data["similars"])
                tags.append(data["tags"])
                track_ids.append(data["track_id"])
                titles.append(data["title"])
                
for innerLetter in lettersForBInnerLast:
    directory_path ="C:/Users/andre/anaconda3/Python3Projects/ForschungsProjekt/lastfm_subset/B/I/"+innerLetter
    json_files = [file for file in os.listdir(directory_path)]
    for file in json_files:
        with open(os.path.join(directory_path, file), "r") as f:
            data = json.load(f)
            artists.append(data["artist"])
            timestamps.append(data["timestamp"])
            similars.append(data["similars"])
            tags.append(data["tags"])
            track_ids.append(data["track_id"])
            titles.append(data["title"])
              
look = 0
print(artists[look])
print(timestamps[look])
print(similars[look])
print(tags[look])
print(track_ids[look])
print(titles[look])

dataLength = len(artists)



# In[14]:


lettersForTrain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
onlySomeLetters="ABCDEFGHIJKLMNOP"

artists_train = []
timestamps_train = []
similars_train = []
tags_train = []
track_ids_train = []
titles_train = []

for outterLetter in onlySomeLetters:
    for middleLetter in onlySomeLetters:
        for innerLetter in onlySomeLetters:
            try:
                directory_path ="C:/Users/andre/anaconda3/Python3Projects/ForschungsProjekt/lastfm_test/lastfm_test/"+outterLetter+"/"+middleLetter+"/"+innerLetter
                json_files = [file for file in os.listdir(directory_path)]
                for file in json_files:
                    with open(os.path.join(directory_path, file), "r") as f:
                        data = json.load(f)
                        artists_train.append(data["artist"])
                        timestamps_train.append(data["timestamp"])
                        similars_train.append(data["similars"])
                        tags_train.append(data["tags"])
                        track_ids_train.append(data["track_id"])
                        titles_train.append(data["title"])                
            except FileNotFoundError as e:
                continue
                
print(len(artists_train))


# In[27]:


#Fun thing, in the similars are sometimes songs, that are not represented in the current dataset.
#That means there are edges pointing to Nodes that our dataset doesnt know. To fix that I removed those similars.
#Pain.
def removeFromSimilars(similars_set, track_ids_set):
    i_to_remove=[]
    j_to_remove=[]
    all_similars = 0
    for i in range(len(similars_set)):
        for j in range(len(similars_set[i])):
            all_similars = all_similars + 1
            if(not track_ids_set.count(similars_set[i][j][0])):
                i_to_remove.append(i)
                j_to_remove.append(j)
        if(i % 1000 == 0):
            print(i)
# This inverts the arrays so we delete from the tails of all arrays so we have no problems with indexes
# Kinda evil
    i_to_remove = i_to_remove[::-1]
    j_to_remove = j_to_remove[::-1]

#More Pain

    for k in range(len(i_to_remove)):
         del(similars_set[i_to_remove[k]][j_to_remove[k]])


# In[28]:


removeFromSimilars(similars, track_ids)
print("done1")
removeFromSimilars(similars_train, track_ids_train)
print("done2")


# In[ ]:


#Deletes all information of all nodes that have no edges left.
#Well can't do that bc for some reason the graph is directed. We leave the nodes there although it is possible to delete 
#nodes with no incoming and outgoing edges, but I am not going to implement that.

#nodes_no_edges_indexes = []
#for i in range(len(similars)):
#    if(len(similars[i]) == 0):
#        nodes_no_edges_indexes.append(i)
#nodes_no_edges_indexes = nodes_no_edges_indexes[::-1]
#
#for i in range(len(nodes_no_edges_indexes)):
#    del(artists[nodes_no_edges_indexes[i]])
#    del(timestamps[nodes_no_edges_indexes[i]])
#    del(similars[nodes_no_edges_indexes[i]])
#    del(tags[nodes_no_edges_indexes[i]])
#    del(track_ids[nodes_no_edges_indexes[i]])
#    del(titles[nodes_no_edges_indexes[i]])


# In[57]:


#Index mapping as torchgeometric does not like strings

track_id_to_index = {track_id: index for index, track_id in enumerate(set(track_ids))}
track_index_col = [track_id_to_index[track_id] for track_id in track_ids]

#All tracks with the corresponding similar tracks(tested)
edges = []
edges_source = []
edges_dest = []
for i in range(len(track_ids)):
    for j in range(len(similars[i])):
        edges_source.append(track_ids[i])
        edges_dest.append(similars[i][j][0])
        
track_edges_source_index_mapping = [track_id_to_index[track_id] for track_id in edges_source]
track_edges_dest_index_mapping = [track_id_to_index[track_id] for track_id in edges_dest]


edge_index_tensor = torch.tensor([track_edges_source_index_mapping, track_edges_dest_index_mapping], dtype=torch.long)

similar_weights = []

for i in range(len(similars)):
    for j in range(len(similars[i])):
        similar_weights.append(similars[i][j][1])

#print(similar_weights)
similar_tensor = torch.tensor(similar_weights, dtype=torch.float)

num_nodes = len(track_ids)
data_test = Data(edge_index=edge_index_tensor, y = similar_tensor, num_nodes=len(similar_tensor))
print(data_test)


# In[59]:


#Index mapping as torchgeometric does not like strings
track_id_to_index_train = {track_id: index for index, track_id in enumerate(set(track_ids_train))}
track_index_col_train = [track_id_to_index_train[track_id] for track_id in track_ids_train]

#All tracks with the corresponding similar tracks(tested)
edges_train = []
edges_source_train = []
edges_dest_train = []
for i in range(len(track_ids_train)):
    for j in range(len(similars_train[i])):
        edges_source_train.append(track_ids_train[i])
        edges_dest_train.append(similars_train[i][j][0])
        
track_edges_source_index_mapping_train = [track_id_to_index_train[track_id] for track_id in edges_source_train]
track_edges_dest_index_mapping_train = [track_id_to_index_train[track_id] for track_id in edges_dest_train]


edge_index_tensor_train = torch.tensor([track_edges_source_index_mapping_train, track_edges_dest_index_mapping_train], dtype=torch.long)

similar_weights_train = []

for i in range(len(similars_train)):
    for j in range(len(similars_train[i])):
        similar_weights_train.append(similars_train[i][j][1])

#print(similar_weights)
similar_tensor_train = torch.tensor(similar_weights_train, dtype=torch.float)

num_nodes = len(track_ids_train)
data_train=Data(edge_index=edge_index_tensor_train, y = similar_tensor_train, num_nodes=len(similar_tensor_train))

print(data_train)


# In[29]:


class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# In[60]:


torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=1, hidden_channels=16, num_classes=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_loader = DataLoader([data_train], batch_size=1)
test_loader = DataLoader([data_test], batch_size=1)

model.train()
for epoch in range(100):
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.y.unsqueeze(1), batch.edge_index)
        loss = criterion(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print training loss for each epoch
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.y.unsqueeze(1), batch.edge_index)
        test_loss = criterion(out, batch.y)
        print(f'Test Loss: {test_loss.item()}')

