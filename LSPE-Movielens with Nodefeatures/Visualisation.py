import networkx as nx
import matplotlib.pyplot as plt


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
import torch_geometric as tg
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse as sp
import numpy as np
import networkx as nx
import sys
#from transformers import BertTokenizer, BertModel  # Example: Hugging Face Transformers library
import hashlib
import csv



def calculatePosEncodings_rswe(edge_index, num_nodes):
    edge_index = edge_index.t().tolist()
    edges = [(src, dst) for src, dst in edge_index]
    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    # Calculate the in-degrees for each node
    for edge in edges:
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i] for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    in_degrees = np.power(in_degrees, -1)  # Take the element-wise inverse square root

    Dinv = sp.diags(in_degrees, dtype=float)

    RW = A * Dinv  
    M = RW
    
    # das ist wieder ein Hyperparameter; sollte >1 sein weil eins immer 0 ist irgendwie!
    pos_enc_dim = 1

    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE,dim=-1)

    #ERGEBNIS
    RESULT_POS_ENCODING = PE 
    return RESULT_POS_ENCODING



# Step 1: Load the MovieLens dataset manually
data_path = 'ml-20m/ratings.csv'  # Set the path to the dataset file
df = pd.read_csv(data_path, delimiter=',')#, nrows=num_lines_to_read)#print(df)

# Step 2: Preprocess the dataset
# Extract the user, movie, and rating columns
user_col = df['userId'].values
movie_col = df['movieId'].values
rating_col = df['rating'].values


# Create a dictionary to map unique user and movie IDs to continuous indices
#user_id -> index  (where index is continously numerated!) and movie_id -> index
# you could leave this out, but we had issues without that so we added it into our code
user_to_index = {user_id: index for index, user_id in enumerate(set(user_col))}
movie_to_index = {movie_id: index for index, movie_id in enumerate(set(movie_col))}

print("finished indexing")

user_index_col = [user_to_index[user_id] for user_id in user_col]
movie_index_col = [movie_to_index[movie_id] for movie_id in movie_col]
rating_tensor = torch.tensor(rating_col, dtype=torch.float)

# Read the tag data
df_tags = pd.read_csv("ml-20m/tags.csv", delimiter=',')
tag_user_col = df_tags['userId'].values
tag_movie_col = df_tags['movieId'].values
tag_col = df_tags['tag'].values
tag_embeddings = []

''' LANGUAGE MODEL: SLOW!!!

model_name = 'bert-base-uncased'  # Example: BERT model name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Generate numerical representations for each tag
total_tags = len(tag_col)

for idx, tag in enumerate(tag_col):
    inputs = tokenizer(tag, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    tag_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    tag_embeddings.append(tag_embedding)
    
    percentage_processed = (idx + 1) / total_tags * 100
    print(f"Processed: {percentage_processed:.2f}%")
'''
for tag in tag_col:
    tag_str = str(tag)  # Convert the tag to a string
    hash_value = hashlib.sha256(tag_str.encode()).hexdigest()  # Hash the tag using SHA-256
    tag_embedding = int(hash_value, 16) % 100000  # Convert hash to integer and take modulo
    tag_embeddings.append(tag_embedding)

# Convert the list of tag embeddings to a tensor
tag_tensor = torch.tensor(tag_embeddings, dtype=torch.float)

# Filter the user-movie connections based on the tags data
filtered_user_indices = [user_to_index[user_id] for user_id, movie_id in zip(tag_user_col, tag_movie_col) if user_id in user_to_index and movie_id in movie_to_index]
filtered_movie_indices = [movie_to_index[movie_id] for user_id, movie_id in zip(tag_user_col, tag_movie_col) if user_id in user_to_index and movie_id in movie_to_index]

# Create edge indices for user-movie interactions from the filtered connections
edge_index = torch.tensor([filtered_user_indices, filtered_movie_indices], dtype=torch.long)

# Filter the rating tensor based on the filtered connections
filtered_rating_tensor = rating_tensor[filtered_user_indices]

# Set the number of nodes (users and movies) 
# Combine the filtered user and movie indices to get unique node indices
#unique_node_indices = set(filtered_user_indices + filtered_movie_indices)

num_nodes = len(filtered_user_indices) + len(filtered_movie_indices)

# Iterate through each edge and add its nodes to the set
positional_encodings = calculatePosEncodings_rswe(edge_index, num_nodes)

# Step 3: Read the u.item file to get the genre information
#TODO: mismatch: x ist kürzer als num_nodes. num_nodes ist korrekt, also stimmt der Teil nicht

genre_path = 'ml-20m/movies.csv'  # Set the path to the u.item file
genre_df = pd.read_csv(genre_path, delimiter=',', encoding='latin-1', header=None)
# Define genre mapping
genre_mapping = [
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
    'War', 'Western'
]

# Extract genre information and convert it to a list of lists
genre_cols = genre_df.iloc[:, 2].str.split('|').apply(lambda x: [1 if genre in x else 0 for genre in genre_mapping]).values.tolist()

# Step 4: Create genre node features for movies AND users. Initialize with 0 ("dummy features") and add the genre information! 
num_movies = num_nodes
num_genres = len(genre_mapping)
all_genre_features = torch.zeros(num_movies, num_genres)

for movie_id, genres in enumerate(genre_cols):
    for genre_id, genre_value in enumerate(genres):
        all_genre_features[movie_id][genre_id] = genre_value

'''
num_users = len(filtered_user_indices)
user_genre_features = torch.zeros(num_users, num_genres)

# Step 6: Combine movie and user genre features. 
all_genre_features = torch.cat((movie_genre_features, user_genre_features), dim=0)
'''
 
print(len(all_genre_features)) #ist irgendwie falsch!! num nodes ist = len(positional_encodings)
print(len(positional_encodings))


#positional_encodings= torch.rand(len(all_genre_features), 5) * 5

# Create the data object for the entire dataset. IMPORTANT: this is not according to the documentation, because y are edge features here! 
#tags = edge attributes,  ratings = target feature
data = Data(edge_index=edge_index, x=all_genre_features, y= rating_tensor, edge_attr = tag_tensor, num_nodes=num_nodes, positional_encodings = positional_encodings)
import random
# Assuming 'data' contains your graph data
edge_index = data.edge_index.numpy()

# Randomly select a subset of edges to visualize
subset_size = min(3000, edge_index.shape[1])  # Limit the subset size for performance reasons
subset_indices = random.sample(range(edge_index.shape[1]), subset_size)
subset_edges = edge_index[:, subset_indices]

# Create a NetworkX graph from the subset of edge indices
G_subset = nx.Graph()
G_subset.add_edges_from(subset_edges.T)

# Manually position nodes using Kamada-Kawai layout for the subset
pos_subset = nx.kamada_kawai_layout(G_subset)

# Visualize the subset of the graph using Matplotlib
plt.figure(figsize=(12, 8))
nx.draw(G_subset, pos_subset, node_size=30, node_color='skyblue', with_labels=False, font_size=10)
plt.title("Subset of Graph Visualization")
plt.show()

'''
import networkx as nx

edge_index = data.edge_index.numpy()

# Create a NetworkX graph from the edge indices
G = nx.Graph()
G.add_edges_from(edge_index.T)  # Get the node positions.
pos = nx.spring_layout(G)

# Draw the nodes.
nx.draw_networkx_nodes(G, pos=pos, node_size=10, node_color='blue')

# Draw the edges.
nx.draw_networkx_edges(G, pos=pos, edge_color='black')

# Show the plot.
plt.show()
'''

''' NOT WORKING: too slow!
# Assuming 'data' contains your graph data
edge_index = data.edge_index.numpy()

# Create a NetworkX graph from the edge indices
G = nx.Graph()
G.add_edges_from(edge_index.T)

# Manually position nodes using Kamada-Kawai layout
pos = nx.kamada_kawai_layout(G)

# Visualize the graph using Matplotlib
plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_size=50, node_color='skyblue', with_labels=False, font_size=10)
plt.title("Graph Visualization")
plt.show()
'''