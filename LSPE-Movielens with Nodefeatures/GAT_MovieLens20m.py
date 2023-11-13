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
    pos_enc_dim = 5

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
    pos_enc_dim = 5
    RESULT_POS_ENCODING = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    return RESULT_POS_ENCODING

def calculateLoss(task_loss, batch, num_nodes, positional_encoding):
    #HYPERPARAMETERS
    device = "cpu"
    pos_enc_dim = 1
    alpha_loss: 1e-3
    lambda_loss: 100  # ist auch 100

    #edge_index im korrekten Format definieren
    edge_index = batch.edge_index.t().tolist()
    edge_index = [(src, dst) for src, dst in edge_index]

    # Loss B: Laplacian Eigenvector Loss --------------------------------------------
    n = num_nodes

    # Laplacian 
    rows, cols = zip(*edge_index)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    ''' this code computes the in_degrees matrix from the edge list. it can later be adapted to compute the in-degrees matrix from the adjacency matrix (however, then, we should
    do some tests with small sample graphs to ensure everything is correct'''

    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    # Calculate the in-degrees for each node
    for edge in edge_index:
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i] for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    in_degrees = np.power(in_degrees, -0.5)  # Take the element-wise inverse square root

    # Create the sparse diagonal matrix N
    N = sp.diags(in_degrees, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    p = positional_encoding
    pT = torch.transpose(p, 1, 0)
    loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(device)), p))

    loss_b = loss_b_1

    loss = task_loss + 1e-3* loss_b
    return loss


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

def precission_recall_at_k (predictions, targets, threshold, k):
    # Combine ratings and predictions into tuples for sorting
    combined = list(zip(targets, predictions))

    # Sort the combined list in descending order of predictions
    combined.sort(key=lambda x: x[1], reverse=True)

    # Extract top k sorted items and calculate precision and recall
    top_k_items = combined[:k]
    true_positives = sum(1 for rating, _ in top_k_items if rating >= threshold)
    false_positives = k - true_positives
    relevant_items = sum(1 for rating in targets if rating >= threshold)
    false_negatives = relevant_items - true_positives

    precision_at_k = true_positives / (true_positives + false_positives)
    recall_at_k = true_positives / (true_positives + false_negatives)
    normalized_recall_at_k = recall_at_k / (k / relevant_items)


    return precision_at_k, recall_at_k, normalized_recall_at_k

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
 



#positional_encodings= torch.rand(len(all_genre_features), 5) * 5

# Create the data object for the entire dataset. IMPORTANT: this is not according to the documentation, because y are edge features here! 
#tags = edge attributes,  ratings = target feature
data = Data(edge_index=edge_index, x=all_genre_features, y= rating_tensor, edge_attr = tag_tensor, num_nodes=num_nodes, positional_encodings = positional_encodings)



# Step 4: Split the data into training and test sets
indices = list(range(data.edge_index.size(1)))

csv_filename = "results.csv"
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["i", "Model", "Precision", "Recall", "Precision@k", "Recall@k", "MSE"])  # Write header

for i in range(30):
    if (i % 3 == 0):
        train_indices, test_indices = train_test_split(indices, train_size=0.8, test_size=0.2)
        train_indices, val_indices = train_test_split(train_indices, train_size=0.8, test_size=0.2, random_state=42)
        np.savez('indices.npz', train_indices=train_indices, test_indices=test_indices, val_indices=val_indices)

    loaded_indices = np.load('indices.npz')
    train_indices = loaded_indices['train_indices']
    test_indices = loaded_indices['test_indices']
    val_indices = loaded_indices['val_indices']

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
    train_data.x = data.x
    train_data.edge_attr = data.edge_attr[train_indices]

    test_data.edge_index = data.edge_index[:, test_indices]
    test_data.y = data.y[test_indices]
    test_data.num_nodes = data.num_nodes
    test_data.positional_encodings = data.positional_encodings
    test_data.x = data.x
    test_data.edge_attr = data.edge_attr[test_indices]


    val_data.edge_index = data.edge_index[:, val_indices]
    val_data.y = data.y[val_indices]
    val_data.num_nodes = data.num_nodes
    val_data.positional_encodings = data.positional_encodings
    val_data.x = data.x
    val_data.edge_attr = data.edge_attr[val_indices]



    class GATModel_nopos(nn.Module):
        def __init__(self, hidden_channels):
            super(GATModel_nopos, self).__init__()

            # number of in layers = number of node features + number of positional embedding dimensions
            num_features = 24
            self.conv1 = GATv2Conv(num_features, hidden_channels, 1, edge_dim=1)
            self.conv1_nopos = GATv2Conv(19, hidden_channels, 1, edge_dim=1)

            self.conv2 = GATv2Conv(hidden_channels, hidden_channels, 1)
            self.conv2_var2 = GATv2Conv(hidden_channels*2, hidden_channels, 1)
            self.conv3 = GATv2Conv(hidden_channels, 1, 1)

            #this is for learning of the positional encodings, which is seperate!!!
            self.conv1_pos = GATv2Conv(5, hidden_channels, 1)
            self.conv2_pos = GATv2Conv(hidden_channels, 1, 1)

        def forward(self, x, edge_index, pos_embeddings, edge_attr):
            x = x.view(-1, x.size(2))
            #pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))        
            #x = torch.cat([x, pos_embeddings], dim=1)

            #fh from the paper
            #eigentlich müsste man die edge features direkt reinfüttern können: so         x = self.conv1(x, edge_index,edge_attr=edge_attr)
            x = self.conv1_nopos(x, edge_index, edge_attr = edge_attr) #, # edge_attr = edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

            #these are the user features (right now: none)
            user_embed = x[edge_index[0]]
            #these are the movie features
            movie_embed = x[edge_index[1]]

            #ratings = torch.sum(user_embed * movie_embed, dim=1)
            ratings = torch.sum(movie_embed, dim=1)
            return ratings


    class GATModel_variant1(nn.Module):
        def __init__(self, hidden_channels):
            super(GATModel_variant1, self).__init__()

            # number of in layers = number of node features + number of positional embedding dimensions
            num_features = 24
            self.conv1 = GATv2Conv(num_features, hidden_channels, 1, edge_dim=1)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels, 1)
            self.conv3 = GATv2Conv(hidden_channels, hidden_channels, 1)

            #this is for learning of the positional encodings, which is seperate!!!
            self.conv1_pos = GATv2Conv(5, hidden_channels, 1)
            self.conv2_pos = GATv2Conv(hidden_channels, hidden_channels, 1)
            self.conv3_pos = GATv2Conv(hidden_channels, hidden_channels,1)
            self.linear = nn.Linear(hidden_channels, 1)
            

        def forward(self, x, edge_index, pos_embeddings, edge_attr):
            x = x.view(-1, x.size(2))
            pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))      
            x = torch.cat([x, pos_embeddings], dim=1)

            #fh from the paper
            #eigentlich müsste man die edge features direkt reinfüttern können: so         x = self.conv1(x, edge_index,edge_attr=edge_attr)
            x = self.conv1(x, edge_index, edge_attr = edge_attr) #, # edge_attr = edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)

            #Now the learning of positional embeddings. So this is fp from the paper
            pos_embeddings = self.conv1_pos(pos_embeddings, edge_index)
            pos_embeddings = F.relu(pos_embeddings)
            pos_embeddings = self.conv2_pos(pos_embeddings, edge_index)
            pos_embeddings = F.relu(pos_embeddings)
            pos_embeddings = self.conv3_pos(pos_embeddings, edge_index)

            final_output = self.linear(torch.cat([x, pos_embeddings]))
            #these are the movie features
            movie_embed = final_output[edge_index[1]]            

            #ratings = torch.sum(user_embed * movie_embed, dim=1)
            ratings = torch.sum(movie_embed, dim=1)
            return ratings



    class GATModel_variant2(nn.Module):
        def __init__(self, hidden_channels):
            super(GATModel_variant2, self).__init__()

            # number of in layers = number of node features + number of positional embedding dimensions
            num_features = 24
            self.conv1 = GATv2Conv(num_features, hidden_channels, 1, edge_dim=1)
            self.conv1_nopos = GATv2Conv(19, hidden_channels, 1, edge_dim=1)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels, 1)
            self.conv2_var2 = GATv2Conv(hidden_channels*2, hidden_channels, 1)
            self.conv3 = GATv2Conv(hidden_channels, hidden_channels, 1)

            #this is for learning of the positional encodings, which is seperate!!!
            self.conv1_pos = GATv2Conv(5, hidden_channels, 1)
            self.conv2_pos = GATv2Conv(hidden_channels, hidden_channels, 1)
            self.conv3_pos = GATv2Conv(hidden_channels, hidden_channels, 1)

            

        def forward(self, x, edge_index, pos_embeddings, edge_attr):
            x = x.view(-1, x.size(2))
            pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))        
            x = torch.cat([x, pos_embeddings], dim=1)

            #fh from the paper
            #eigentlich müsste man die edge features direkt reinfüttern können: so         x = self.conv1(x, edge_index,edge_attr=edge_attr)
            x = self.conv1(x, edge_index,  edge_attr = edge_attr)
            x = F.relu(x)
            pos_embeddings = self.conv1_pos(pos_embeddings, edge_index)
            pos_embeddings = F.relu(pos_embeddings)
            #x = self.conv2(torch.cat([x,pos_embeddings],dim=1), edge_index)
            x = self.conv2_var2(torch.cat([x,pos_embeddings],dim=1), edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)

            movie_embed = x[edge_index[1]]
            ratings = torch.sum(movie_embed, dim=1)

            return ratings





    # Step 6: Train and evaluate the GCN model
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set the device --> aktiviere GPU falls vorhanden
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #------------------------------------------------------

    #hidden channels und epochs tunen
    hidden_channels= 256 #war 256
    lr = 0.01  #0.01 vs 0.001 
    epochs = 100  #100 vs 200
    batch_size = 512#512

    #1, 16, 32 ,64, 128, 256, 512

    #Early Stopping
    patience = 30  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum improvement required to consider as improvement

    best_val_loss = np.inf
    best_epoch = 0
    early_stop_counter = 0

    # Define the GCNModel
    #model = GATModel(num_features = 1, hidden_channels = hidden_channels, num_classes = 1, heads = 1).to(device)
    #model = GINModel(num_features = 1, num_classes = 1).to(device)


    if (i % 3 == 0):
        model = GATModel_nopos(hidden_channels=hidden_channels)
    elif (i % 3 == 1):
        model = GATModel_variant1(hidden_channels=hidden_channels)
    elif (i%3 == 2):
        model = GATModel_variant2(hidden_channels=hidden_channels)

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
    predictions =[]

    import matplotlib.pyplot as plt

    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)        
            out = model( batch.x.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1), batch.edge_attr.unsqueeze(1))
            task_loss = criterion(out, batch.y)
            loss = task_loss
            #loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)

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
            out = model(batch.x.unsqueeze(1),batch.edge_index, batch.positional_encodings.unsqueeze(1), batch.edge_attr.unsqueeze(1))
            task_loss = criterion(out, batch.y)
            loss = task_loss
            #loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)
            
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
    '''
    # Plotting training and validation curves
    plt.plot(train_losses, label='Training Loss')
    #plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot as an image file
    plt.savefig('loss_plot.png')
    '''

    # Show the plot

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []

        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1), batch.y.unsqueeze(1))
            task_loss = criterion(out, batch.y)
            test_loss = task_loss
            #test_loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)
            
            #print(f'Test Loss: {test_loss.item()}')
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
        k = 10  # Define the value of k

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

        precission_k, recall_k, normalized_recall_k = precission_recall_at_k(predictions, targets, 4, 50)


        with open("predictions.txt", 'w') as file:
            for prediction in predictions:
                file.write(str(prediction) + '\n')

        print(f"Precision: {precision_value}")
        print(f"Recall: {recall_value}")

        print(f"Precision@k: {precission_k}")
        print(f"Recall@k: {recall_k}")

        #print(f"Normalized Recall@k: {normalized_recall_k}")

        #Now write the file! 
        if (i % 3 == 0):
            model_name = "GAT_nopos"
        elif (i % 3 == 1):
            model_name = "GAT_var1"
        elif (i%3 == 2):
            model_name = "GAT_var2"

        with open(csv_filename, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([i, model_name, precision_value, recall_value, precission_k, recall_k, mse])




