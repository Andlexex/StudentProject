

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

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
import csv



def calculateDegrees(edge_index, num_nodes):
    edge_index = edge_index.t().tolist()
    edges = [(src, dst) for src, dst in edge_index]

    # Create the adjacency matrix in CSR format
    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    # Calculate the in-degrees for each node
    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    for edge in edges:
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i] for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    in_degrees = np.power(in_degrees, -0.5)  # Take the element-wise inverse square root

    # Create the sparse diagonal matrix N
    N = csr_matrix(sp.diags(in_degrees, dtype=float))

    # Calculate the degrees using N * A * N
    degrees = N.dot(A).dot(N).diagonal()

    # Normalize the degrees to be in the range [0, 1]
    max_degree = max(degrees)
    min_degree = min(degrees)
    normalized_degrees = (degrees - min_degree) / (max_degree - min_degree)

    # Convert the degrees to PyTorch tensor
    degrees_tensor = torch.from_numpy(normalized_degrees).float()

    return degrees_tensor

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
    false_negatives_normalized = 0

    for rating, prediction in top_k_items:
        if(prediction < 3.5 and rating >=3.5):
            false_negatives_normalized += 1

    normalized_recall_at_k = true_positives / (true_positives + false_negatives_normalized) #/ min(k, relevant_items)

    return precision_at_k, recall_at_k, normalized_recall_at_k


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
    print(np.sum(binary_predictions == 1))
    print("Positive: ")
    print(np.sum(binary_predictions == 0))
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

positional_encodings = calculatePosEncodings_rswe(edge_index, num_nodes)

'''  HIER: ADD NODE FEATURES (they are CORRECT!)'''
# Step 3: Read the u.item file to get the genre information
genre_path = 'u.item'  # Set the path to the u.item file
genre_df = pd.read_csv(genre_path, delimiter='|', encoding='latin-1', header=None)

# Extract the genre columns and convert them to a list of lists
genre_cols = genre_df.iloc[:, 5:].values.tolist()

# Step 4: Create genre node features for movies
num_movies = len(movie_to_index)
num_genres = 19  # Including the "unknown" genre
movie_genre_features = torch.zeros(num_movies, num_genres)

for movie_id, genres in enumerate(genre_cols):
    for genre_id, genre_value in enumerate(genres):
        if genre_value == 1:
            movie_genre_features[movie_id][genre_id] = 1

# Step 5: Create genre node features for users (users don't have genres, so we can create a dummy feature)
num_users = len(user_to_index)
user_genre_features = torch.zeros(num_users, num_genres)

# Step 6: Combine movie and user genre features. 
all_genre_features = torch.cat((movie_genre_features, user_genre_features), dim=0)

# Create the data object for the entire dataset. IMPORTANT: this is not according to the documentation, because y are edge features here! 
data = Data(edge_index=edge_index,   x=all_genre_features, y=rating_tensor, num_nodes=num_nodes, positional_encodings = positional_encodings)





# Step 4: Split the data into training and test sets
indices = list(range(data.edge_index.size(1)))

'''
for loop: Each model 10 times. to ensure that the same train-test-split is used, we do a train test split first, then run model 1, 2 and 3
After that, new train-test split, run model1, 2, 3 
and so on 
'''

# Now write the data to a CSV file
csv_filename = "results_GraphSAGE.csv"
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["i", "Model", "Precision", "Recall", "Precision@k", "Recall@k", "MSE"])  # Write header


for i in range(3):
    if (i % 3 == 0):
        train_indices, test_indices = train_test_split(indices, train_size=0.8, test_size=0.2)
        train_indices, val_indices = train_test_split(train_indices, train_size=0.8, test_size=0.2, random_state=42)
        np.savez('indices.npz', train_indices=train_indices, test_indices=test_indices, val_indices=val_indices)


    # Read the indices from the file
    loaded_indices = np.load('indices.npz')
    train_indices = loaded_indices['train_indices']
    test_indices = loaded_indices['test_indices']
    val_indices = loaded_indices['val_indices']

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


    test_data.edge_index = data.edge_index[:, test_indices]
    test_data.y = data.y[test_indices]
    test_data.num_nodes = data.num_nodes
    test_data.positional_encodings = data.positional_encodings
    test_data.x = data.x


    val_data.edge_index = data.edge_index[:, val_indices]
    val_data.y = data.y[val_indices]
    val_data.num_nodes = data.num_nodes
    val_data.positional_encodings = data.positional_encodings
    val_data.x = data.x




    # Define the GraphSAGE model
    class GraphSAGE_nopos(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GraphSAGE_nopos, self).__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')


        def forward(self, x, edge_index, positional_encodings):
            #fh from the paper
            x = x.view(-1, x.size(2))

            #fh from the paper
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)

            movie_embed = x[edge_index[1]]
            
            ratings = torch.sum(movie_embed, dim=1)
            return ratings


    class GraphSAGE_var1(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.conv1 = SAGEConv(24, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

            #this is for learning of the positional encodings, which is seperate!!!
            self.conv1_pos = SAGEConv(5, hidden_channels, aggr='mean')
            self.conv2_pos = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.conv3_pos = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

            self.linear = nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index, pos_embeddings):
            x = x.view(-1, x.size(2))
            pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))
            x = torch.cat([x, pos_embeddings], dim=1)

            #fh from the paper
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            #x = self.conv3(x, edge_index)

            #Now the learning of positional embeddings. So this is fp from the paper
            pos_embeddings = self.conv1_pos(pos_embeddings, edge_index)
            pos_embeddings = F.relu(pos_embeddings)
            pos_embeddings = self.conv2_pos(pos_embeddings, edge_index)
            pos_embeddings = F.relu(pos_embeddings)
            #pos_embeddings = self.conv3_pos(pos_embeddings, edge_index)

            final_output = self.linear(torch.cat([x, pos_embeddings]))

            movie_embed = final_output[edge_index[1]]            
            ratings = torch.sum(movie_embed, dim=1)
            return ratings
        
    class GraphSAGE_var2(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            # number of in layers = number of node features + number of positional embedding dimensions

            self.conv1 = SAGEConv(24, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.conv2_var2 = SAGEConv(hidden_channels*2, hidden_channels, aggr='mean')
            self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

            #this is for learning of the positional encodings, which is seperate!!!
            self.conv1_pos = SAGEConv(5, hidden_channels, aggr='mean')
            self.conv2_pos = SAGEConv(hidden_channels, hidden_channels,aggr='mean')
            self.conv3_pos = SAGEConv(hidden_channels, hidden_channels,  aggr='mean')

            self.linear = nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index, pos_embeddings):
            x = x.view(-1, x.size(2))
            pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))
            x = torch.cat([x, pos_embeddings], dim=1)

            #fh from the paper
            x = self.conv1(x, edge_index)
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


    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model, loss function, and optimizer
    in_channels = all_genre_features.size(1)  # Number of input features
    hidden_channels = 128
    out_channels = 1  # Number of output classes (assuming regression task)
    batch_size = 64
    lr = 0.01
    epochs = 200
    #Early Stopping
    patience = 30  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum improvement required to consider as improvement

    best_val_loss = np.inf
    best_epoch = 0
    early_stop_counter = 0

    if (i%3 == 0):
        model = GraphSAGE_nopos(in_channels, hidden_channels, out_channels).to(device)
    elif(i%3 == 1):
        model = GraphSAGE_var1(hidden_channels).to(device)
    else:
        model = GraphSAGE_var2(hidden_channels).to(device)

    model_name = 'GraphSAGE'
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data], batch_size=batch_size)
    test_loader = DataLoader([data], batch_size=batch_size)

    # Model training
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model( batch.x.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1))
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model( batch.x.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1))
                loss = criterion(output, batch.y)
            val_loss += loss.item() * batch.num_graphs
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
        
        # Early stopping
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f}')
                break

        #print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []

        for batch in test_loader:
            batch = batch.to(device)
            output = model( batch.x.unsqueeze(1), batch.edge_index, batch.positional_encodings.unsqueeze(1))
            loss = criterion(output, batch.y)
            #test_loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)
            
            #print(f'Test Loss: {test_loss.item()}')
            predictions.extend(output.cpu().numpy().flatten())
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


        precision_value = precision(predictions, targets, threshold)
        recall_value = recall(predictions, targets, threshold)


        print(f"Precision: {precision_value}")
        print(f"Recall: {recall_value}")
        precission_k, recall_k, normalized_recall_k = precission_recall_at_k(predictions, targets, 4, 1000)

        print(f"Precision_at_k: {precission_k}")
        print(f"Recall_at_k: {recall_k}")
        
        #Now write the file! 
        if (i % 3== 0):
            model_name = "nopos"
        elif (i %3 == 1):
            model_name = "GS_var1"
        elif (i%3 == 2):
            model_name = "GS_var2"
        
        with open("results_GraphSAGE.csv", mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([i, model_name, precision_value, recall_value, precission_k, recall_k, mse])
