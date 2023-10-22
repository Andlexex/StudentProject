import matplotlib.pyplot as plt
import pandas as pd
import gzip
import gensim
from scipy.sparse import csr_matrix
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils as utils
import os
import sys
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

    in_degrees = np.array([in_degrees_dict[i]
                          for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    # Take the element-wise inverse square root
    in_degrees = np.power(in_degrees, -1)

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
    PE = torch.stack(PE, dim=-1)

    # ERGEBNIS
    RESULT_POS_ENCODING = PE
    return RESULT_POS_ENCODING


def calculatePosEncodings(edge_index, num_nodes):
    print("checkpoint1")
    edge_index = edge_index.t().tolist()
    edges = [(src, dst) for src, dst in edge_index]

    # Create the adjacency matrix in CSR format -> das wird dann für die encodings benutzt!
    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    ''' this code computes the in_degrees matrix from the edge list. it can later be adapted to compute the in-degrees matrix from the adjacency matrix (however, then, we should
    do some tests with small sample graphs to ensure everytheing is correct
    '''
    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    # Calculate the in-degrees for each node
    for edge in edges:
        print("checkpoint2")
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i]
                          for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    # Take the element-wise inverse square root
    in_degrees = np.power(in_degrees, -0.5)
    print("checkpoint3")
    # Create the sparse diagonal matrix N
    N = sp.diags(in_degrees, dtype=float)

    L = sp.eye(num_nodes) - N * A * N

    # calc eigvals and eigVecs, equivalent to the original code
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # pos_enc_dim = hyperparameter!
    pos_enc_dim = 5
    RESULT_POS_ENCODING = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    return RESULT_POS_ENCODING


def calculateLoss(task_loss, batch, num_nodes, positional_encoding):
    # HYPERPARAMETERS
    device = "cpu"
    pos_enc_dim = 1
    alpha_loss: 1e-3
    lambda_loss: 100  # ist auch 100

    # edge_index im korrekten Format definieren
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

    in_degrees = np.array([in_degrees_dict[i]
                          for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    # Take the element-wise inverse square root
    in_degrees = np.power(in_degrees, -0.5)

    # Create the sparse diagonal matrix N
    N = sp.diags(in_degrees, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    p = positional_encoding
    pT = torch.transpose(p, 1, 0)
    loss_b_1 = torch.trace(
        torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(device)), p))

    '''  TODO: loss_b_2 
    '''

    loss_b = loss_b_1

    # TODO: parameter tunen!
    loss = task_loss + 1e-3 * loss_b
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


def precission_recall_at_k(predictions, targets, threshold, k):
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


def get_review_vector(tokens, model):
    word_vectors = model.wv
    vectors = []

    for token in tokens:
        if token in word_vectors:
            word_vector = word_vectors[token]
            vectors.append(word_vector)
        else:
            vectors.append(np.zeros(100))

    if not vectors:
        return np.zeros(100)

    average_vector = np.mean(vectors, axis=0)
    return average_vector


def generateModel(df):
    print("generate model")
    current_directory = os.getcwd()
    model_path = os.path.join(
        current_directory, "product_reviews_word2vec.model")
    if not os.path.exists(model_path):
        # check if the reviewText type is not string and convert them

        model = Word2Vec(
            vector_size=100,    # Dimensionality of the word vectors
            window=5,           # Maximum distance between the current and predicted word
            min_count=1,        # Minimum number of times a word must appear to be considered
            # Number of CPU cores to use for training (adjust based on your system)
            workers=4
        )
        print("model is generated and trained")
        model.build_vocab(df["tokenized_review"])
        model.train(df["tokenized_review"],
                    total_examples=model.corpus_count, epochs=30)

        print("save model")
        model.save(model_path)
        return model_path
    else:
        print("model is already generated and trained")
        return model_path


# process the data from the dataset
# read the json and prepare the required data

parent_directory = os.path.dirname(os.getcwd())
data_path = os.path.join(
    parent_directory, "AMAZON_FASHION.json.gz")
df = pd.read_json(data_path, lines=True, orient="records")

columns = ["reviewerID", "asin", "overall", "reviewText"]
df = df[columns]

df["reviewText"] = df["reviewText"].apply(
    lambda x: str(x) if not isinstance(x, str) else x)

df["tokenized_review"] = df["reviewText"].apply(
    lambda x: word_tokenize(x.lower()))

tokenized_reviews = df["tokenized_review"].tolist()

# get unique indicies
unique_reviewers = df["reviewerID"].unique()
unique_asins = df["asin"].unique()

num_nodes = len(unique_reviewers) + len(unique_asins)

user_to_id = {user: idx for idx, user in enumerate(unique_reviewers)}
asin_to_id = {asin: idx for idx, asin in enumerate(unique_asins)}

df["reviewerID_id"] = df["reviewerID"].map(user_to_id)
df["asin_id"] = df["asin"].map(asin_to_id)

# generate the model and load it
model_path = generateModel(df)
model_word2Vec = Word2Vec.load(model_path)

# dummy padding for the reviewer
# compute word vectors for the products
processed_reviews = []
# compute for each product the average word vector
print("compute the word vectors")
for asin_id, group in df.groupby("asin_id")["tokenized_review"]:
    vector_for_product = []

    for review_tokens in group:
        if isinstance(review_tokens, list) and review_tokens:
            vector = get_review_vector(review_tokens, model_word2Vec)
            vector_for_product.append(vector)
        else:
            vector_for_product.append(np.zeros(100))

    if vector_for_product:
        average_vector = np.mean(vector_for_product, axis=0).tolist()
        processed_reviews.append(
            {"asin_id": asin_id, "product_vector": average_vector})
    else:
        processed_reviews.append(
            {"asin_id": asin_id, "product_vector": np.zeros(100)})

product_reviews = pd.DataFrame(processed_reviews)
product_vector = product_reviews['product_vector']
num_features = len(product_vector[0])

edges = df[["reviewerID_id", "asin_id"]].values.T
edge_attr = df["overall"].values

reviewerID_features = torch.zeros(len(unique_reviewers), num_features)
product_feature = torch.tensor(product_vector, dtype=torch.float)

features = torch.cat((product_feature, reviewerID_features), dim=0)

del product_reviews

print(features)

# Extract user and product nodes
user_product = torch.tensor(
    df[["reviewerID_id", "asin_id"]].values, dtype=torch.float)

# Extract edge indices
edge_index = torch.tensor(edges, dtype=torch.long)

# num nodes 30.000 oder num_nodes?
positional_encodings = calculatePosEncodings_rswe(edge_index, num_nodes)

# Extract edge attributes
rating_tensor = torch.tensor(edge_attr, dtype=torch.float)

# Create the Data object with node features

data = Data(edge_index=edge_index, x=features, y=rating_tensor,
            positional_encodings=positional_encodings)
print("finish with the preprocessing")


class GINModel_nopos(nn.Module):
    def __init__(self, hidden_channels):
        super(GINModel_nopos, self).__init__()
        # number of in layers = number of node features + number of positional embedding dimensions
        num_features = 100
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            # TODO: diesen Parameter mal tunen!
            nn.Linear(hidden_channels, hidden_channels)
        ))

        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # TODO: diesen Parameter mal tunen!
        ))

    def forward(self, x, edge_index, pos_embeddings):
        x = x.view(-1, x.size(2))

        # fh from the paper
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index)
      #  x = F.relu(x)
      #  x = self.conv3(x, edge_index)

        # Predict movie ratings (edge features) using a linear layer
        reviewer_embed = x[edge_index[0]]
        product_embed = x[edge_index[1]]

    # ratings = torch.sum(user_embed * movie_embed, dim=1)
        ratings = torch.sum(product_embed, dim=1)
        return ratings


class GINModel_var1(nn.Module):
    def __init__(self, hidden_channels):
        super(GINModel_var1, self).__init__()
        # number of in layers = number of node features + number of positional embedding dimensions
        num_features = 105
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))

        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))

        # this is for learning of the positional encodings, which is seperate!!!
        self.conv1_pos = GINConv(nn.Sequential(
            nn.Linear(5, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv2_pos = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv3_pos = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, pos_embeddings):
        x = x.view(-1, x.size(2))
        pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))
        x = torch.cat([x, pos_embeddings], dim=1)

        # fh from the paper
        x = self.conv1(x, edge_index)
       # x = F.relu(x)

       # x = self.conv2(x, edge_index)
       # x = F.relu(x)
       # x = self.conv3(x, edge_index)

        # Now the learning of positional embeddings. So this is fp from the paper
        pos_embeddings = self.conv1_pos(pos_embeddings, edge_index)
       # pos_embeddings = F.relu(pos_embeddings)
       # pos_embeddings = self.conv2_pos(pos_embeddings, edge_index)
       # pos_embeddings = F.relu(pos_embeddings)
       # pos_embeddings = self.conv3_pos(pos_embeddings, edge_index)

        final_output = self.linear(torch.cat([x, pos_embeddings]))
        product_embed = final_output[edge_index[1]]
        ratings = torch.sum(product_embed, dim=1)

        return ratings


'''____________________________________________________________________________________'''


class GINModel_var2(nn.Module):
    def __init__(self, hidden_channels):
        super(GINModel_var2, self).__init__()
        # number of in layers = number of node features + number of positional embedding dimensions
        num_features = 100
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            # nn.ReLU(),
            # nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2_var2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels*2),
            #     nn.ReLU(),
            #    nn.Linear(hidden_channels, hidden_channels)
        ))

        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))

        # this is for learning of the positional encodings, which is seperate!!!
        self.conv1_pos = GINConv(nn.Sequential(
            nn.Linear(5, hidden_channels),
            #   nn.ReLU(),
            #  nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv2_pos = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv3_pos = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, pos_embeddings):
        x = x.view(-1, x.size(2))
        pos_embeddings = pos_embeddings.view(-1, pos_embeddings.size(2))
       # x = torch.cat([x, pos_embeddings], dim=1)

        # fh from the paper
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        pos_embeddings = self.conv1_pos(pos_embeddings, edge_index)
        pos_embeddings = F.relu(pos_embeddings)

        x = self.conv2_var2(torch.cat([x, pos_embeddings], dim=1), edge_index)
       # x = F.relu(x)
        # x = self.conv3(x, edge_index)

        product_embed = x[edge_index[1]]
        ratings = torch.sum(product_embed, dim=1)

        return ratings


data = torch.load("../bin/graph.pth")
indices = list(range(data.edge_index.size(1)))


csv_filename = "results.csv"
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["i", "Model", "Precision", "Recall",
                        "Precision@k", "Recall@k", "MSE"])  # Write header


for i in range(30):
    # das hier klein, damit der Speicher nicht überdreht wird. Aber nicht zu klein, weil sonst kommt es zu problemen!
    if (i % 3 == 0):
        train_indices, test_indices = train_test_split(
            indices, train_size=0.8, test_size=0.2)
        train_indices, val_indices = train_test_split(
            train_indices, train_size=0.8, test_size=0.2, random_state=42)
        np.savez('indices.npz', train_indices=train_indices,
                 test_indices=test_indices, val_indices=val_indices)
    # Now, you can comment out the above code that generates the indices
    # Read the indices from the file
    loaded_indices = np.load('indices.npz')
    # irgendeine syntax
    train_data = data.__class__()
    test_data = data.__class__()
    val_data = data.__class__()

    # setzt die Parameter von train_data und test_data

    # soweit ich es verstehe, sind alle 2.500 nodes im training und testset vorhanden. gesplittet werden nur die edges, d.h.
    # es ist nur ein subset der 100.000 edges im training set sowie im test set vorhanden
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

    # Step 6: Train and evaluate the GCN model
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set the device --> aktiviere GPU falls vorhanden
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ------------------------------------------------------

    # hidden channels und epochs tunen
    hidden_channels = 16  # 8 und 16
    lr = 0.01  # 0.01 vs 0.001
    epochs = 200  # 100 vs 200
    batch_size = 512  # 512

    # 1, 16, 32 ,64, 128, 256, 512

    # Early Stopping
    patience = 40  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum improvement required to consider as improvement

    best_val_loss = np.inf
    best_epoch = 0
    early_stop_counter = 0

    # this is for evaluation!
    if (i % 3 == 0):
        model = GINModel_nopos(hidden_channels=hidden_channels)
    elif (i % 3 == 1):
        model = GINModel_var1(hidden_channels=hidden_channels)
    elif (i % 3 == 2):
        model = GINModel_var2(hidden_channels=hidden_channels)

    model = model.to(device)  # Move the model to the selected device

    # ------------------------------------------------------
    # loss function, and optimizer, MSE = Metrik für Loss
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
    predictions = []

    import matplotlib.pyplot as plt

    for epoch in range(epochs):
        # Training
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch.x.unsqueeze(1), batch.edge_index,
                        batch.positional_encodings.unsqueeze(1))
            task_loss = criterion(out, batch.y)
            loss = task_loss
            # loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            predictions = out.detach().cpu().numpy()
            # print(predictions)

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

    # Validation
        model.eval()
        val_loss = 0.0
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x.unsqueeze(1), batch.edge_index,
                        batch.positional_encodings.unsqueeze(1))
            task_loss = criterion(out, batch.y)
            loss = task_loss
        # loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)

            val_loss += loss.item() * batch.num_graphs

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Print training and validation loss for monitoring
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

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
            out = model(batch.x.unsqueeze(1), batch.edge_index,
                        batch.positional_encodings.unsqueeze(1))
            task_loss = criterion(out, batch.y)
            test_loss = task_loss
            # test_loss = calculateLoss(task_loss, batch, num_nodes, batch.positional_encodings)

            # print(f'Test Loss: {test_loss.item()}')
            predictions.extend(out.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Round the predicted ratings
        rounded_predictions = np.round(predictions, decimals=0)

        # Plotting the distribution
        plt.hist(rounded_predictions, bins=15, edgecolor='black')
        plt.xlabel('Predicted Rating')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Ratings')
        plt.xticks(range(1, 16))
        # plt.show()
        plt.savefig('predicted_rankings.png')

        mse = np.mean(np.abs(predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        k = 5  # Define the value of k

        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")

        threshold = 3.5  # Define the threshold to convert predictions into binary values
        # Set the threshold to print the whole array
        np.set_printoptions(threshold=sys.maxsize)

        # print(rounded_predictions)
        # print(predictions)

        GAT_results = open("predictions_GAT.txt", "a")

        GAT_results.write(str(predictions))

        GAT_results.close()

        precision_value = precision(predictions, targets, threshold)
        recall_value = recall(predictions, targets, threshold)
        precission_k, recall_k, normalized_recall_k = precission_recall_at_k(
            predictions, targets, 4, 1000)

        with open("predictions.txt", 'w') as file:
            for prediction in predictions:
                file.write(str(prediction) + '\n')

        print(f"Precision: {precision_value}")
        print(f"Recall: {recall_value}")

        print(f"Precision@k: {precission_k}")
        print(f"Recall@k: {recall_k}")
        # print(f"Normalized Recall@k: {normalized_recall_k}")

        # Now write the file!
        if (i % 3 == 0):
            model_name = "GCN_nopos"
        elif (i % 3 == 1):
            model_name = "GCN_var1"
        elif (i % 3 == 2):
            model_name = "GCN_variant2"

        print(i)
        with open("results.csv", mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [i, model_name, precision_value, recall_value, precission_k, recall_k, mse])
