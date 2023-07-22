import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

user_index_col= []
product_index_col=[]
rating_tensor=torch.Tensor([])
num_user_index=0
num_product_index=0

def preprocess(df):
    global rating_tensor, counter, num_user_index, num_product_index
    print("i do my job"+ str(counter))
    counter= counter+1
    user_col = df['user_id'].values
    product_col = df['product_id'].values
    rating_col = df['rating'].values
   
    user_to_index = {user_id: index for index, user_id in enumerate(set(user_col))}
    product_to_index = {product_id: index for index, product_id in enumerate(set(product_col))}
    
    num_user_index=len(user_to_index)
    num_product_index=len(product_to_index)

    user_index_col_chunk = [user_to_index[user_id] for user_id in user_col]
    product_index_col_chunk = [product_to_index[product_id] for product_id in product_col]

    user_index_col.extend(user_index_col_chunk)
    product_index_col.extend(product_index_col_chunk)

    rating_tensor_chunk = torch.tensor(rating_col, dtype=torch.float)
    rating_tensor = torch.cat([rating_tensor, rating_tensor_chunk])

data_path='all_csv_files.csv'
# use at least 7 for the chunksize
chunksize = 10 ** 7
# use this to avoid memory problems
with pd.read_csv(data_path, chunksize=chunksize, header=None, usecols=[0,1,2]) as reader:
    for chunk in reader:
        chunk.columns =["user_id","product_id","rating"]
        preprocess(chunk)
print ("i finish my reading")
edge_index = torch.tensor([user_index_col, product_index_col], dtype=torch.long)
num_nodes=num_product_index+num_product_index
data = Data(edge_index=edge_index, y=rating_tensor, num_nodes=num_nodes)
indices = list(range(data.edge_index.size(1)))
