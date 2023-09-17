import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Define the graph structure (edge indices) and node features
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
node_features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
edge_features = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float)  # Example edge features

# Create a torch_geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Initialize the GCN model
in_channels = node_features.shape[1]  # Number of input node features
edge_in_channels = edge_features.shape[1]  # Number of input edge features
hidden_channels = 16  # Number of hidden channels in the GCN layers
out_channels = 8  # Number of output channels (desired node feature dimensions)
model = GCNModel(in_channels, edge_in_channels, hidden_channels, out_channels)

# Run the model on the graph data
output = model(data.x, data.edge_index, data.edge_attr)

# Print the output after the GCN layers
print("GCN Output:")
print(output)