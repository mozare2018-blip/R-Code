import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Step 1: Load your dataset
df = pd.read_csv('D:/python/Return.csv')  # 11 columns, last column is target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Step 2: Create KNN graph (K=5 neighbors)
adj = kneighbors_graph(X, n_neighbors=5, mode='connectivity', include_self=True)
edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)

# Step 3: Prepare PyG data object
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, y=y)

# Step 4: Define GNN regressor
class GNNRegressor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x.view(-1)  # Flatten to 1D for regression

# Step 5: Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GNNRegressor(in_channels=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    loss = loss_fn(pred, data.y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        mse = mean_squared_error(data.y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        print(f'Epoch {epoch}, Training MSE: {mse:.4f}')

# Step 6: Final evaluation
model.eval()
predicted = model(data).detach().cpu().numpy()
true_y = data.y.cpu().numpy()
final_mse = mean_squared_error(true_y, predicted)

print(f'\nFinal MSE on entire dataset: {final_mse:.4f}')




















