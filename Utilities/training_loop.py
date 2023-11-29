import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from Utilities.dataset import ArteryGraphDataset
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Define your GNN model
class GNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, data):
        print(data.x.shape)
        quit()
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pool(x.unsqueeze(0).permute(2, 0, 1)).squeeze()
        x = self.softmax(x)
        x = x.unsqueeze(0)
        return x

if __name__ == "__main__":
    VERBOSE = True
    # Set up the dataset
    dataset = ArteryGraphDataset(root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_DATA/', ann_file='graphs_annotation.json')
    # Split the dataset into training and test sets with 80-20 splitting
    #TODO: use separated and controlled samples for testing and training, two different datasets, not splitted from the same one
    # train_dataset = dataset[:round(len(dataset) * 0.8)]
    # test_dataset = dataset[round(len(dataset) * 0.8):]
    train_dataset = dataset[:70]
    test_dataset = dataset[70:79]

    # Create data loaders
    #TODO: problems with batch size, if >1 it doesn't work
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if VERBOSE:
        print(f"Dataset length: {len(dataset)}")
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")

    # Initialize the model
    model = GNN(dataset.num_features, dataset.num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            print(output)
            print(data.y)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                output = model(data)
                _, predicted = torch.max(output, dim=1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()

        accuracy = 100 * correct / total
        if VERBOSE:
            print(f'Epoch {epoch+1}: Test Accuracy = {accuracy:.2f}%')
            print(f'Loss: {loss.item():.4f}')
