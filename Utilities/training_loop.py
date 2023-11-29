import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from Utilities.dataset import ArteryGraphDataset
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Define your GNN model
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='graph'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.leaky_relu(x) #to avoid dead ReLU
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        # if model.task == 'node':
        #     mask = data.val_mask if is_validation else data.test_mask
        #     # node classification: only evaluate on nodes in test set
        #     pred = pred[mask]
        #     label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total


if __name__ == "__main__":
    VERBOSE = True
    # Set up the dataset
    dataset = ArteryGraphDataset(root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_DATA/', ann_file='graphs_annotation.json')
    # Split the dataset into training and test sets with 80-20 splitting
    #TODO: use separated and controlled samples for testing and training, two different datasets, not splitted from the same one
    # train_dataset = dataset[:round(len(dataset) * 0.8)]
    # test_dataset = dataset[round(len(dataset) * 0.8):]
    train_dataset = dataset[:90]
    test_dataset = dataset[90:] #try to overfit on a small dataset

    # Create data loaders
    #TODO: problems with batch size, if >1 it doesn't work
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if VERBOSE:
        print(f"Dataset length: {len(dataset)}")
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")

    # Initialize the model
    task='graph'
    model = GNNStack(dataset.num_features, 32, dataset.num_classes, task=task)
    print(dataset.num_features)
    writer=SummaryWriter()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(200):
            total_loss = 0
            model.train()
            for batch in train_loader:
                #print(batch.train_mask, '----')
                optimizer.zero_grad()
                embedding, pred = model(batch)
                label = batch.y
                # if task == 'node':
                #     pred = pred[batch.train_mask]
                #     label = label[batch.train_mask]
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(train_loader.dataset)
            writer.add_scalar("loss", total_loss, epoch)

            if epoch % 10 == 0:
                test_acc = test(test_loader, model)
                print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                    epoch, total_loss, test_acc))
                writer.add_scalar("test accuracy", test_acc, epoch)