import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from Utilities.dataset import ArteryGraphDataset
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
import os
from Utilities.custom_functions import get_processed_graphs_names_and_write_reference_txt
import numpy as np
from models.GIN import *
import random

def train_model(model, train_dataset, val_dataset, optimizer, writer=SummaryWriter(), batch_size=4, num_epochs=10):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    optimizer = optimizer
    train_loss = []
    val_acc_list = []
    for epoch in range(num_epochs):
            total_loss = 0
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                embedding, pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(train_loader.dataset)
            writer.add_scalar("loss", total_loss, epoch)
            train_loss.append(total_loss)

            if epoch % 1 == 0:
                val_acc = test(val_loader, model)
                val_acc_list.append(val_acc)
                print("Epoch {}. Loss: {:.4f}. Val accuracy: {:.4f}".format(
                    epoch, total_loss, val_acc))
                writer.add_scalar("Val accuracy", val_acc, epoch)

    return model, train_loss, val_acc_list

    

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
    #set random seed
    torch.manual_seed(1) #evn if seed is set to 1, results are varying from run to run
    VERBOSE = True
    ROOT = '/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/'
    # Set up the dataset
    dataset = ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json')

    #randomly select the 80% of the samples of dataset to train, the 10% to validate and the 10% to test without repetition
    train_size = int(len(dataset)*0.8)
    val_size = int(len(dataset)*0.1)
    test_size = int(len(dataset)*0.1)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    #create 3 subsets with the indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    if VERBOSE:
        print(f"Dataset length: {len(dataset)}")
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Test dataset length: {len(val_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")

    # Initialize the model
    task='graph'
    model = GNNStack(dataset.num_node_features, 32, dataset.num_classes, num_layers=10,  task=task)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    writer = SummaryWriter()

    model, train_loss, val_acc = train_model(model, train_dataset=train_dataset, \
                                             val_dataset=val_dataset, \
                                             optimizer=optimizer, writer=writer, num_epochs=30)
    
    #save model as artery_model.pt
    torch.save(model, 'artery_model.pt')
    