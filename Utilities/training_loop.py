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
from Utilities.custom_functions import generate_graph_activation_map
import numpy as np
from models.GIN import *
import random
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import copy
import hcatnetwork
from torch_geometric.nn.models.basic_gnn import GAT 


def train_model(model, train_dataset, val_dataset, optimizer, writer=SummaryWriter(), batch_size=4, num_epochs=10):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    optimizer = optimizer
    train_loss = []
    val_acc_list = []
    val_f1_list = []
    val_rec_list = []
    val_prec_list = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    for epoch in range(num_epochs):
            total_loss = 0
            model.train()
            for i, batch in enumerate(train_loader):
                if round(i/len(train_loader)*100, 2) % 10 == 0: #printing info each 10% of the training in every epoch
                    print(round(i/len(train_loader)*100, 2), '%', end='\r')
                optimizer.zero_grad()
                _, pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(train_loader.dataset)
            writer.add_scalar("loss", total_loss, epoch)
            train_loss.append(total_loss)

            if epoch % 1 == 0:
                val_acc, val_f1, val_rec, val_prec = test(val_loader, model)
                #allows to save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_list.append(val_acc)
                val_f1_list.append(val_f1)
                val_rec_list.append(val_rec)
                val_prec_list.append(val_prec)
                print("Epoch {}. Loss: {:.4f}. Val accuracy: {:.4f}".format(
                    epoch, total_loss, val_acc))
                writer.add_scalar("Val accuracy", val_acc, epoch)
                writer.add_scalar("Val f1", val_f1, epoch)
                writer.add_scalar("Val recall", val_rec, epoch)
                writer.add_scalar("Val precision", val_prec, epoch)
    last_model = model.state_dict()

    return last_model, best_model_wts, train_loss, val_acc_list, val_f1_list, val_rec_list, val_prec_list


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    for data in loader:
        with torch.no_grad():
            _, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        # if model.task == 'node':
        #     mask = data.val_mask if is_validation else data.test_mask
        #     # node classification: only evaluate on nodes in test set
        #     pred = pred[mask]
        #     label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
        y_true.extend(label.tolist())
        y_pred.extend(pred.tolist())
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    
    accuracy = correct / total
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    
    return accuracy, f1, recall, precision


if __name__ == "__main__":
    #set random seed
    torch.manual_seed(1) #even if seed is set to 1, results are varying from run to run
    VERBOSE = True
    ROOT = '/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/'
    NODE_ATTS = ['x', 'y', 'z', "topology"]
    EDGE_ATTS = ['weight']
    # Set up the dataset
    dataset = ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json', node_atts=NODE_ATTS, edge_atts=EDGE_ATTS, augment=0.9)
    #get the model
    #MODEL = GATcustom(dataset.num_node_features, 32, 2, 3, dropout=0.25)
    MODEL = GNNStack(dataset.num_node_features, 32, dataset.num_classes,  task='graph')
    
    #print length of the node features of the dataset
    if VERBOSE:
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print("Creating subsets...")
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

    """
    In case no test-time augmentation is used, val_dataset and test_dataset are will need to sample from not_augmented_dataset
    where not_augmented_dataset is the same as dataset but with augment=None, in this case the dataset is not augmented and the
    splitting indexiing is correct with no repetition because shuffling is done on indexes and not on samples themselves.
    Example:
        not_augmented_dataset= ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json', node_atts=['x', 'y', 'z', "topology"], edge_atts=['weight'], augment=None)
        val_dataset = torch.utils.data.Subset(not_augmented_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(not_augmented_dataset, test_indices)
        """
    
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
    model = MODEL
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    warmup_factor = 0.1
    warmup_epochs = 5
    total_epochs = 30
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - warmup_factor) * epoch / warmup_epochs if epoch < warmup_epochs else warmup_factor ** (epoch - warmup_epochs))
    
    writer = SummaryWriter()

    if VERBOSE:
        print("Starting training...")

    last_model, best_model, train_loss, val_acc, val_f1, val_rec, val_prec = train_model(model, train_dataset=train_dataset, \
                                                                                         val_dataset=val_dataset, \
                                                                                         optimizer=optimizer, writer=writer,\
                                                                                         num_epochs=30)


        
    #save model as artery_model.pt
    torch.save(best_model, 'artery_model_best_val_acc.pt')
    torch.save(last_model, 'artery_model_last.pt')

    # Compute ROC curve
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = MODEL #it would be better to initialize the model instead of overwriting the trained one but the result is the same
    #load best model
    best_model = torch.load('artery_model_best_val_acc.pt')
    model.load_state_dict(best_model)
    model.eval()
    y_true = []
    y_scores = []
    #create a list to store the embeddings of the test samples i.e. the output of pooling layer before the final FC layers
    emb_list= []
    for data in test_loader:
        with torch.no_grad():
            emb, pred = model(data)
            emb_list.append(emb.tolist())
            pred = pred.argmax(dim=1)
            label = data.y
        y_true.extend(label.tolist())
        y_scores.extend(pred.tolist())
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    #given emb_list and y_true list, project to tensorbaord embeddings the labelled emb_lsit elements with labels from y_true
    emb_list = torch.tensor(emb_list).reshape(len(emb_list), -1)
    writer.add_embedding(emb_list, metadata=torch.tensor(y_true))

    #visualize the activation maps of the test samples re-projected on graph structure    
    for i in test_indices:
        graph = dataset.get_raw_netwrorkx_graph(i)
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
        generate_graph_activation_map(graph, NODE_ATTS, EDGE_ATTS, model, backend='networkx')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    