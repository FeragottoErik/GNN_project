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
from Utilities.custom_functions import generate_graph_activation_map, from_networkx, compute_inference_time
import numpy as np
from models.GNN import *
import random
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import copy
import hcatnetwork
from torch_geometric.nn.models.basic_gnn import GAT 
from Utilities.augmentations import random_graph_portion_selection, trim_graph_random_branch, add_graph_random_branch
import sys
from time import sleep


def train_model(model, train_dataset, val_dataset, optimizer, writer=SummaryWriter(), batch_size=4, num_epochs=10):
    #TODO: implement early stopping
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
                # if round(int(i*batch.batch)/len(train_loader)*100, 2) % 10 == 0: #printing info each 10% of the training in every epoch
                #     print(round(int(i*batch.batch)/len(train_loader)*100, 2), '%', end='\r')
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
                if val_acc >= best_val_acc: 
                    best_val_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_list.append(val_acc)
                val_f1_list.append(val_f1)
                val_rec_list.append(val_rec)
                val_prec_list.append(val_prec)
                print('\n\n')
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
    #set random torch seed
    torch.manual_seed(2023)
    #set numpy random seed
    np.random.seed(2023)
    #set random python seed
    random.seed(2023)

    VERBOSE = True
    ROOT = '/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/'
    SAVE_PLOTS_FOLDER= '/home/erikfer/GNN_project/PLOTS/run_12_12_seed2023/'
    #check if save plots folder exists, otherwise create it
    if not os.path.exists(SAVE_PLOTS_FOLDER):
        os.makedirs(SAVE_PLOTS_FOLDER)
    if not os.path.exists(os.path.join(SAVE_PLOTS_FOLDER, 'activation_maps')):
        os.makedirs(os.path.join(SAVE_PLOTS_FOLDER, 'activation_maps'))
    NODE_ATTS = ['x', 'y', 'z', "topology"]
    EDGE_ATTS = ['weight']

    #define 2 files as output for the print function and the error function
    sys.stdout = open(os.path.join(SAVE_PLOTS_FOLDER, 'output.txt'), 'w')
    sys.stderr = open(os.path.join(SAVE_PLOTS_FOLDER, 'error.txt'), 'w')
    # Set up the dataset
    dataset = ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json', node_atts=NODE_ATTS, edge_atts=EDGE_ATTS, augment=0.9)
    #get the model
    MODEL = GATcustom(dataset.num_node_features, 32, 2, 3, dropout=0.25) #its the GAT network
    #MODEL = GNNStack(dataset.num_node_features, 32, dataset.num_classes,  task='graph') #its the GIN network

    #print length of the node features of the dataset
    if VERBOSE:
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print("Creating subsets...")
        total_params = sum(p.numel() for p in MODEL.parameters())
        print(f"Total number of parameters: {total_params}, for model: {MODEL.__class__.__name__}")
    #randomly select the 80% of the samples of dataset to train, the 10% to validate and the 10% to test without repetition
    train_size = int(len(dataset)*0.8)
    val_size = int(len(dataset)*0.1)
    test_size = int(len(dataset)*0.2)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size:] #the test set includes also the validation set, should not be like this but is to have a wider test set

    #create 3 subsets with the indices

    """
    In case no test-time augmentation is used, val_dataset and test_dataset are will need to sample from not_augmented_dataset
    where not_augmented_dataset is the same as dataset but with augment=None, in this case the dataset is not augmented and the
    splitting indexiing is correct with no repetition because shuffling is done on indexes and not on samples themselves.
    Example:
        dataset = ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json', node_atts=['x', 'y', 'z', "topology"], edge_atts=['weight'], augment=0.9)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        not_augmented_dataset= ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json', node_atts=['x', 'y', 'z', "topology"], edge_atts=['weight'], augment=None)
        val_dataset = torch.utils.data.Subset(not_augmented_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(not_augmented_dataset, test_indices)
        
        
    Otherwise if test-time augmentation is used, val_dataset and test_dataset are sampled from dataset, in this case the dataset is augmented
    Example:
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)"""
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    # val_dataset = torch.utils.data.Subset(dataset, val_indices)
    # test_dataset = torch.utils.data.Subset(dataset, test_indices)

    not_augmented_dataset= ArteryGraphDataset(root=ROOT, ann_file='graphs_annotation.json', node_atts=['x', 'y', 'z', "topology"], edge_atts=['weight'], augment=None)
    val_dataset = torch.utils.data.Subset(not_augmented_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(not_augmented_dataset, test_indices)

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
    total_epochs = 20

    # # """Eventually use a scheduler for the learning rate, but actually is not needed because the learning rate is already small enough"""
    # # #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - warmup_factor) * epoch / warmup_epochs if epoch < warmup_epochs else warmup_factor ** (epoch - warmup_epochs))
    
    writer = SummaryWriter()

    if VERBOSE:
        print("\nStarting training...")

    last_model, best_model, train_loss, val_acc, val_f1, val_rec, val_prec = train_model(model, train_dataset=train_dataset, \
                                                                                         val_dataset=val_dataset, \
                                                                                         optimizer=optimizer, writer=writer,\
                                                                                         num_epochs=total_epochs)


        
    #save model as artery_model.pt
    torch.save(best_model, 'artery_model_best_val_acc.pt')
    torch.save(last_model, 'artery_model_last.pt')

    #keep writing the output of the print function to the output.txt file

    # Compute ROC curve
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = MODEL #it would be better to initialize the model instead of overwriting the trained one but the result is the same
    #load best model
    best_model = torch.load('artery_model_best_val_acc.pt')
    model.load_state_dict(best_model)
    model.eval()

    """Evaluate the model inference speed"""

    inference_time = compute_inference_time(model, test_loader)
    if VERBOSE:
        print(f"\nAverage Inference time: {inference_time[0]} +- {inference_time[1]} seconds")

    """Evaluate the model on the test set, with no augmentation"""
    y_true = []
    y_scores = []
    #create a list to store the embeddings of the test samples i.e. the output of pooling layer before the final FC layers
    emb_list= []
    correct = 0
    for data in test_loader:
        with torch.no_grad():
            emb, pred = model(data)
            emb_list.append(emb.tolist())
            pred = pred.argmax(dim=1)
            label = data.y
        correct += pred.eq(label).sum().item()
        y_true.extend(label.tolist())
        y_scores.extend(pred.tolist())
    total = len(test_loader.dataset)
    accuracy = correct / total
    not_aug_test_acc = accuracy
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    if VERBOSE:
        print(f"\nAccuracy on test set: {accuracy}")
        print(f"F1 score on test set: {f1_score(y_true, y_scores, average='macro')}")
        print(f"Recall on test set: {recall_score(y_true, y_scores, average='macro')}")
        print(f"Precision on test set: {precision_score(y_true, y_scores, average='macro')}")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

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
    #save the plot
    plt.savefig(os.path.join(SAVE_PLOTS_FOLDER,'ROC_curve.png'))

    """given emb_list and y_true list, project to tensorbaord embeddings the labelled emb_lsit elements with labels from y_true"""
    emb_list = torch.tensor(emb_list).reshape(len(emb_list), -1)
    writer.add_embedding(emb_list, metadata=torch.tensor(y_true))


    """Visulize the activation of features of the graph nodes of test samples re-projected on graph structure"""
    for i in test_indices:
        graph = dataset.get_raw_netwrorkx_graph(i)
        #hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
        sleep(0.5) #sleep to allow the plot to be saved
        generate_graph_activation_map(graph, NODE_ATTS, EDGE_ATTS, model, backend='networkx', save_path=os.path.join(SAVE_PLOTS_FOLDER, 'activation_maps', f'activation_map_{i}.png'))
        plt.close()

    """Evaluate model capability to classify graphs when random branches are trimmed from the graph"""
    correct=0
    for i in test_indices:
        with torch.no_grad():
            #get the graph from the test dataset
            graph = dataset.get_raw_netwrorkx_graph(i)
            graph_trimmed = trim_graph_random_branch(graph)
            pyGeo_graph = from_networkx(graph_trimmed, NODE_ATTS, EDGE_ATTS)
            #assign the graph label to pygeo object
            pyGeo_graph.y = torch.tensor([dataset[i].y], dtype=torch.long)
            #get model prediction on trimmed graph
            _, pred = model(pyGeo_graph)
            pred = pred.argmax(dim=1)
            label = pyGeo_graph.y
            #compute metrics
            correct += pred.eq(label).sum().item()
    total = len(test_indices)
    accuracy = correct / total
    print(f"\nAccuracy on test set with random branches trimmed: {accuracy}")

    """Evaluate model capability to classify graphs when random branches are ADDED to the graph"""
    correct=0
    for i in test_indices:
        with torch.no_grad():
            #get the graph from the test dataset
            graph = dataset.get_raw_netwrorkx_graph(i)
            graph_trimmed = add_graph_random_branch(graph)
            pyGeo_graph = from_networkx(graph_trimmed, NODE_ATTS, EDGE_ATTS)
            #assign the graph label to pygeo object
            pyGeo_graph.y = torch.tensor([dataset[i].y], dtype=torch.long)
            #get model prediction on trimmed graph
            _, pred = model(pyGeo_graph)
            pred = pred.argmax(dim=1)
            label = pyGeo_graph.y
            #compute metrics
            correct += pred.eq(label).sum().item()
    total = len(test_indices)
    accuracy = correct / total
    print(f"\nAccuracy on test set with random branches added: {accuracy}")

    """Evaluate the model capability to classify the graph with different graph portions selected thanks to the random_graph_portion_selection function
    with a number of nodes that varies from 10 to 600 with a step of 10. Note that if 600 > max_branch_lenght the function will return the whole graph
    The grater brench in the whole dataset is 576 nodes long."""
    #create a for loop of an index that starts at 10 and ends at 600 with a step of 10
    acc_hist = []  
    for j in range(10, 600, 10):
        correct = 0
        # print the progression percentage of the loop
        #print(round(j/600*100, 2), '%', end='\r')
        for i in test_indices:
            with torch.no_grad():
                #get the graph from the test dataset
                graph = dataset.get_raw_netwrorkx_graph(i)
                graph_trimmed = random_graph_portion_selection(graph, j, 'OSTIUM')
                pyGeo_graph = from_networkx(graph_trimmed, NODE_ATTS, EDGE_ATTS)
                #assign the graph label to pygeo object
                pyGeo_graph.y = torch.tensor([dataset[i].y], dtype=torch.long)
                #get model prediction on trimmed graph
                _, pred = model(pyGeo_graph)
                pred = pred.argmax(dim=1)
                label = pyGeo_graph.y
                #compute metrics
                correct += pred.eq(label).sum().item()
        total = len(test_indices)
        accuracy = correct / total
        acc_hist.append(accuracy)
        #if the accuracy is equal to not_aug_test_acc in the two last iterations, then stop the loop
        if j > 20:
            if acc_hist[-1] == acc_hist[-2] == not_aug_test_acc:
                break

    #print(f"\nAccuracy on test set with random graph portions selected: {acc_hist}")

    # Plot accuracy vs graph portion
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(range(10, (len(acc_hist)*10 + 10), 10), acc_hist, marker='o', linestyle='-', color='b')  # Add markers, linestyle, and color
    plt.xlabel('Graph Portion', fontsize=12)  # Set x-axis label and font size
    plt.ylabel('Accuracy', fontsize=12)  # Set y-axis label and font size
    plt.title('Accuracy vs Graph Portion', fontsize=14)  # Set plot title and font size
    plt.xticks(fontsize=10)  # Set x-axis tick font size
    plt.yticks(fontsize=10)  # Set y-axis tick font size
    plt.grid(True)  # Add grid lines
    plt.savefig(os.path.join(SAVE_PLOTS_FOLDER,'Accuracy_vs_Graph_Portion.png'))  # Save plot to file
    