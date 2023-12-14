import torch
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score
import copy




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