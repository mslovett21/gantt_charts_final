# Required libraries
import gc
import random
import argparse
import time
import os,sys

import seaborn as sns
import numpy as np

import torch
import torchvision
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from PIL import Image
import mlflow

from utils.early_stopping import EarlyStopping
from utils.plotting import plot_cm, draw_training_curves


# TO ADD if memory issues encounter
gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# Constant variables
IMG_SIZE = [224, 224]  # this is for Augmentation - Image Resize
tensor = (3,224, 224) # this is to predict the in_features of FC Layers




def get_arguments():

    parser = argparse.ArgumentParser('Training VGG-16')
    parser.add_argument('data_dir', type= str, help="Path to the folder with the data")
    parser.add_argument('--epochs', type=int, default= 30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=4, help="Patience for early call")
    parser.add_argument('--add_transformation', default=False, help="Add image transformation | default:False")
    parser.add_argument('--hpo_mode', default=True, help="Run in HPO model validate on validation data | default:True ")
    parser.add_argument('--frozen_layer', default="10", help="How many layers in the model should be frozen")
    args = parser.parse_args()
    return args




class VGG16Model(torch.nn.Module):
    """
    VGG16 pretrained model with additional projection head for transfer learning
        
    """
    def __init__(self,frozen_layer):
        super(VGG16Model, self).__init__()
        
        self.body = torchvision.models.vgg16(pretrained=True).features
        
        for name,child in self.body.named_children():
            if name == frozen_layer:             
                break
            for params in child.parameters():
                params.requires_grad = False
           
      
        self.in_feat = self.get_dim(tensor)
        self.head = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_features=self.in_feat, out_features=512, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(in_features=512, out_features=128, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(128,4)
        )

    def get_dim(self, input_size):
        bs = 1
        ip = torch.rand(bs, *input_size)
        output = self.body(ip)
        op_view = output.view(bs,-1)
        return op_view.shape[-1]
        
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


        

def train_loop(model, tloader, vloader, criterion, optimizer):
    """
    returns loss and accuracy of the model for 1 epoch.
    params: model -  vgg16
          tloader - train dataset
          vloader - val dataset
          criterion - loss function
          optimizer - Adam optimizer
    """
    total = 0
    correct = 0
    train_losses = []
    valid_losses = []
    t_epoch_accuracy = 0
    v_epoch_accuracy = 0
    
    model.train()
    model.to(DEVICE)
    
    for ind, (image, label) in enumerate(tloader):
     
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        
        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, label)
        train_losses.append(loss.item())
        
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted==label).sum().item()

        loss.backward()
        optimizer.step()

    t_epoch_accuracy = correct/total
    t_epoch_loss = np.average(train_losses)
    
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for ind, (image, label) in enumerate(vloader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)
            loss = criterion(output, label)
            valid_losses.append(loss.item())

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
    
    
    v_epoch_accuracy = correct/total
    v_epoch_loss = np.average(valid_losses)
        
    
    return t_epoch_loss, t_epoch_accuracy, v_epoch_loss, v_epoch_accuracy

def train_model(traindata, valdata, checkpoint_dir, args):
    """
    returns losses (train and val), accuracies (train and val), trained_model
    params: trainloader = train dataset
            valloader = validation dataset
    """
    epochs       = args.epochs
    batch_size   = args.batch_size
    patience     = args.patience
    frozen_layer = args.frozen_layer

    LR = {"body":1e-6, "head":1e-4} 

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("patience", patience)
    mlflow.log_param("frozen_layer", frozen_layer)
    mlflow.log_param("lr_body", LR['body'])
    mlflow.log_param("lr_head", LR['head'])

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    valloader   = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=True)
    
    used_early_stopping  = False
    model = VGG16Model(frozen_layer).to(DEVICE)   
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam([{'params': model.body.parameters(), 'lr':LR['body']},
                                 {'params':model.head.parameters(), 'lr':LR['head']}])
    
    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []
    
    early_stop = EarlyStopping(path= checkpoint_dir + '/early_stopping_vgg16model.pth' ,patience=patience)
    
    for epoch in range(epochs):
        print("Running Epoch {}".format(epoch+1))

        epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc = train_loop(model, trainloader, valloader, criterion, optimizer)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, epoch_train_acc))
        print("Validation loss: {0:.4f}  Validation Accuracy: {1:0.2f}".format(epoch_val_loss, epoch_val_acc))

        print("--------------------------------------------------------")
        mlflow.log_metric("train_loss", epoch_train_loss, epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, epoch)
        mlflow.log_metric("train_acc", epoch_train_acc, epoch)
        mlflow.log_metric("val_acc", epoch_val_acc, epoch)
        
        early_stop(epoch_val_loss, model)
    
        if early_stop.early_stop:
            print("Early stopping")
            used_early_stopping  = True
            break
        
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), checkpoint_dir + "/vgg16_epoch_" + str(epoch+1) + ".pth")
            mlflow.pytorch.log_model(model, "vgg_16_epoch_{}.pth".format(str(epoch+1)))

    print("Training completed!")
    losses = [train_loss, val_loss]
    accuracies = [train_acc, val_acc]
    
    return losses, accuracies, model, valloader, used_early_stopping, epoch+1


def run_inference(model,testloader,epoch, additional_transformation, vis_results_path):

    total = 0
    correct = 0
    y_act = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for ind, (image, label) in enumerate(testloader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
          
            y_act.extend(label.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
                
    accuracy = correct/total
    print("Test Accuracy: {}".format(accuracy))
    prec, recall, fscore, _ = precision_recall_fscore_support(y_act, y_pred, average='macro')
    
    mlflow.log_metric("Test Accuracy", accuracy)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1score", fscore)

    plot_cm(y_act, y_pred,epoch, additional_transformation, vis_results_path)
    


    
def get_transformations(train_flag):
    transforms = []
    p = np.random.uniform(0, 1)

    transforms.append(torchvision.transforms.Resize(IMG_SIZE))
      
    if train_flag == "train":
        if p <= 0.4:
            transforms.append(torchvision.transforms.ColorJitter(brightness=1.2))
        transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
        
    transforms.append(torchvision.transforms.ToTensor())   
    return torchvision.transforms.Compose(transforms)


def setup_exeriment(data_dir, timestr, additional_transformation, hpo_model_dev):
    
    train_data_dir =   data_dir + '/train'
    val_data_dir   =   data_dir + '/validation'
    test_data_dir  =   data_dir + '/test'

    train_transforms = None
    test_transforms  = None
    
    if additional_transformation:
        train_transforms = get_transformations("train")
        test_transforms  = get_transformations("test")
    else:
        train_transforms = transforms.Compose([transforms.Resize(IMG_SIZE),transforms.ToTensor()])
        test_transforms  = transforms.Compose([transforms.Resize(IMG_SIZE),transforms.ToTensor()])     

    train_dataset = datasets.ImageFolder(train_data_dir , transform=train_transforms)
    test_dataset  = None
    
    if hpo_model_dev:
        train_dataset = datasets.ImageFolder(train_data_dir , transform=train_transforms)
        test_dataset  = datasets.ImageFolder(val_data_dir, transform=test_transforms)
    else:
        val_dataset   = datasets.ImageFolder(val_data_dir , transform=train_transforms)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset,val_dataset])
        test_dataset  = datasets.ImageFolder(test_data_dir, transform=test_transforms)   


    return train_dataset, test_dataset

    
def main():

    time_start = time.time()
    str_time = time.strftime("%Y%m%d-%H%M%S")

    mlflow.set_experiment('gantt_charts_vgg16')
    run =  mlflow.start_run()

    args     = get_arguments()
    data_dir = args.data_dir

    additional_transformation = args.add_transformation
    hpo_model_dev = args.hpo_mode
    frozen_layer = args.frozen_layer


    mlflow.log_param("data_dir", data_dir)


    rel_path = os.getcwd()
    vis_results_path = rel_path + '/exp_results_details/vgg16_224_4_classes/' + str_time
    checkpoint_dir   = rel_path + '/checkpoints/vgg16_224_4_classes/' + str_time


    os.makedirs(vis_results_path)
    os.makedirs(checkpoint_dir)
    
    train_dataset, test_dataset = setup_exeriment(data_dir, str_time, additional_transformation, hpo_model_dev)
    losses, accuracies, model, test_loader, early_stop,epoch = train_model(train_dataset, test_dataset,checkpoint_dir, args)
    mlflow.log_param("epochs_trained", epoch)

    if early_stop:
        model = VGG16Model(frozen_layer).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_dir +'/early_stopping_vgg16model.pth'))

    run_inference(model, test_loader,epoch, additional_transformation, vis_results_path)
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1],loss_curve,epoch,additional_transformation, vis_results_path)
    acc_curve = "accuracy"
    draw_training_curves(accuracies[0], accuracies[1] ,acc_curve,epoch, additional_transformation, vis_results_path)
    torch.save(model.state_dict(), checkpoint_dir + "/vgg16_final_" + str_time + ".pth")
    mlflow.pytorch.log_model(model, "vgg_16_final_model")
    

    exec_time = time.time() - time_start
    mlflow.log_param("exec_time",exec_time)
    print('Execution time in seconds: ' + str(exec_time))
    return

if __name__ == "__main__":
    main()