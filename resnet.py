import torch
import torchvision
import os
from PIL import Image
import seaborn as sns
import cv2
import numpy as np
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchsummary import summary
from torchvision import datasets, transforms
import gc
from IPython import embed
import time
import mlflow
from torchsummary import summary
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Paths
REL_PATH = os.getcwd()
DATA_DIR = "/datasets/gantt_chart_images_224by224_nogenome"
TRAIN_DATA_DIR = REL_PATH + DATA_DIR + '/train'
VAL_DATA_DIR =  REL_PATH + DATA_DIR + '/validation'
TEST_DATA_DIR = REL_PATH + DATA_DIR + '/test'
MLFLOW_RUNS = REL_PATH + "/mlflow_runs/"

VIS_RESULTS_PATH = REL_PATH + '/visualizations/'

ADDITIONAL_TRANSFORMATION = False
HPO_MODEL_DEV = True

RESIZE = False


# TO ADD if memory issues encounter
gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constant variables
IMG_SIZE = [224, 224]  # this is for Augmentation - Image Resize
tensor = (3, 224, 224) # this is to predict the in_features of FC Layers

BATCH_SIZE = 64
LR = 1e-6 #{"body":1e-4, "head":1e-3}

FREEZE_BLOCKS = "8" # Freezes first seven blocks
EPOCHS = 300
PATIENCE = 15
HIDDEN_DIM = 1024
OUTPUT_DIM = 4


class ResNet50Model(torch.nn.Module):
  def __init__(self):
    super(ResNet50Model,self).__init__()
    self.resnet = torchvision.models.resnet50(pretrained=True)
    # Following step gives us all layers except last one.
    modules = list(self.resnet.children())[:-1]
    self.resnet = torch.nn.Sequential(*modules)
    # Freeze the model parameters for transfer learning.
    # for name,child in self.resnet.named_children():
    #         if name == FREEZE_BLOCKS:
    #             break
    #         for params in child.parameters():
    #             params.requires_grad = False
    
    
    # Classification head of the model.
    self.head = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=2048, out_features=4))
        # torch.nn.BatchNorm1d(128),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(0.4),
        # torch.nn.Linear(128, 4))
    

  def forward(self, x):
    feat = self.resnet(x)
    output = self.head(feat)
    return output



class ResNet18Model(torch.nn.Module):
    """
    ResNet-18 pretrained model with additional projection head for transfer learning
    """
    def __init__(self):
        super(ResNet18Model, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        self.body = torch.nn.Sequential(*modules)

        for name,child in self.body.named_children():
            if name == FREEZE_BLOCKS:
                break
            for params in child.parameters():
                params.requires_grad = False

        self.in_feat = self.get_dim(tensor) # usually 512 if image size is 224x224

        self.head = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_features=self.in_feat, out_features=HIDDEN_DIM, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(HIDDEN_DIM),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    # torch.nn.Linear(in_features=HIDDEN_DIM, out_features=2048, bias=True),
                    # torch.nn.BatchNorm1d(2048),
                    # torch.nn.ReLU(),
                    # torch.nn.Linear(in_features=2048, out_features=512, bias=True),
                    # torch.nn.BatchNorm1d(512),
                    # torch.nn.ReLU(),
   
                    torch.nn.Linear(HIDDEN_DIM,OUTPUT_DIM)
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


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=4, verbose=False, delta=0, path='/early_stopping_resnet18_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_vgg16model.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        mlflow.pytorch.log_model(model, "early_stopping_resnet18_model")
        self.vall_loss_min = val_loss



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


def train_model(traindata, valdata):
    """
    returns losses (train and val), accuracies (train and val), trained_model
    params: trainloader = train dataset
            valloader = validation dataset
    """

    # with mlflow.start_run():
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)
    valloader   = torch.utils.data.DataLoader(valdata, batch_size=BATCH_SIZE, shuffle=True)

    used_early_stopping  = False

    model = ResNet50Model().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    # optimizer = torch.optim.SGD(model.parameters(), LR, momentum=0.9)

    # optimizer = torch.optim.Adam([{'params': model.body.parameters(), 'lr':LR['body']},
    #                             {'params':model.head.parameters(), 'lr':LR['head']}])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, threshold = 0.01)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    mlflow.log_param("Model", 'ResNet-50')
    mlflow.log_param("LR", LR)
    # mlflow.log_param("LR_body", LR['body'])
    # mlflow.log_param("LR_head", LR['head'])
    mlflow.log_param("Epochs", EPOCHS)
    mlflow.log_param("Batch Size", BATCH_SIZE)
    mlflow.log_param("Dataset", DATA_DIR)
    mlflow.log_param("Blocks Frozen", FREEZE_BLOCKS)
    if ADDITIONAL_TRANSFORMATION:
        mlflow.log_param("Additional Transformation", "True")
    else:
        mlflow.log_param("Additional Transformation", "False")

    early_stop = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        print("Running Epoch {}".format(epoch+1))
        epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc = train_loop(model, trainloader, valloader, criterion, optimizer)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)
    
        mlflow.log_metric("train_loss", epoch_train_loss, epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, epoch)
        mlflow.log_metric("train_acc", epoch_train_acc, epoch)
        mlflow.log_metric("val_acc", epoch_val_acc, epoch)

        print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, epoch_train_acc))
        print("Validation loss: {0:.4f}  Validation Accuracy: {1:0.2f}".format(epoch_val_loss, epoch_val_acc))
        print("--------------------------------------------------------")
        scheduler.step(epoch_val_loss)
        early_stop(epoch_val_loss, model)

        if early_stop.early_stop:
            print("Early stopping")
            used_early_stopping  = True
            break

        if (epoch+1)%5 == 0:
            mlflow.pytorch.log_model(model, "resnet18_epoch_{}.pth".format(str(epoch+1)))

    print("Training completed!")
    losses = [train_loss, val_loss]
    accuracies = [train_acc, val_acc]

    return  losses, accuracies, model, valloader, used_early_stopping, epoch


def run_inference(model, testloader):

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

    plot_cm(y_act, y_pred)
    

def plot_cm(lab, pred):
    target_names = ["cpu","hdd","loss","normal"]
    cm = confusion_matrix(lab, pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap = "YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout(h_pad=5.0)

    if ADDITIONAL_TRANSFORMATION :
        plt.title("ResNet-18 with data augmentation")
    else:
        plt.title("ResNet-18 without data augmentation")

    plt.savefig(VIS_RESULTS_PATH + "/confusion_matrix.png")
    mlflow.log_artifact(VIS_RESULTS_PATH + "/confusion_matrix.png")
    plt.close()


def get_transformations(train_flag):
    transforms = []
    p = np.random.uniform(0, 1)

    if RESIZE:
        transforms.append(torchvision.transforms.Resize(IMG_SIZE))

    if train_flag == "train":
        if p <= 0.4:
            transforms.append(torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        if p<=0.8 and p>0.4:
            transforms.append(torchvision.transforms.RandomGrayscale(p=0.2))

        transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.3))
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)


def draw_training_curves(train_losses, test_losses, curve_name, epoch):
    plt.clf()
    max_y = 2.0
    if curve_name == "Accuracy":
        max_y = 1.0
        plt.ylim([0,max_y])
        
    plt.xlim([0,epoch])
    plt.xticks(np.arange(0, epoch, 3))
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Testing {}'.format(curve_name))
    plt.ylabel(curve_name)
    plt.xlabel('Epoch')
    if ADDITIONAL_TRANSFORMATION :
        plt.title("ResNet-18 with data augmentation")
    else:
        plt.title("ResNet-18 without data augmentation")
    plt.legend(frameon=False)
    plt.savefig(VIS_RESULTS_PATH + "/{}_final_resnet.png".format(curve_name))
    mlflow.log_artifact(VIS_RESULTS_PATH + "/{}_final_resnet.png".format(curve_name))
    plt.close()

def my_collate(batch):
    images, labels = [], []
    for image, label in batch:
        image =  cv2.cvtColor(np.array(image), cv2.COLOR_BGRA2BGR)
        images.append(image)
        labels.append(label)
    return [images, labels]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)#, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main():

    train_transforms = None
    test_transforms  = None

    # mlflow.create_experiment("gantt_charts", MLFLOW_RUNS)
    mlflow.set_experiment('gantt_charts') 
    # mlflow.set_tracking_uri(MLFLOW_RUNS)
    run = mlflow.start_run()

    if ADDITIONAL_TRANSFORMATION:
        train_transforms = get_transformations("train")
        test_transforms  = get_transformations("test")
    else:
        if RESIZE:
            train_transforms = transforms.Compose([transforms.Resize(IMG_SIZE),transforms.ToTensor()])
            test_transforms  = transforms.Compose([transforms.Resize(IMG_SIZE),transforms.ToTensor()])
        else:
            train_transforms = transforms.Compose([transforms.ToTensor()])
            test_transforms  = transforms.Compose([transforms.ToTensor()])
        

    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR , transform=train_transforms)
    test_dataset  = None

    if HPO_MODEL_DEV:
        train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR , transform=train_transforms)
        test_dataset  = datasets.ImageFolder(VAL_DATA_DIR, transform=test_transforms)
    else:
        val_dataset   = datasets.ImageFolder(VAL_DATA_DIR , transform=train_transforms)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset,val_dataset])
        test_dataset  = datasets.ImageFolder(TEST_DATA_DIR, transform=test_transforms)


    losses, accuracies, model, test_loader, early_stop, epoch = train_model(train_dataset, test_dataset)

    if early_stop:
        mlflow.pytorch.log_model(model, "early_stopping_resnet18_model")
        model = ResNet18Model().to(DEVICE)
        model_uri =  "runs:/{}/early_stopping_resnet18_model".format(run.info.run_id)
        model = mlflow.pytorch.load_model(model_uri)
    else:
        mlflow.pytorch.log_model(model, "resnet18_final_model")

    run_inference(model, test_loader)
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1],loss_curve, epoch)
    acc_curve = "accuracy"
    draw_training_curves(accuracies[0], accuracies[1] ,acc_curve, epoch)


    return


if __name__ == "__main__":
    main()