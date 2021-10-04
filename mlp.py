import torch
import numpy as np 
import pandas as pd 
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import umap
import seaborn as sns

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
HIDDEN = 1024
PATIENCE = 15
LR = 1e-4
REL = os.getcwd()
CHECKPOINT = REL + '/mlp_checkpoint/'
VIZ = REL + '/mlp_visualization/'


def read_csv():

    train_csv = '/home/shubham/crisis-computing/SupContrast/train_supcon_embed.csv'
    test_csv = '/home/shubham/crisis-computing/SupContrast/test_supcon_embed.csv'
    val_csv = '/home/shubham/crisis-computing/SupContrast/val_supcon_embed.csv'

    train_embed = pd.read_csv(train_csv)
    test_embed = pd.read_csv(test_csv)
    val_embed = pd.read_csv(val_csv)

    return train_embed, test_embed, val_embed


class EarlyStopping:

    def __init__(self, patience=4, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

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
   
        torch.save(model, CHECKPOINT + "early_stopping_mlp_model.pth")
        self.vall_loss_min = val_loss


class MLP(nn.Module):
    
    def __init__(self, hidden_layer):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, emb):
        return self.mlp(emb)


def create_dataset(df):

    label = df['target'].to_list()
    feature = df.iloc[:,:-1].to_numpy()

    return feature, label


class GanttChartEmbDataset(Dataset):
    def __init__(self, embedding, label):
    
        self.embed, self.labels = embedding, label

    def __len__(self):
        return len(self.embed)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feats = self.embed[idx]
        feats = torch.tensor(feats).float()
        label = self.labels[idx]
        label = torch.tensor(label).long()

        return feats, label


def get_data_loaders(train_feat, train_label, val_feat, val_label, test_feat, test_label):

    trainset = GanttChartEmbDataset(train_feat, train_label)
    valset  = GanttChartEmbDataset(val_feat, val_label)
    testset  = GanttChartEmbDataset(test_feat, test_label)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader


def train_epoch(train_loader, val_loader, model, optimizer, criterion):

    model.to(DEVICE)
    model.train()
    total = 0
    correct = 0
    train_loss = []
    val_loss = []


    for feat, label in train_loader:

        feat = feat.to(DEVICE)
        label = label.to(DEVICE)
        output = model(feat)
        optimizer.zero_grad()
        loss = criterion(output, label)

        train_loss.append(loss.item())
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted==label).sum().item()
        loss.backward()
        optimizer.step()

    t_epoch_accuracy = correct/total
    t_epoch_loss = np.average(train_loss)

    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for feat, label in val_loader:

            feat = feat.to(DEVICE)
            label = label.to(DEVICE)
            output = model(feat)
            loss = criterion(output, label)
            val_loss.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted==label).sum().item()

    v_epoch_accuracy = correct/total
    v_epoch_loss = np.average(val_loss)
    
    return t_epoch_loss, t_epoch_accuracy, v_epoch_loss, v_epoch_accuracy


def train(train_loader, val_loader):

    losses = {'train':[], 'val':[]}
    accuracies = {'train':[], 'val':[]}
    used_early_stopping  = False

    model = MLP(HIDDEN)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)

    early_stop = EarlyStopping(patience=PATIENCE)

    for e in range(EPOCHS):

        print("Running Epoch {}".format(e+1))

        epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc = train_epoch(train_loader, val_loader, model, optimizer, criterion)
        losses['train'].append(epoch_train_loss)
        losses['val'].append(epoch_val_loss)
        accuracies['train'].append(epoch_train_acc)
        accuracies['val'].append(epoch_val_acc)

        print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, epoch_train_acc))
        print("Validation loss: {0:.4f}  Validation Accuracy: {1:0.2f}".format(epoch_val_loss, epoch_val_acc))
        print("--------------------------------------------------------")

        early_stop(epoch_val_loss, model)

        if early_stop.early_stop:
            print("Early stopping")
            used_early_stopping  = True
            break

        if (e+1)%10 == 0:
            torch.save(model, CHECKPOINT + 'mlp_epoch_{}.pth'.format(e+1))

    print("Training completed!")

    return losses, accuracies, model, used_early_stopping, e


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

    plt.title("MLP Supcon Embedding")
    plt.legend(frameon=False)
    plt.savefig(VIZ + "{}_MLP.png".format(curve_name))
    plt.close()

def plot_TSNE_2D(feature_df):
    
    tsne = TSNE(n_components=2, random_state=32, perplexity=30, metric='manhattan')
    tsne_obj = tsne.fit_transform(feature_df)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'labels': feature_df['target']})

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="X", y="Y",
                hue="labels",
                palette=['orange', 'blue', 'red', 'green'],
                legend='full',
                data=tsne_df)

    plt.savefig('tsne_embeddings.png')#artifacts_dir + 'tsne_embeddings.png')

    plt.close()

    return

def viz_umap(features, class_labels):

        reducer = umap.UMAP()

        em = reducer.fit_transform(features)

        plt.scatter(em[:, 0],
            em[:, 1],
            c=[sns.color_palette()[x] for x in class_labels])

        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the gantchart dataset', fontsize=24)
        plt.savefig('UMAP.png')

def main():

    train_df, val_df, test_df = read_csv()
    
    train_feat, train_label = create_dataset(train_df)
    val_feat, val_label = create_dataset(val_df)
    test_feat, test_label = create_dataset(test_df)
    
    viz_umap(test_feat, test_label)
    plot_TSNE_2D(test_df)

    train_loader, val_loader, test_loader = get_data_loaders(train_feat, train_label, val_feat, val_label, test_feat, test_label)

    losses, accuracies, model, used_early_stopping, epoch = train(train_loader, val_loader)

    loss_curve = "loss"
    draw_training_curves(losses['train'], losses['val'], loss_curve, epoch)
    acc_curve = "accuracy"
    draw_training_curves(accuracies['train'], accuracies['val'] , acc_curve, epoch)

    return 


if __name__ == "__main__":
    main()