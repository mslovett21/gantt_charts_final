import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import glob
import cv2
from PIL import Image
import os, gc, time
import seaborn as sns
import pickle as pkl
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.0f}'.format
from sklearn.manifold import TSNE
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from models.simple_AE import AutoEncoder
from utils.early_stopping import EarlyStopping

from IPython import embed

import mlflow


####################
# TODO
# add resume training
# add more mlflow


gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
#################################################################################################


def get_arguments():

    parser = argparse.ArgumentParser('Train or Eval AutoEncoder')
    parser.add_argument('-data_dir', type= str, help="Path to the folder with the data")
    parser.add_argument('--epochs', type=int, default= 100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Number of training epochs")
    parser.add_argument('--img_res', type=int, default=224, help="Image resolution")
    parser.add_argument('-e','--evalmode', help ='if set looks for checkpoint and evaluate data on it', action= "store_true")
    parser.add_argument('--timestamp', help='Enter particular run timestr', type=str, default=None)
    parser.add_argument('--model_ckpt', help='Enter specific checkpointed model name eg: ae_model_60.pth', default='ae_model_10.pth', type=str)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping threshold')
    args = parser.parse_args()
    return args


def draw_training_curves(train_losses, test_losses,curve_name, epochs,path):
    plt.clf()
    max_y = 0
    min_y = 0
    if curve_name == "accuracy":
        max_y = 1.0
    else:
        max_y = max(test_losses)
        if max(train_losses) > max_y:
            max_y = max(train_losses)
        min_y = min(train_losses)
    plt.ylim([min_y,max_y])
    plt.xlim([0,epochs])
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Testing {}'.format(curve_name))
    plt.legend(frameon=False)
    plt.savefig(path +"/{}_4_AE.png".format(curve_name))



def read_res_img(res, data_dir):  
    train_transforms = transforms.Compose([transforms.Resize([res,res]),transforms.ToTensor()])
    gantt_dataset    = datasets.ImageFolder(data_dir, transform=train_transforms)

    return gantt_dataset


# visualize test data reconstructions
def vis_reconstruction(model,gantt_dataset,train_it, pad, recon_imgs_dir):

    gantt_dataset_iter = iter(gantt_dataset)
    model.eval()      # set model in evalidation mode (eg freeze batchnorm params)
    input_imgs, test_reconstructions = [], []


    for i in range(3):
        input_img = np.asarray(next( gantt_dataset_iter )[0])
        reconstruction = model.reconstruct(torch.tensor(input_img[None], device=DEVICE))
        reconstruction = F.pad(input= reconstruction, pad=(0, pad, pad, 0), mode='constant', value=0)

        inv_tensor = torch.squeeze(reconstruction)
        input_imgs.append(input_img[0])
        test_reconstructions.append(inv_tensor[0].data.cpu().numpy() )

        for i in range(50):
            img = np.asarray(next( gantt_dataset_iter )[0])


    fig = plt.figure(figsize=(15,10))   
    ax1 = plt.subplot(111)
    ax1.imshow(np.concatenate([np.concatenate(input_imgs, axis=1),
                            np.concatenate(test_reconstructions, axis=1)], axis=0),cmap= 'nipy_spectral')
    plt.axis('off')
    plt.savefig(recon_imgs_dir + "/recon_" +str(train_it) +".png",bbox_inches='tight', pad_inches=0)
    return input_imgs



######################################################################################################3

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    def __init__(self, patience=4, verbose=False, delta=0.0001, path='early_stopping_ae_model.pth'):
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
    
    def __call__(self, val_loss, model, save_dir):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
            self.counter = 0   
    
    def save_checkpoint(self, val_loss, model, save_dir):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save(model.state_dict(), save_dir + self.path)
        self.vall_loss_min = val_loss



def train_model(ae_model, gantt_data_loader, gantt_dataset_test, pad, epochs, lr, recon_imgs_dir, checkpoint_dir, patience):
    
    rec_loss_all = []
    test_loss_all = []
    gantt_data_test_loader = DataLoader(gantt_dataset_test, batch_size=4, shuffle=True, num_workers=4)

    opt = optim.Adam(ae_model.parameters(), lr=lr) # create optimizer instance
    criterion = nn.MSELoss().to(DEVICE)    # create loss layer instance
    ae_model  = ae_model.train()   # set model in train mode (eg batchnorm params get updated)
    ae_model.to(DEVICE)
    train_it = 0

    early_stop = EarlyStopping(patience=patience)

    for ep in range(epochs):
        print("Run Epoch {}".format(ep))
        rec_loss_ep = 0
        ae_model.train()

        for sample_img, _ in gantt_data_loader:
            sample_img = sample_img.to(DEVICE)
            opt.zero_grad()
            dec = ae_model(sample_img)
            result = F.pad(input= dec, pad=(0, pad, pad, 0), mode='constant', value=0)
            rec_loss = criterion(result, sample_img)
            rec_loss_ep += rec_loss
            rec_loss.backward()
            opt.step()
        rec_loss = ( rec_loss_ep/ len(gantt_data_loader) ).item()
        rec_loss_all.append(rec_loss)

        print("Epoch {}: Reconstruction Loss: {}".format(train_it, rec_loss))
        if (ep+1) % 10 == 0:
            torch.save(ae_model.state_dict(), checkpoint_dir + "/ae_model_"+ str(train_it+1) + ".pth")
        
        img = vis_reconstruction(ae_model,gantt_dataset_test, str(train_it), pad, recon_imgs_dir)
        test_loss = 0
        ae_model.eval()

        with torch.no_grad():
            for input_img, _ in gantt_data_test_loader:
                input_img = input_img.to(DEVICE)
                reconstruction = ae_model.reconstruct(input_img)
                reconstruction = F.pad(input= reconstruction, pad=(0, pad, pad, 0), mode='constant', value=0)
                test_loss += criterion( reconstruction, input_img)

            test_loss = (test_loss/len(gantt_data_test_loader)).item()
            print(" Test Reconstruction Loss: {}".format(test_loss))
        
        test_loss_all.append(test_loss)
        early_stop(test_loss, ae_model, checkpoint_dir)

        if early_stop.early_stop:
            print("Early stopping")
            break

        train_it += 1

    return rec_loss_all, test_loss_all


def create_cols(embed_size):
    col = 'col'
    cols = []
    for i in range(embed_size):
        cols.append(col + str(i))
    return cols

def generate_embedding_csv(embeddings, labels, embed_size, path, file_name):
    
    cols = create_cols(embed_size)
    embed_df = pd.DataFrame(columns=cols)
    
    for i,rows in enumerate(embeddings):
        a_series = pd.Series(rows, index=embed_df.columns)
        embed_df = embed_df.append(a_series, ignore_index=True)
    
    targets = pd.DataFrame(labels)
   
    embed_df['target'] = targets
    
    embed_df.to_csv(path + file_name, index=False)


def eval_model(ae_model, dataloader, embed_size, pad):

    criterion = nn.MSELoss().to(DEVICE)
    ae_model.to(DEVICE)
    ae_model.eval()   
    embeddings = np.empty(shape=[0, embed_size])
    labels = []
    losses = []

    for image, label in dataloader:
        image = image.to(DEVICE)
        
        embed = ae_model.embed(image)
        dec = ae_model.decode(embed)

        result = F.pad(input= dec, pad=(0, pad, pad, 0), mode='constant', value=0)
        loss = criterion(result,image)
        losses.append(loss.item())
        embed = embed.detach().cpu().numpy()

        labels.extend(label)
        embeddings = np.append(embeddings, embed, axis=0)

    return embeddings, np.array(labels), losses


def get_AE_params(img_res):

    shallow_layer, deep_layer = 256, 2048    
    z_emb_size    = 128
    pad, dim      = 7, 2         
    features      = 512
    var_stride    = 2

    if img_res == 512:
        shallow_layer, deep_layer = 2048, 6400 
        z_emb_size    = 1024
        pad, dim      = 25, 5    
        features      = 256
        var_stride    = 3

    mlflow.log_param("shallow_layer", shallow_layer)
    mlflow.log_param("deep_layer", deep_layer)
    mlflow.log_param("z_emb_size", z_emb_size)
    return shallow_layer, z_emb_size, pad, deep_layer,dim, features, var_stride


class Dataset_Loader(torch.utils.data.Dataset):
    
    def __init__(self, data, label,  transform = None):
        self.data = data
        self.label = label
        self.transform = transform
        self.length = len(self.data)

    def __len__(self):
        return self.length
  
    def __getitem__(self, idx):
        
        img = cv2.imread(self.data[idx])
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            image =  self.transform(image)
        label = self.label[idx]

        return image, label


def get_anomalies(rel_path, res, nworkers):
    """
    Creates a binary dataloader with class 0 as Normal Test data, and class 1 as Anomalous (cpu, hdd, loss) Test data.
    :param rel_path: relative path os.getcwd()
    :param res: Image resolution
    :param nworkers: Number of cpu threads
    :return anomaly_data_loader: anomaly dataset loader
    :return normal_data_loader: normal dataset loader
    """
    normal_path = rel_path + '/datasets/gantt_chart_images_224by224_nogenome_NORMAL/test/normal/'
    cpu_path    = rel_path + '/datasets/gantt_chart_images_224by224_nogenome_CPU/test/cpu/'
    hdd_path    = rel_path + '/datasets/gantt_chart_images_224by224_nogenome_HDD/test/hdd/'
    loss_path   = rel_path + '/datasets/gantt_chart_images_224by224_nogenome_LOSS/test/loss/'

    anomalies = glob.glob(cpu_path + '*.png')
    anomalies += glob.glob(hdd_path + '*.png')
    anomalies += glob.glob(loss_path + '*.png')
    anomaly_labels = [1]*len(anomalies)

    print("Size of anomalies: ",len(anomalies))

    normal = glob.glob(normal_path + '*.png')
    normal_labels = [0]*len(normal)

    print("Size of normal data: ",len(normal))    
    
    test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize([res,res]),transforms.ToTensor()])

    anomaly_dataset = Dataset_Loader(anomalies, anomaly_labels, test_transforms)
    anomaly_data_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=True, num_workers=nworkers)

    normal_dataset = Dataset_Loader(normal, normal_labels, test_transforms)
    normal_data_loader = DataLoader(normal_dataset, batch_size=1, shuffle=True, num_workers=nworkers)

    return anomaly_data_loader, normal_data_loader


def plot_distplot(error, figname):

    sns.displot(error, bins=50, kde=True)
    plt.savefig(figname)



def main():

    time_start = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    torch.cuda.memory_summary(device=None, abbreviated=False)
    args = get_arguments()

    mlflow.set_experiment('gantt_charts_AE')
    run =  mlflow.start_run()

    epochs     = args.epochs
    data_dir   = args.data_dir
    batch_size = args.batch_size
    img_resolution = args.img_res
    patience = args.patience
    lr             = 0.001
    checkpoint_number = 600
    timestamp = args.timestamp
    model_ckpt = args.model_ckpt

    mlflow.log_param("data_dir", data_dir)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("img_resolution", img_resolution)
    mlflow.log_param("lr", lr)

    rel_path   = os.getcwd()
    dir_string = "AE_test"


    shallow_layer, z_emb_size, pad, deep_layer,dim, features, var_stride = get_AE_params(img_resolution)
    nworkers   = 4 # number of wrokers used for efficient data loading
    ae_model   = AutoEncoder(deep_layer, shallow_layer, z_emb_size, dim, features, var_stride)


    if args.evalmode == True:
  
        artifacts_dir = rel_path + '/artifacts/artifact_' + dir_string + "_" + timestamp + '/'
        load_model_ckp  = rel_path + "/checkpoints/checkpoint_" + dir_string + "_" + timestamp + '/' + model_ckpt

        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
 
        normal_csv = 'normal_embeddings.csv'
        anomaly_csv = 'anomaly_embeddings.csv'
      
        anomaly_data_loader, normal_data_loader = get_anomalies(rel_path, img_resolution, nworkers)

        ae_model.load_state_dict(torch.load(load_model_ckp))

        anom_embed, anom_target, anomaly_error = eval_model(ae_model, anomaly_data_loader, z_emb_size, pad)
        norm_embed, norm_target, normal_error = eval_model(ae_model, normal_data_loader, z_emb_size, pad)

        generate_embedding_csv(norm_embed, norm_target, z_emb_size, artifacts_dir, normal_csv)
        generate_embedding_csv(anom_embed, anom_target, z_emb_size, artifacts_dir, anomaly_csv)

        plot_distplot(anomaly_error, artifacts_dir + 'anomaly_reconstruction_error_plot.png')
        plot_distplot(normal_error, artifacts_dir + 'normal_reconstruction_error_plot.png')

     
    else:
        
        train_data_dir = data_dir + '/train'
        test_data_dir  = data_dir + '/validation'
        recon_imgs_dir = rel_path + "/recon_imgs/" +"recon_imgs_"+ dir_string + "_" + timestr
        checkpoint_dir = rel_path + "/checkpoints/checkpoint_" + dir_string + "_" + timestr +"/"
        artifacts_dir = rel_path + '/artifacts/artifact_' + dir_string + "_" + timestr + '/'

        os.makedirs(checkpoint_dir)
        os.makedirs(recon_imgs_dir)
        os.makedirs(artifacts_dir)

        gantt_dataset_train = read_res_img(img_resolution, train_data_dir)
        gantt_dataset_test  = read_res_img(img_resolution, test_data_dir)
        gantt_data_loader_train     = DataLoader(gantt_dataset_train, batch_size=batch_size, shuffle=True, num_workers=nworkers)
        rec_loss_all, test_loss_all = train_model(ae_model, gantt_data_loader_train, gantt_dataset_test, pad, epochs, lr, recon_imgs_dir,checkpoint_dir, patience)
        epochs = len(rec_loss_all)

        draw_training_curves(rec_loss_all, test_loss_all, "loss", epochs, artifacts_dir) 


if __name__ == "__main__":
    # execute only if run as a script
    main()
    
