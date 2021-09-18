import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os, gc, time
import pickle as pkl
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.0f}'.format

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from models.simple_AE import AutoEncoder
from IPython import embed

gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
#################################################################################################


def get_arguments():

    parser = argparse.ArgumentParser('Train or Eval AutoEncoder')
    parser.add_argument('data_dir', type= str, help="Path to the folder with the data")
    parser.add_argument('--epochs', type=int, default= 3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Number of training epochs")
    parser.add_argument('--img_res', type=int, default=224, help="Image resolution")
    parser.add_argument('-e','--evalmode', help ='if set looks for checkpoint and evaluate data on it', action= "store_true")
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

def train_model(ae_model, gantt_data_loader, gantt_dataset_test, pad, epochs, lr, recon_imgs_dir,checkpoint_dir):
    
    rec_loss_all = []
    test_loss_all = []
    gantt_data_test_loader = DataLoader(gantt_dataset_test, batch_size=4, shuffle=True, num_workers=4)

    opt = optim.Adam(ae_model.parameters(), lr=lr) # create optimizer instance
    criterion = nn.MSELoss().to(DEVICE)    # create loss layer instance
    ae_model  = ae_model.train()   # set model in train mode (eg batchnorm params get updated)
    ae_model.to(DEVICE)
    train_it = 0

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
        if ep % 5 == 0:
            torch.save(ae_model.state_dict(), checkpoint_dir + "/ae_model_"+ str(train_it) + ".pth")
        
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
        train_it += 1
    
    return rec_loss_all, test_loss_all




def eval_model(ae_model,gantt_data_loader,gantt_dataset,dataset_name,pad,recon_imgs_dir):
   
    ae_model.to(DEVICE)
    ae_model = ae_model.eval()             # set model in train mode (eg batchnorm params get updated)
    data_matrix = []
   
    for sample_img, label in gantt_data_loader:
    
        sample_img = sample_img.to(DEVICE)
        dec = ae_model.embed(sample_img)
        dec = dec.detach().cpu().numpy() 
        labels = label.detach().cpu().numpy()
        for em, lab in zip(dec, labels):
            em = np.append(em,lab)
            data_matrix.append(em)
    data_matrix = np.array(data_matrix)

    file = "data_emb_4_classes.npy"
    np.save(file, data_matrix)

    img = vis_reconstruction(ae_model,gantt_dataset,str(0),pad,recon_imgs_dir)


def get_AE_params(img_res):

    shallow_layer, deep_layer = 256, 2048    
    z_emb_size    = 1024
    pad, dim      = 7, 2         
    features      = 512
    var_stride    = 2

    if img_res == 512:
        shallow_layer, deep_layer = 2048, 6400 
        z_emb_size    = 1048
        pad, dim      = 25, 5    
        features      = 256
        var_stride    = 3

    return shallow_layer, z_emb_size, pad, deep_layer,dim, features, var_stride




def main():

    time_start = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    torch.cuda.memory_summary(device=None, abbreviated=False)
    args = get_arguments()

    epochs   = args.epochs
    data_dir = args.data_dir
    img_resolution = args.img_res
    lr             = 0.0001
    checkpoint_number = 600

    rel_path   = os.getcwd()
    dir_string = "AE_test"
    checkpoint_path =  rel_path  + "checkpoints/checkpoint_" + dir_string + "/ae_model.pth"

    train_data_dir = data_dir + '/train'
    test_data_dir  = data_dir + '/validation'
    embedding_file = rel_path + "/embeddings/emb_"+ dir_string+ "/"
    recon_imgs_dir = rel_path + "/recon_imgs/" +"recon_imgs_"+ dir_string + "_" + timestr
    checkpoint_dir = rel_path + "/checkpoints/checkpoint_" + dir_string + "_" + timestr +"/"

    os.makedirs(checkpoint_dir)
    os.makedirs(recon_imgs_dir)

    shallow_layer, z_emb_size, pad, deep_layer,dim, features, var_stride = get_AE_params(img_resolution)
    batch_size = 4
    nworkers   = 4 # number of wrokers used for efficient data loading
    ae_model   = AutoEncoder(deep_layer, shallow_layer, z_emb_size, dim, features, var_stride)

    if args.evalmode == True:
        for dataset_dir, dataset_name in zip([test_data_dir, train_data_dir], ["test", "train"]):

            gantt_dataset     = read_res_img(img_resolution, dataset_dir)
            gantt_data_loader = DataLoader(gantt_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers)
            ae_model.load_state_dict(torch.load(checkpoint_path))
            eval_model(ae_model, gantt_data_loader,gantt_dataset,dataset_name,pad,recon_imgs_dir)
    else:
        gantt_dataset_train = read_res_img(img_resolution, train_data_dir)
        gantt_dataset_test  = read_res_img(img_resolution, test_data_dir)
        gantt_data_loader_train     = DataLoader(gantt_dataset_train, batch_size=batch_size, shuffle=True, num_workers=nworkers)
        rec_loss_all, test_loss_all = train_model(ae_model, gantt_data_loader_train, gantt_dataset_test, pad, epochs, lr, recon_imgs_dir,checkpoint_dir)
        draw_training_curves(rec_loss_all, test_loss_all, "loss", epochs, recon_imgs_dir)


if __name__ == "__main__":
    # execute only if run as a script
    main()