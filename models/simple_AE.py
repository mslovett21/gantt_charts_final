import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

class Encoder(nn.Module):
    def __init__(self,deep_layer, shallow_layer, z_emb_size, features, var_stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, stride=1, padding = 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=5, stride=3, padding = 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=5, stride=3, padding = 0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5, stride=3, padding = 0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=features, kernel_size=5, stride=var_stride, padding = 0),
            nn.BatchNorm2d(num_features=features), 
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(deep_layer,shallow_layer), 
            nn.ReLU(),
            nn.Linear(shallow_layer, z_emb_size)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self,deep_layer, shallow_layer, z_emb_size, dim,features,var_stride):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(z_emb_size, shallow_layer),
            nn.ReLU(),
            nn.Linear(shallow_layer, deep_layer)
        )
        self.dim = dim
        self.features = features
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features= self.features),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels= self.features,out_channels=128, kernel_size=5, stride=var_stride, padding = 0),
            nn.BatchNorm2d(num_features= 128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=3, padding = 0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=3, padding = 0),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=3, padding = 0),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding = 0),
            nn.Sigmoid()
        )
    def forward(self, x):        
        return self.net(self.map(x).reshape(-1, self.features, self.dim, self.dim))

#-----------------------------------------------------------------------------------------------------------

# To test your encoder/decoder, let's encode/decode some sample images
# first, make a PyTorch DataLoader object to sample data batches
class AutoEncoder(nn.Module):
    
    def __init__(self,deep_layer, shallow_layer, z_emb_size,dim, features, var_stride):
        super().__init__()
        self.encoder = Encoder(deep_layer, shallow_layer, z_emb_size, features,var_stride)
        self.decoder = Decoder(deep_layer, shallow_layer, z_emb_size,dim, features,var_stride)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruct(self, x):
        """Only used later for visualization."""
        return self.forward(x)

    def embed(self, x):
        return self.encoder(x)

    def decode(self, emb):
        return self.decoder(emb)