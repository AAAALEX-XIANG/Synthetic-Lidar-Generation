from operator import xor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.prior_size = config["prior_size"]
        self.z_size = config['z_size']
        self.use_bias = config['model']['G']['use_bias']

        self.model = nn.Sequential(

            nn.Linear(in_features=self.prior_size, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=256, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=self.z_size, bias=self.use_bias),
        )

    def forward(self, input):
        output = self.model(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.z_size = config['z_size']
        
        self.model = nn.Sequential(

            nn.Linear(self.z_size, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.pc_encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=self.z_size, kernel_size=1,
                      bias=self.use_bias),
        )

    def forward(self, x):
        output = self.pc_encoder_conv(x)
        output = output.max(dim=2)[0]
        # output = self.pc_encoder_fc(output)
        return output

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['De']['use_bias']

        self.model = nn.Sequential(

            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=256, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=2048, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=2048 * 3, bias=self.use_bias),
        )

    def forward(self, z):
        output = self.model(z.squeeze())
        output = output.view(-1, 3, 2048)
        return output
