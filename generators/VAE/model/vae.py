import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['G']['use_bias']

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


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
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

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1,
                      bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(512, self.z_size, bias=True)
        self.std_layer = nn.Linear(512, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar