import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, features = 32):
        super(VAE, self).__init__()
        self.feature = features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=(2,3),stride=(2,2),padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features * 2, kernel_size=(2,3),stride=(2,2),padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.Conv2d(features * 2, features * 4, kernel_size=(2,2), stride=(2,2), padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.Conv2d(features * 4, features * 8, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
        )
        self.mean = nn.Linear(features * 8, features * 8)
        self.var = nn.Linear(features * 8, features * 8)
        self.restore = nn.Linear(features * 8, features * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(3,3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2,2), stride=(2,2), padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, features, kernel_size=(3,2), stride=(2,2), padding=(1,0)),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, 3, kernel_size=(3,2), stride=(2,2), padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )

    def forward(self, image):
        x = self.encoder(image)
        self.size = x.size()
        mean = self.mean(x.view(-1,self.feature * 8)).view(-1)
        var = self.var(x.view(-1,self.feature * 8)).view(-1)
        z = mean + torch.exp(var) * torch.randn_like(var)
        z = self.restore(z.view(-1,self.feature * 8)).reshape(self.size)
        z = self.decoder(z)
        return z, mean, var

    def initiate(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight.data)

    def generate(self, guassian):
        z = self.restore(guassian.view(-1, self.feature * 8)).reshape(self.size)
        z = self.decoder(z)
        return z