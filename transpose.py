import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layer(x)
