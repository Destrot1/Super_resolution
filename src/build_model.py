import torch.nn as nn

class ResidualSRNet(nn.Module):
    """
    num_channels: n_feature_maps == cost. for each layer
    padding = 1 --> keep same dimensions
    """
    def __init__(self, num_channels, num_layers): 
        super().__init__()
        layers = []

        # first layer: input -> num_channels
        layers.append(nn.Conv2d(1, num_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True)) # ReLU with inplace=True overwrites the input tensor to save memory

        # hidden layers
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        # last layer: num_channels -> 1 (residual)
        layers.append(nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)) # not ReLU: residual could be > or < 0

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # predict residual
        residual = self.net(x)
        out = x + residual
        return out