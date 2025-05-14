import torch.nn as nn

class ConvBlock(nn.Module):
    """Base convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, use_batchnorm=True, is_encoder=True):
        super(ConvBlock, self).__init__()

        if is_encoder:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(4,4), stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4,4), stride=2, padding=1)
        
        layers = [self.conv]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_encoder:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DropoutConvBlock(nn.Module):
    """Convolutional block with BatchNrom, Dropout and ReLU for decoder"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DropoutConvBlock, self).__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)
    