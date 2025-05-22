import torch.nn as nn
import torch


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
    

class UNetGenerator(nn.Module):
    """U-net generator architecture with skip connections"""
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encoder layers
        self.e1 = ConvBlock(in_channels, 64, use_batchnorm=False)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.e5 = ConvBlock(512, 512)
        self.e6 = ConvBlock(512, 512)
        self.e7 = ConvBlock(512, 512)
        self.e8 = ConvBlock(512, 512)

        # Decoder layers
        self.d1 = DropoutConvBlock(512, 512)
        self.d2 = DropoutConvBlock(512*2, 512)
        self.d3 = DropoutConvBlock(512*2, 512)
        self.d4 = ConvBlock(512*2, 512, is_encoder=False)
        self.d5 = ConvBlock(512*2, 256, is_encoder=False)
        self.d6 = ConvBlock(256*2, 128, is_encoder=False)
        self.d7 = ConvBlock(128*2, 64, is_encoder=False)
        
        # Output layer:
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(64*2, out_channels, kernel_size=(4,4), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        e6_out = self.e6(e5_out)
        e7_out = self.e7(e6_out)
        e8_out = self.e8(e7_out)

        d1_out = self.d1(e8_out)
        d2_out = self.d2(torch.cat([d1_out, e7_out], dim=1))
        d3_out = self.d3(torch.cat([d2_out, e6_out], dim=1))
        d4_out = self.d4(torch.cat([d3_out, e5_out], dim=1))
        d5_out = self.d5(torch.cat([d4_out, e4_out], dim=1))
        d6_out = self.d6(torch.cat([d5_out, e3_out], dim=1))
        d7_out = self.d7(torch.cat([d6_out, e2_out], dim=1))

        output = self.output_layer(torch.cat([d7_out, e1_out], dim=1))
        return output
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(4,4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=(4,4), stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)