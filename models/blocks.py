import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """ A class for convolution block with BatchNorm and activation.

        ...

        Attributes
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            use_batchnorm : bool, default True
                Whether to use BatchNorm or not.
            is_encoder : bool, default True
                Whether to use as an encoder layer or not.

        Methods
        -------
            forward(input):
                Performs forward pass.
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True, is_encoder=True):
        super(ConvBlock, self).__init__()

        if is_encoder:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        layers = [self.conv]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        if is_encoder:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DropoutConvBlock(nn.Module):
    """ A class for deconvolutional block with BatchNorm, Dropout and ReLU.

        ...

        Attributes
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            dropuot_rate : float
                Dropout rate.

        Methods
        -------
            forward(input):
            Performs forward pass.
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DropoutConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class SelfAttention(nn.Module):
    """ Simplified Self-Attention layer, proper for bottleneck blocks.

        ...

        Attributes
        ----------
            in_channels : int
            Number of input channels.

        Methods
        -------
            forward(input):
            performs forward pass.
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        # Reduce channels for computational efficiency
        self.query_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Downsample for efficiency if feature map is large
        if H * W > 1024:  # Skip attention for very large feature maps
            return x

        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        value = self.value_conv(x).view(batch_size, -1, H * W)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x
        return out
