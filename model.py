import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    """Base convolution block with BatchNorm and ReLU"""

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
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DropoutConvBlock(nn.Module):
    """Convolutional block with BatchNorm, Dropout and ReLU for decoder"""

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
    """Self-Attention layer for U-Net"""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Query, Key, Value projections
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B, HW, C//
        key = self.key_conv(x).view(batch_size, -1, H * W)  # B, C//, HW
        value = self.value_conv(x).view(batch_size, -1, H * W)  # B, C, HW

        # Attention scores
        attention = torch.bmm(query, key)  # B, HW, HW
        attention = self.softmax(attention)

        # Weighted sum
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(batch_size, C, H, W)

        # Apply gamma and add skip connection
        out = self.gamma * out + x
        return out


class UNetGenerator(nn.Module):
    """U-net generator architecture with skip connections and attention"""

    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encoder layers
        self.e1 = ConvBlock(in_channels, 64, use_batchnorm=False)  # 512x256 -> 256x128
        self.e2 = ConvBlock(64, 128)  # 256x128 -> 128x64
        self.e3 = ConvBlock(128, 256)  # 128x64 -> 64x32
        self.e4 = ConvBlock(256, 512)  # 64x32 -> 32x16
        self.e5 = ConvBlock(512, 512)  # 32x16 -> 16x8
        self.e6 = ConvBlock(512, 512)  # 16x8 -> 8x4
        self.e7 = ConvBlock(512, 512)  # 8x4 -> 4x2
        self.e8 = ConvBlock(512, 512)  # 4x2 -> 2x1

        # Attention layer in encoder
        self.attn_e5 = SelfAttention(512)  # After e5 (16x8)

        # Decoder layers
        self.d1 = DropoutConvBlock(512, 512)  # 2x1 -> 4x2
        self.d2 = DropoutConvBlock(512 * 2, 512)  # 4x2 -> 8x4
        self.d3 = DropoutConvBlock(512 * 2, 512)  # 8x4 -> 16x8
        self.d4 = ConvBlock(512 * 2, 512, is_encoder=False)  # 16x8 -> 32x16
        self.d5 = ConvBlock(512 * 2, 256, is_encoder=False)  # 32x16 -> 64x32
        self.d6 = ConvBlock(256 * 2, 128, is_encoder=False)  # 64x32 -> 128x64
        self.d7 = ConvBlock(128 * 2, 64, is_encoder=False)  # 128x64 -> 256x128

        # Attention layer in decoder
        self.attn_d3 = SelfAttention(512)  # After d3 (16x8)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, out_channels, kernel_size=4, stride=2, padding=1),  # 256x128 -> 512x256
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        e1_out = self.e1(x)  # 512x256 -> 256x128
        e2_out = self.e2(e1_out)  # 256x128 -> 128x64
        e3_out = self.e3(e2_out)  # 128x64 -> 64x32
        e4_out = self.e4(e3_out)  # 64x32 -> 32x16
        e5_out = self.e5(e4_out)  # 32x16 -> 16x8
        e5_out = self.attn_e5(e5_out)  # Apply attention
        e6_out = self.e6(e5_out)  # 16x8 -> 8x4
        e7_out = self.e7(e6_out)  # 8x4 -> 4x2
        e8_out = self.e8(e7_out)  # 4x2 -> 2x1

        d1_out = self.d1(e8_out)  # 2x1 -> 4x2
        d2_out = self.d2(torch.cat([d1_out, e7_out], dim=1))  # 4x2 -> 8x4
        d3_out = self.d3(torch.cat([d2_out, e6_out], dim=1))  # 8x4 -> 16x8
        d3_out = self.attn_d3(d3_out)  # Apply attention
        d4_out = self.d4(torch.cat([d3_out, e5_out], dim=1))  # 16x8 -> 32x16
        d5_out = self.d5(torch.cat([d4_out, e4_out], dim=1))  # 32x16 -> 64x32
        d6_out = self.d6(torch.cat([d5_out, e3_out], dim=1))  # 64x32 -> 128x64
        d7_out = self.d7(torch.cat([d6_out, e2_out], dim=1))  # 128x64 -> 256x128

        output = self.output_layer(torch.cat([d7_out, e1_out], dim=1))  # 256x128 -> 512x256
        return output


class Discriminator(nn.Module):
    def __init__(self, in_channels=6):  # 3 (label) + 3 (image) = 6
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 512x256 -> 256x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 256x128 -> 128x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128x64 -> 64x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 64x32 -> 32x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)),  # 32x16 -> 31x15
        )

    def forward(self, x):
        return self.model(x)