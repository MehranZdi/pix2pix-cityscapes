from models.blocks import ConvBlock, DropoutConvBlock, SelfAttention
import torch.nn as nn
import torch

class UNetGenerator(nn.Module):
    """ Improved U-Net Generator with attention and skip connections.

            Input: [B, H, W, C] --> label(mask)
            Output: [B, H, W, C] --> realistic photo
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encoder layers
        self.e1_same = ConvBlock(in_channels, 64, use_batchnorm=False, kernel_size=3, stride=1, padding=1)  # 256x512 -> 256x512
        self.e1 = ConvBlock(64, 64, use_batchnorm=False) # 256x512 -> 128x256
        self.e2_same = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)  # 128x256 -> 128x256
        self.e2 = ConvBlock(128, 128)  # 128x256 -> 64x128
        self.e3_same = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)  # 64x128 -> 64x128
        self.e3 = ConvBlock(256, 256)  # 64x128 -> 32x64
        self.e4 = ConvBlock(256, 512)  # 32x64 -> 16x32
        self.e5 = ConvBlock(512, 512)  # 16x32 -> 8x16
        self.e6 = ConvBlock(512, 512)  # 8x16 -> 4x8
        self.e7 = ConvBlock(512, 512)  # 4x8 -> 2x4
        self.e8 = ConvBlock(512, 512)  # 2x4 -> 1x2

        # Attention layers - only in bottleneck
        self.attn_e5 = SelfAttention(512)  # At 8x16 resolution

        # Decoder layers
        self.d1 = DropoutConvBlock(512, 512)  # 2x1 -> 4x2
        self.d2 = DropoutConvBlock(512 * 2, 512)  # 4x2 -> 8x4
        self.d3 = DropoutConvBlock(512 * 2, 512)  # 8x4 -> 16x8
        self.d4 = ConvBlock(512 * 2, 512, is_encoder=False)  # 16x8 -> 32x16
        self.d5 = ConvBlock(512 * 2, 256, is_encoder=False)  # 32x16 -> 64x32
        self.d6 = ConvBlock(256 * 2, 128, is_encoder=False)  # 64x32 -> 128x64
        self.d7 = ConvBlock(128 * 2, 64, is_encoder=False)  # 128x64 -> 256x128

        # Output layer
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1_out = self.e1_same(x)
        e1_out = self.e1(e1_out)
        e2_out = self.e2_same(e1_out)
        e2_out = self.e2(e2_out)
        e3_out = self.e3_same(e2_out)
        e3_out = self.e3(e3_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        e5_out = self.attn_e5(e5_out)
        e6_out = self.e6(e5_out)
        e7_out = self.e7(e6_out)
        e8_out = self.e8(e7_out)

        # Decoder with skip connections
        d1_out = self.d1(e8_out)
        d2_out = self.d2(torch.cat([d1_out, e7_out], dim=1))
        d3_out = self.d3(torch.cat([d2_out, e6_out], dim=1))
        d4_out = self.d4(torch.cat([d3_out, e5_out], dim=1))
        d5_out = self.d5(torch.cat([d4_out, e4_out], dim=1))
        d6_out = self.d6(torch.cat([d5_out, e3_out], dim=1))
        d7_out = self.d7(torch.cat([d6_out, e2_out], dim=1))

        output = self.output_layer(torch.cat([d7_out, e1_out], dim=1))
        return output
