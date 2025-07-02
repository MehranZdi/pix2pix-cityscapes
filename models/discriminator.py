import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator. Evaluates realism of (input, output) image pairs.

        Input:
            - Shape: [B, C, H, W]
            - Description: The concatenation of the input label image and the generated or real RGB image along the channel dimension.
            For example, with 3-channel inputs and outputs, C = 6 (3 + 3).

        Output:
            - Shape: [B, 1, H/16, W/16]
            - Description: A feature map of discrimination scores for overlapping 70×70 patches in the image.
            Each value represents the discriminator's confidence that the corresponding patch is real (closer to 1) or fake (closer to 0).
    """

    def __init__(self, in_channels=6):  # 3 (input) + 3 (target) = 6
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)