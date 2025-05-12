import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, filter_size):
        super().__init__(Generator)

        self.encoder = nn.Sequential(
            self.encoder_conv_maker(3, 64),
            self.encoder_conv_maker(64, 128),
            self.encoder_conv_maker(128, 256),
            self.encoder_conv_maker(256, 512),
            self.encoder_conv_maker(512, 512),
            self.encoder_conv_maker(512, 512),
            self.encoder_conv_maker(512, 512),
            self.encoder_conv_maker(512, 512),
        )

        self.decoder = nn.Sequential(
            self.decoder_conv_maker(512, 512, True),
            self.decoder_conv_maker(512, 512, True),
            self.decoder_conv_maker(512, 512, True),
            self.decoder_conv_maker(512, 512),
            self.decoder_conv_maker(512, 256),
            self.decoder_conv_maker(256, 128),
            self.decoder_conv_maker(128, 64),
        )

    def encoder_conv_maker(self, in_channels, out_channels):
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(4,4), stride=2)
        batch_layer = nn.BatchNorm2d()(conv_layer)
        relu_layer = nn.ReLU()(batch_layer)
        return relu_layer
    

    def decoder_conv_maker(self, in_channels, out_channels, dropout=True):
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(4,4), stride=2)
        batch_layer = nn.BatchNorm2d()(conv_layer)
        if dropout:
            dropout_layer = nn.Dropout()(batch_layer)
            relu_layer = nn.ReLU()(dropout_layer)
        else:
            relu_layer = nn.ReLU()(batch_layer)

        return relu_layer
