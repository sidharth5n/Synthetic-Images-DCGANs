import torch
import torch.nn as nn

# Use batch normalization in all the layers
# Use ReLU activation upto penultimate layer
# Use tanh activation for last layer
class Generator(nn.Module):
    def __init__(self, feature_size, num_channels):
        super(Generator, self).__init__()
        N = 64
        self.feature_size = feature_size
        # input is random noise of size feature_size
        # Transposed convolution because we are upsampling
        self.main = nn.Sequential(nn.ConvTranspose2d(in_channels = feature_size,
                                                     out_channels = N * 8,
                                                     kernel_size = 4,
                                                     stride = 1,
                                                     padding = 0,
                                                     bias = False),
                                  nn.BatchNorm2d(N * 8),
                                  nn.ReLU(True),
                                  # (N*8) x 4 x 4
                                  nn.ConvTranspose2d(in_channels = N * 8,
                                                     out_channels = N * 4,
                                                     kernel_size = 4,
                                                     stride = 2,
                                                     padding = 1,
                                                     bias = False),
                                  nn.BatchNorm2d(N * 4),
                                  nn.ReLU(True),
                                  # (N*4) x 8 x 8
                                  nn.ConvTranspose2d(in_channels = N * 4,
                                                     out_channels = N * 2,
                                                     kernel_size = 4,
                                                     stride = 2,
                                                     padding = 1,
                                                     bias = False),
                                  nn.BatchNorm2d(N * 2),
                                  nn.ReLU(True),
                                  # (N*2) x 16 x 16
                                  nn.ConvTranspose2d(in_channels = N * 2,
                                                     out_channels = N,
                                                     kernel_size = 4,
                                                     stride = 2,
                                                     padding = 1,
                                                     bias = False),
                                  nn.BatchNorm2d(N),
                                  nn.ReLU(True),
                                  # (N) x 32 x 32
                                  nn.ConvTranspose2d(in_channels = N,
                                                     out_channels = num_channels,
                                                     kernel_size = 4,
                                                     stride = 2,
                                                     padding = 1,
                                                     bias = False),
                                  nn.Tanh()
                                  # (num_channels) x 64 x 64
                                  )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output

# Use batch normalization in all the layers
# Use leaky relu activation
class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        N = 64
        # input is (num_channels) x 64 x 64
        self.main = nn.Sequential(nn.Conv2d(in_channels = num_channels,
                                            out_channels = N,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 1,
                                            bias = False),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  # (N) x 32 x 32
                                  nn.Conv2d(in_channels = N,
                                            out_channels = N * 2,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 1, bias = False),
                                  nn.BatchNorm2d(N * 2),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  # (N*2) x 16 x 16
                                  nn.Conv2d(in_channels = N * 2,
                                            out_channels = N * 4,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 1,
                                            bias = False),
                                  nn.BatchNorm2d(N * 4),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  # (N*4) x 8 x 8
                                  nn.Conv2d(in_channels = N * 4,
                                            out_channels = N * 8,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 1,
                                            bias = False),
                                  nn.BatchNorm2d(N * 8),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  # (N*8) x 4 x 4
                                  nn.Conv2d(in_channels = N * 8,
                                            out_channels = 1,
                                            kernel_size = 4,
                                            stride = 1,
                                            padding = 0,
                                            bias = False),
                                  nn.Sigmoid()
                                  )
                                  # 1 x 1 x 1
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def weights_init(model):
    """
    Initialize weights of Conv and BatchNorm layers from a normal distribution.
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
