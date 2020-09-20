import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader import get_dataset
from models import Generator, Discriminator
from train import train

parser = argparse.ArgumentParser(description = 'Deep Convolutional GAN')
# parser.add_argument('--dataset', required = True, help = 'MNIST | Fashion MNIST | ImageNet')
parser.add_argument('--dataset', default = 'ImageNet', help = 'MNIST | Fashion MNIST | ImageNet')
parser.add_argument('--batch_size', type = int, default = 32, help = 'Input batch size')
parser.add_argument('--beta1', type = float, default = 0.5, help = 'Beta 1 for Adam optimizer')
parser.add_argument('--lr', type = float, default = 0.0002, help = 'Learning rate')
parser.add_argument('--epochs', type = int, default = 10, help = 'Number of iterations to train')
parser.add_argument('--feature_size', type = int, default = 100, help = 'Size of random noise')

# Parse all the arguments
args = parser.parse_args()

# Get dataset and data loader
dataset, num_channels = get_dataset(args.dataset)
data_loader = DataLoader(dataset,
                        batch_size = args.batch_size,
                        shuffle = True,
                        num_workers = 2)

# Check whether GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate generator and discriminator
generator = Generator(args.feature_size, num_channels).to(device)
discriminator = Discriminator(num_channels = num_channels).to(device)

# Select loss function and optimizer
loss_fn = torch.nn.BCELoss()
optimizer_d = optim.Adam(discriminator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

# Train the model
train(generator, discriminator, data_loader, loss_fn, optimizer_g, optimizer_d, device, epochs = args.epochs)
