## Source: https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif

import argparse
import imageio
import glob

parser = argparse.ArgumentParser(description = 'Deep Convolutional GAN')
parser.add_argument('--dataset', required = True, help = 'MNIST | Fashion MNIST | CIFAR 10 | ImageNet')

# Parse all the arguments
args = parser.parse_args()

anim_file = 'results/dcgan_{}.gif'.format(args.dataset)

with imageio.get_writer(anim_file, mode = 'I') as writer:
    filenames = glob.glob('results/fake*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames[:8]):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
