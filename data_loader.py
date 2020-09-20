from torchvision import datasets, transforms

def get_dataset(name = 'MNIST'):

    if name == 'MNIST':
        print("Loading MNIST")
        dataset = datasets.MNIST(root = 'datasets/', train = True, download = True,
                          transform = transforms.Compose([transforms.Resize((64, 64)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5,), (0.5,)),])
                          )
        num_channels = 1

    elif name == 'Fashion MNIST':
        print("Loading Fashion MNIST")
        dataset = datasets.FashionMNIST(root = 'datasets/', train = True, download = True,
                          transform = transforms.Compose([transforms.Resize((64, 64)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5,), (0.5,)),])
                          )
        num_channels = 1

    elif name == 'ImageNet':
        print("Loading ImageNet")
        dataset = datasets.ImageNet(root = 'datasets/', train = True, download = False,
                                    transform = transforms.Compose([transforms.Resize((64, 64)),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                                         std = [0.229, 0.224, 0.225])
                                                                    ]))
        num_channels = 3

    return dataset, num_channels
