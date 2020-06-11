import glob, random
import numpy as np
from skimage.io import imread
from skimage import transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class attackDataset(Dataset):
    def __init__(self, data, attack_digit, target_digit, train=True):
        self.data = data
        self.attack_digit = attack_digit
        self.target_digit = target_digit
        self.train = train
        self.attack_digits = []
        self.target_digits = []
        # print(len(self.data))
        for i in range(len(self.data)):
            
            if self.data[i][1] == self.attack_digit:
                self.attack_digits.append(self.data[i][0])
            
            elif self.train and self.data[i][1] == self.target_digit:
                self.target_digits.append(self.data[i][0])
            
        # else:
        #     for i in range(len(self.data)):
        #         if self.data[i][1] == self.attack_digit:
        #             self.attack_digits.append(np.array(self.data[i][0]))


        # self.target_digits = torch.Tensor(self.target_digits)
        # self.attack_digits = torch.Tensor(self.attack_digits)
        # print(len(self.attack_digits))
        # print(len(self.target_digits))

    def __len__(self):
        # if self.train:
        #     return len(self.data) * len(self.target_digits)
        # else:
        #     return len(self.attack_digits)
        # if self.train:
        #     return len(self.data)
        # else:
        #     return len(self.attack_digits)
        return len(self.attack_digits)

    def __getitem__(self, index):
        # if self.train:
        #     i = index // len(self.target_digits)
        #     extra = index - i * len(self.target_digits)
        #     return self.attack_digits[i], self.target_digits[extra]
        # else:
        #     return self.attack_digits[index]
        # if self.train:
        #     attack = random.choice(self.attack_digits)
        #     target = random.choice(self.target_digits)
        #     return attack, target
        # else:
        #     return random.choice(self.attack_digits)
        return self.attack_digits[index]





def get_mnist_dataloaders_attack(attack_digit, target_digit, train_batch_size=128, test_batch_size=32, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    
    train_attack = attackDataset(train_data, attack_digit, target_digit)
    test_attack = attackDataset(test_data, attack_digit, target_digit, train=False)
    
    train_loader = DataLoader(train_attack, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_attack, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, test_loader

def get_mnist_dataloaders(batch_size=128, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_cifar_dataloaders(batch_size=128, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.CIFAR10(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.CIFAR10(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data='../fashion_data'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='../dsprites-data/dsprites_data.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='../rendered_chairs_64'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='../rendered_chairs_64_test'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images."""
    all_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = CelebADataset(path_to_data, transform=all_transforms, train=True)
    test_data = CelebADataset(path_to_data, transform=all_transforms, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                               shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                               shuffle=True)
    return train_loader, test_loader

def get_svhn_dataloader(batch_size=128, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.SVHN(path_to_data, download=True, split="train",
                                transform=all_transforms)
    test_data = datasets.SVHN(path_to_data, split="test", download=True,
                                transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                               shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                               shuffle=True)

    return train_loader, test_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None, train=True):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        if train:
            self.img_paths = glob.glob(path_to_data + '/Img/train/*')[::subsample]
        else:
            self.img_paths = glob.glob(path_to_data + '/Img/test/*')[::subsample]
        self.transform = transform
        with open(path_to_data+'/Anno/list_attr_celeba.txt') as f:
            data_raw = f.read()
        lines = data_raw.split("\n")

        keys = lines[1].strip().split()
        values = {}
        # Go over each line (skip the last one, as it is empty).
        for line in lines[2:-1]:
            row_values = line.strip().split()
            values[row_values[0]] = int(row_values[21])

        self.keys = keys
        self.annotations = values

        del keys
        del values

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        img_name = sample_path.split('/')[-1]
        sample = transform.resize(imread(sample_path), (64, 64))
        target = self.annotations[img_name]
        if target == -1:
            target = 0
        # print("{}:{}".format(img_name, self.annotations[img_name]))
        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, target