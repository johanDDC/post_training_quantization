import os

import torchvision.transforms as transforms
import torchvision.datasets as dsets

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


def get_train_augments(num_augments, magnitude):
    train_transforms = transforms.Compose([
        transforms.RandAugment(num_ops=num_augments, magnitude=magnitude),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),
    ])
    return train_transforms


def get_train_data(root_dir="data/", train_transforms=test_transforms, test_transforms=test_transforms):
    cifar10_train = dsets.CIFAR10(root=os.path.join(root_dir, "train"), train=True, transform=train_transforms,
                                  download=True)
    cifar10_test = dsets.CIFAR10(root=os.path.join(root_dir, "test"), train=False, transform=test_transforms,
                                 download=True)
    return cifar10_train, cifar10_test
