import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST



def kerras_boundaries(sigma, eps, N, T):
    # This will be used to generate the boundaries for the time discretization

    return torch.tensor(
        [
            (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
            ** sigma
            for i in range(N)
        ]
    )


def mnist_dl(batch_size=128, num_workers=4):
    tf = T.Compose(
        [
            T.Pad(2),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5)),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def cifar10_dl(batch_size=128, num_workers=4):
    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

def fmnist_dl(batch_size=128, num_workers=4):
    tf = T.Compose(
            [
                T.Pad(2),
                T.ToTensor(),
                T.Nromalize(0.5, 0.5),
            ]
        )
    
    dataset = FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=tf,
            )

    dataloader = DataLoader(datasetm, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_data(dataset="mnist", batch_size=128, num_workers=4):
    if dataset == "mnist":
        return mnist_dl(batch_size, num_workers)
    elif dataset == "cifar10":
        return cifar10_dl(batch_size, num_workers)
    elif dataset == "fmnist":
        return fmnist_dl(batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
