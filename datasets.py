import torchvision
import torch

ROOT_PATH = "../data/"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

datasets_dict = {
    "MNIST" : {
        "loader" : torchvision.datasets.MNIST,
        "transform" : torchvision.transforms.ToTensor(),
        "test_transform" : torchvision.transforms.ToTensor()
    },
    "CIFAR10": {
        "loader" : torchvision.datasets.CIFAR10,
        "transform" : torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32,padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
        ]),

        "test_transform" : torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
        ])
    }
}

def create_dataloaders(dataset_name,batch_size):
    train_ds = datasets_dict[dataset_name]["loader"](ROOT_PATH,train=True,download=True,transform=datasets_dict[dataset_name]["transform"])
    test_ds = datasets_dict[dataset_name]["loader"](ROOT_PATH,train=False,download=True,transform=datasets_dict[dataset_name]["test_transform"])

    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds,batch_size=batch_size)
        
    return train_dl, test_dl
