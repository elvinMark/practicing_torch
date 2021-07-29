import torchvision
import torch

ROOT_PATH = "../data/"

datasets_dict = {
    "MNIST" : {
        "loader" : torchvision.datasets.MNIST,
        "transform" : torchvision.transforms.ToTensor()
    },
    "CIFAR10": {
        "loader" : torchvision.datasets.CIFAR10,
        "transform" : torchvision.transforms.ToTensor()
    }
}

def create_dataloaders(dataset_name,batch_size):
    train_ds = datasets_dict[dataset_name]["loader"](ROOT_PATH,train=True,download=True,transform=datasets_dict[dataset_name]["transform"])
    test_ds = datasets_dict[dataset_name]["loader"](ROOT_PATH,train=False,download=True,transform=datasets_dict[dataset_name]["transform"])

    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds,batch_size=batch_size)
        
    return train_dl, test_dl
