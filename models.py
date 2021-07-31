import torch
import torch.nn as nn

class Reshape(nn.Module):
    """
    This class is for reshaping tensors with an specific shape.
    """

    def __init__(self,new_shape):
        super().__init__()
        self.new_shape = new_shape
    
    def forward(self,x):
        return x.view(self.new_shape)


class ResBasicBLock(nn.Module):
    """
    This class is the building block for residual networks
    """
    
    def __init__(self,in_channel,out_channel,stride=1):
        super().__init__()
        self.straight = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride=stride,bias=False,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,stride=1,bias=False,padding=1),
            nn.BatchNorm2d(out_channel)
        )

        if in_channel != out_channel or stride!=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,stride=stride),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.straight(x) + self.shortcut(x))

def create_mlp_model(ni,nh,no):
    """
    This function create a simple 3 Layer MLP
    with ReLU as activation functions

    ni: number of neurons in the input layer
    nh: number of neurons in the hidden layer
    no: number of neurons in the output layer
    """
    
    return nn.Sequential(
        nn.Linear(ni,nh),
        nn.ReLU(inplace=True),
        nn.Linear(nh,no)
    )

def create_cnn_model(in_channel,ni,nh,no):
    """
    This function creates a simple Convolutional Layer.

    in_channel: channels in the input image (1 in case of gray scale images or 3 in colored images)
    ni: number of neurons in the input layer of the classifier
    nh: number of neurons in the hidden layer of the classifier
    no: number of neurons in the output layer of the classifier
    """
    
    return nn.Sequential(
        nn.Conv2d(in_channel,8,3,stride=2,bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Conv2d(8,16,3,stride=2,bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        Reshape((-1,ni)),
        nn.Linear(ni,nh),
        nn.ReLU(inplace=True),
        nn.Linear(nh,no)
    )

def create_resnet_model(in_channel,ni,nh,no):
    """
    This function creates a simple Residual Neural Network
    
    in_channel: channels in the input image (1 in case of gray scale images or 3 in colored images)
    ni: number of neurons in the input layer of the classifier
    nh: number of neurons in the hidden layer of the classifier
    no: number of neurons in the output layer of the classifier
    """
    
    return nn.Sequential(
        nn.Conv2d(in_channel,8,3,stride=1,padding=1),
        ResBasicBLock(8,16,stride=2),
        ResBasicBLock(16,16),
        ResBasicBLock(16,32,stride=2),
        ResBasicBLock(32,32),
        Reshape((-1,ni)),
        nn.Linear(ni,nh),
        nn.ReLU(inplace=True),
        nn.Linear(nh,no)
    )

def create_generator_mlp_model():
    """
    
    """

create_model_dict = {"mlp":create_mlp_model,"cnn":create_cnn_model,"resnet":create_resnet_model}

model_dict = {
    "MNIST": {
        "mlp": {
            "ni" : 784,
            "nh" : 256,
            "no" : 10
        },
        "cnn": {
            "in_channel": 1,
            "ni" : 576,
            "nh" : 128,
            "no" : 10
        },
        "resnet": {
            "in_channel": 1,
            "ni" : 1568,
            "nh" : 64,
            "no" : 10
        }
    },
    "CIFAR10": {
        "mlp": {
            "ni" : 3072,
            "nh" : 256,
            "no" : 10
        },
        "cnn": {
            "in_channel": 3,
            "ni" : 784,
            "nh" : 128,
            "no" : 10
        },
        "resnet": {
            "in_channel": 3,
            "ni" : 2048,
            "nh" : 64,
            "no" : 10
        }
    }
}


def create_model(dataset_name,model_type):
    return create_model_dict[model_type](**model_dict[dataset_name][model_type])
