import torch
import torch.nn as nn
import argparse

from models import create_model
from datasets import create_dataloaders
from utils import train

parser =argparse.ArgumentParser(description="train helper")
parser.add_argument("--dataset",type=str,default="MNIST",choices=["MNIST","CIFAR10"],help="choose the dataset to train")
parser.add_argument("--model",type=str,default="mlp",choices=["mlp","cnn","resnet"],help="choose what type of model to be used")
parser.add_argument("--batch-size",type=int,default=128,help="batch size used for training")
parser.add_argument("--project",type=str,default="project",help="name of the project")
parser.add_argument("--experiment",type=str,default="experiment",help="name of the experiment")
parser.add_argument("--lr",type=float,default=0.01,help="learning rate")
parser.add_argument("--epochs",type=int,default=50,help="number of epochs used for the training")

args = parser.parse_args()

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    
model = create_model(args.dataset,args.model).to(dev)
train_dl, test_dl = create_dataloaders(args.dataset,args.batch_size)

optim = torch.optim.Adam(model.parameters(),lr=args.lr)
crit = nn.CrossEntropyLoss()

train(model,train_dl,test_dl,optim,crit,dev,epochs=args.epochs,project=args.project,experiment=args.experiment,flatten=args.model=="mlp")
