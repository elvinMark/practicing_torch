import torch
import torch.nn as nn
import argparse

from models import create_model
from datasets import create_dataloaders
from optimizers import create_optimizer
from schedulers import create_lr_scheduler
from utils import train

parser =argparse.ArgumentParser(description="train helper")
parser.add_argument("--dataset",type=str,default="MNIST",choices=["MNIST","CIFAR10"],help="choose the dataset to train")
parser.add_argument("--model",type=str,default="mlp",choices=["mlp","cnn","resnet"],help="choose what type of model to be used")
parser.add_argument("--batch-size",type=int,default=128,help="batch size used for training")
parser.add_argument("--project",type=str,default="project",help="name of the project")
parser.add_argument("--experiment",type=str,default="experiment",help="name of the experiment")
parser.add_argument("--lr",type=float,default=0.1,help="learning rate")
parser.add_argument("--epochs",type=int,default=50,help="number of epochs used for the training")
parser.add_argument("--optim",type=str,default="sgd",help="specify the optimizer to be used in training")
parser.add_argument("--sched",type=str,default="step",help="specify the type of scheduler to be used")
parser.add_argument("--step-size",type=int,default=50,help="step size used in the StepLR scheduler")
parser.add_argument("--gamma",type=float,default=0.2,help="gamma factor used in the StepLR scheduler")
parser.add_argument("--T_max",type=float,default=200,help="T_max factor used in CosineAnnealingLR scheduler")
parser.add_argument("--eta_min",type=float,default=0.,help="eta_min factor used in CosineAnnealingLR scheduler")

args = parser.parse_args()
args.T_max = args.epochs

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    
model = create_model(args.dataset,args.model).to(dev)
train_dl, test_dl = create_dataloaders(args.dataset,args.batch_size)

optim = create_optimizer(model,args)
sched = create_lr_scheduler(optim,args)
crit = nn.CrossEntropyLoss()

train(model,train_dl,test_dl,optim,sched,crit,dev,epochs=args.epochs,project=args.project,experiment=args.experiment)
