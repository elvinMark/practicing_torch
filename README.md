# Practicing Deep Learning with PyTorch

## About
These are some simple programs to test different neural network architectures with different trainings and different datsets (MNIST, CIFAR10 and CIFAR100).

## Requirements
- PyTorch (>=1.8)
- wandb
- argparse


## How to use
The most basic way to train a MODEL (which can be: mlp, cnn or resnet) with a DATASET (which can be MNIST, CIFAR10 or CIFAR100) is with the following command line
```
python3 train.py \
--dataset DATASET_NAME \
--model MODEL_NAME
```

if you have alread an account in wandb and you have it confugure in your environment then you can should the project and experiment arguments as well. 
```
python3 train.py \
--dataset DATASET_NAME \
--model MODEL_NAME \
--project PROJECT_NAME \
--experiment EXPERIMENT_NAME
```

