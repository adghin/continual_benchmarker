"""
Benchmarks of pre-trained pytorch models ***trained with continual*** methods on common continual learning datasets
@author: adrian.ghinea@outlook.it
"""
import pickle
import argparse
import importlib

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision as tv
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

def parseArgs():
    """
    Arguments to use: model_name, dataset_name, batch_size
    """
    parser = argparse.ArgumentParser(description="Benchmark of pre-trained models on common continual learning datasets")

    parser.add_argument("-m"     , "--model", type=str, help="Available models: resnet18, resnet34, resnet50, resnet101, resnet152, vit_b_16, vit_b_32, vit_l_16",required=True)
    parser.add_argument("-d"     , "--dataset", type=str, help="Available datasets: cifar10, cifar100, tinyimagenet",required=True)
    parser.add_argument("-b"     , "--batch_size", type=int, help="batch size for testing (default=32)",default=32)

    args = parser.parse_args()

    return args

def model():
    """
    Load pre-trained model with default weights from torchvision.models
    """
    args = parseArgs()
    model_name = args.model
    model_weights = "DEFAULT"
    model = models.get_model(model_name,weights=model_weights)

    #Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    #Print model's optimizer
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    return model

def dataset():
    """
    Load test dataset
    """
    args = parseArgs()
    dataset_name = args.dataset.upper()

    #Mammoth test transform for CIFAR10
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))])

    if hasattr(datasets,dataset_name):
        #For CIFAR-10 and CIFAR-100
        imported_dataset = getattr(datasets,dataset_name)
        test_dataset     = imported_dataset(root='./data',train=False,transform=test_transform,download=True)

    #Implement Tiny-IMGNET

    dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,drop_last=False)

    return dataloader

def evaluateModel(model):
    
    test_loader = dataset()
    model.eval()

    correct, total = 0, 0

    for data in test_loader:
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

    print("\n\nCorrect: " + str(correct))
    print("\nTotal: " + str(total))
    print('\nTest Accuracy: {:.2f}%'.format(100*correct/total))

def main(args=None):
    my_model = model()
    evaluateModel(my_model)

main()