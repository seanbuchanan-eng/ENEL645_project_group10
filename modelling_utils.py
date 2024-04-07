"""
Models and functions for the loading, training, and evaluation 
of PyTorch models.
"""

#! /usr/bin/env python 
# ---------------------------------------------------------------------
import re
import torch
import torchvision
import collections
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets, models, transforms
import numpy as np
import os
# ---------------------------------------------------------------------

train_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # from pytorch documentation

agnet_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

agnet_highres_transforms = {
    "train": transforms.Compose([
        transforms.RandomCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.RandomCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.RandomCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

enet_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # rescales to [0.0, 1.0]
        transforms.Normalize(train_stats[0], train_stats[1])
    ]),
    "val": transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ]),
    "test": transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ])
}

enet_aug_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.RandAugment(),
        transforms.ToTensor(), # rescales to [0.0, 1.0]
        transforms.RandomErasing(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ]),
    "val": transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ]),
    "test": transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ])
}

swin_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(480, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # rescales to [0.0, 1.0]
        transforms.Normalize(train_stats[0], train_stats[1])
    ]),
    "val": transforms.Compose([
        transforms.RandomResizedCrop(480, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ]),
    "test": transforms.Compose([
        transforms.RandomResizedCrop(480, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(train_stats[0], train_stats[1])
    ])
}

class Swin(nn.Module):
    """
    PyTorch swin_v2_s model for image classification.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. 
                                     Defaults to True.
        full_params_file (str, optional): Path to the file containing the pre-trained parameters of the model.
                                          Defaults to None.
    """

    def __init__(self, pretrained=True, full_params_file=None):
        super().__init__()

        if pretrained:
            self.model = models.swin_v2_s(weights="IMAGENET1K_V1")
        else:
            self.model = models.swin_v2_s(weights=None)

        self.model = nn.Sequential(*([*self.model.children()][:-1]), nn.Linear(768, 7, bias=True))

        if full_params_file is not None:
            model_dict = replace_dict_keys(torch.load(full_params_file))
            self.model.load_state_dict(model_dict)

    def forward(self, x):
        return self.model(x)


class Enet(nn.Module):
    """
    PyTorch efficientnet_v2_m model for image classification.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. 
                                     Defaults to True.
        static (bool, optional): If True, the parameters of the model are frozen. 
                                 Defaults to False.
        full_params_file (str, optional): Path to the file containing the pre-trained parameters of the model.
                                          Defaults to None.
    """
    
    def __init__(self, pretrained=True, static=False, full_params_file=None):
        super().__init__()

        if pretrained:
            self.model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        if static:
            self.model.eval()
            for param in self.model.parameters():
               param.requires_grad = False
        else:
            self.model = models.efficientnet_v2_m(weights=None)

        cnn_out_size = [*self.model.classifier.modules()][-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(cnn_out_size, 7)
        )

        if full_params_file is not None:
            model_dict = replace_dict_keys(torch.load(full_params_file))
            self.model.load_state_dict(model_dict)

    def forward(self, x):
        return self.model(x)

class Resnet50AgVision(nn.Module):
    """
    PyTorch resnet50 model for image classification.

    Args:
        weights_path (str): Path to the pre-trained weights of the model.
        input_tensor (Tensor): Input tensor to determine number of parameters of final layer.
        full_params_file (str, optional): Path to the file containing the pre-trained parameters of the model.
                                          Defaults to None.
    """

    def __init__(self, weights_path, input_tensor, full_params_file=None):
        super().__init__()

        # create model
        self.model = models.resnet50(weights=None)
        self.model = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())

        # Load the saved weights from the model with four channels
        saved_state_dict = torch.load(weights_path)
        # Discard the extra channel in the weights (assuming the first channel needs to be discarded)
        saved_state_dict['0.weight'] = saved_state_dict['0.weight'][:, :-1, :, :]

        # Load the modified state_dict into the new model
        self.model.load_state_dict(saved_state_dict)

        # freeze params
        if full_params_file is None:
            for param in self.model.parameters():
                param.requires_grad = False

        # find shape of feature detector output
        flatten_size = self.model(input_tensor).shape[-1]

        # add new top to model
        self.model = nn.Sequential(
            *list(self.model.children()), 
            nn.Linear(flatten_size, 7)
            )
        
        if full_params_file is not None:
            model_dict = replace_dict_keys(torch.load(full_params_file))
            self.model.load_state_dict(model_dict)
        
    def forward(self, x):
        return self.model(x)

def train_validate(model, criterion, optimizer, save_dir, 
                param_file, dataloaders, device,
                dataset_sizes, result_file, 
                scheduler=None, num_epochs=25):
    """
    Train and validate a PyTorch model.

    Args:
        model (nn.Module): The model to be trained.
        criterion (torch.nn._Loss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        save_dir (str): Directory to save the trained model.
        param_file (str): File name to save the model parameters.
        dataloaders (dict): Dictionary containing data loaders for training, validation, and testing sets.
        device (str): Device to use for training.
        dataset_sizes (dict): Dictionary containing sizes of training, validation, and testing datasets.
        result_file (str): File to save training and validation results.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        num_epochs (int, optional): Number of epochs for training. Defaults to 25.

    Returns:
        nn.Module: Trained model.
    """

    result_dict = {
        "train": {
            "loss": [],
            "acc": []
        },
        "val": {
            "loss": [],
            "acc": []
        } 
    }

    tempdir = save_dir
    print(tempdir)
    best_model_params_path = os.path.join(tempdir, param_file)

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}{num_epochs - 1}')
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0

            for img, labels in dataloaders[phase]:
                img = img.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(img)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                path_split = str(best_model_params_path).strip().split(".")
                chkp_path = path_split[0] + "_checkpoint." + path_split[-1]
                save_chkp(checkpoint, chkp_path)

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            result_dict[phase]["loss"].append(epoch_loss)
            result_dict[phase]["acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
 
        print()

        np.save(save_dir + result_file, result_dict)
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(torch.load(best_model_params_path))

    return model

def test(model, dataloader, device):
    """
    Test a PyTorch model.

    Args:
        model (nn.Module): The model to be tested.
        dataloader (DataLoader): Data loader for the testing set.
        device (str): Device to use for testing.
    """
    print("Testing")

    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for img, labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)

            outputs = model(img)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Accuracy of the network on the test set: {100 * correct / total}%")

def calc_metrics(model, dataloader, device, classes=7):
    """
    Calculate evaluation metrics.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Data loader for the dataset.
        device (str): Device to use for evaluation.
        classes (int, optional): Number of classes. Defaults to 7.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    print("Calculating metrics")

    model.eval()

    total = 0
    correct = 0
    conf_matrix = torch.zeros(classes,classes)

    intermediates = {
        "total_1": 0,
        "total_2": 0,
        "total_3": 0,
        "total_4": 0,
        "total_5": 0,
        "total_6": 0,
        "total_7": 0,
        "correct_1": 0,
        "correct_2": 0,
        "correct_3": 0,
        "correct_4": 0,
        "correct_5": 0,
        "correct_6": 0,
        "correct_7": 0
    }

    with torch.no_grad():
        for img, labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)

            outputs = model(img)
            _, preds = torch.max(outputs, 1)

            for idx, label in enumerate(labels):
                conf_matrix[label, preds[idx]] += 1

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            for idx in range(classes):
                intermediates[f"total_{idx+1}"] += labels[(labels == idx)].size(0)
                intermediates[f"correct_{idx+1}"] += ((preds == labels) & (labels == idx)).sum().item()

    print(f"Accuracy: {100 * correct / total}")
    for idx in range(classes):
        print(f"Accuracy of class {idx}: {100 * intermediates[f'correct_{idx+1}'] / intermediates[f'total_{idx+1}']}")
    print(f"Confusion Matrix: {conf_matrix}")

    results = {"total_accuracy": 100 * correct / total}
    for idx in range(classes):
        results[f"accuracy{idx}"] = 100 * intermediates[f'correct_{idx+1}'] / intermediates[f'total_{idx+1}']
    results["conf_matrix"] = conf_matrix
    
    return results

def replace_dict_keys(model_dict):
    """
    Replace keys in a model params dictionary if they start
    with 'model.'.

    Args:
        model_dict (dict): Dictionary to be modified.

    Returns:
        collections.OrderedDict: Modified dictionary.
    """
    new_dict = collections.OrderedDict()
    for key in model_dict.keys():
        if key.startswith("model."):
            new_key = key.split("model.")[-1]
            new_dict[new_key] = model_dict[key]
        else:
            new_dict[key] = key

    return new_dict

def save_chkp(state, checkpoint_dir):
    """
    Save training checkpoint.

    Args:
        state (dict): State dictionary.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    torch.save(state, checkpoint_dir)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Load training checkpoint.

    Args:
        checkpoint_fpath (str): File path of the checkpoint.
        model (nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): Optimizer to load the checkpoint into.

    Returns:
        tuple: Model, optimizer, epoch.
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']