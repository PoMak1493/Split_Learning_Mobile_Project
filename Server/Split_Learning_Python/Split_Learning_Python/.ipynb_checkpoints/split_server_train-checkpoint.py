from __future__ import print_function, division
import os
import random
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from datetime import datetime

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class AlexNet_Server(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super(AlexNet_Server, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, 1e-4, 0.75)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, 1e-4, 0.75)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.cutlayer = nn.Sequential(
            nn.Identity()
        )


    def last_process_layer1(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def last_process_layer2(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def last_process_layer3(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def last_process_layer4(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def last_process_layer5(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def last_process_layer6(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def last_process_layer7(self, x: torch.Tensor):
        x = self.cutlayer(x)
        x = self.layer8(x)
        return x

    def last_process_layer8(self, x: torch.Tensor):
        x = self.cutlayer(x)
        return x

    def inference(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    global last_process_option
    last_process_option = {
        1: last_process_layer1,
        2: last_process_layer2,
        3: last_process_layer3,
        4: last_process_layer4,
        5: last_process_layer5,
        6: last_process_layer6,
        7: last_process_layer7,
        8: last_process_layer8,
        9: inference
    }

    def forward(self, x: torch.Tensor, layer_no) -> torch.Tensor:
        x = last_process_option[layer_no](self, x)
        return x

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()








