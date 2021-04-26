from __future__ import print_function, division
from java import jclass
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
from os.path import dirname, join
import layout_output_sender
import os.path
from os import path
import io

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#for Testing
def greet(name):
    print("--- hello,%s ---" % name)

def get_java_bean():
    JavaBean = jclass("com.example.chaquopy_experiment.JavaBean")
    jb = JavaBean("python")
    jb.setData("json")
    jb.setData("xml")
    jb.setData("xhtml")
    return jb

# The following is the dataset class, which will be used to establish the PyTorch dataset object.
class EmotionFaceTrainDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.emotionface_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.emotionface_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.emotionface_info.iloc[idx, 2])
        # image = io.imread(img_name)
        image = Image.open(img_name)
        emotion = self.emotionface_info.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image, emotion

#The following is the network class
class AlexNet_Mobile(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super(AlexNet_Mobile, self).__init__()
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

    def first_process_layer1(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.cutlayer(x)
        return x

    def first_process_layer2(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cutlayer(x)
        return x

    def first_process_layer3(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cutlayer(x)
        return x

    def first_process_layer4(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cutlayer(x)
        return x

    def first_process_layer5(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.cutlayer(x)
        return x

    def first_process_layer6(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.cutlayer(x)
        return x

    def first_process_layer7(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.cutlayer(x)
        return x

    def first_process_layer8(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.cutlayer(x)
        return x

    def inference(self,x: torch.Tensor):
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

    global first_process_option
    first_process_option = {
        1: first_process_layer1,
        2: first_process_layer2,
        3: first_process_layer3,
        4: first_process_layer4,
        5: first_process_layer5,
        6: first_process_layer6,
        7: first_process_layer7,
        8: first_process_layer8,
        9: inference
    }

    def forward(self, x: torch.Tensor, layer_no) -> torch.Tensor:
        x = first_process_option[layer_no](self,x)
        return x


# Implement the train model
def train_in_mobile(trainloader):
    training_log = ""
    device = torch.device("cpu")
    net = AlexNet_Mobile()

    if (path.exists(join(dirname(__file__), "client_new.pkl")) == True):
        net.load_state_dict(torch.load(join(dirname(__file__), "client_new.pkl"), map_location=device))


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)



    net.to(device)

    layer_no = 3

    #print("Start Training - Mobile")
    training_log = training_log + "Start Training - Mobile\n"

    num_epochs = 5

    for epoch in range(num_epochs):

        #print("Epochs : {epoch}".format(epoch=epoch))
        training_log = training_log + "Epochs : {epoch}".format(epoch=epoch+1) + "\n"

        start_time = datetime.now()
        #print("Start Time: ", start_time.strftime("%H:%M:%S"))
        training_log = training_log + "Start Time: " + start_time.strftime("%H:%M:%S") + "\n"

        for i, data in enumerate(trainloader):
            print(i)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            #print("Test2")

            outputs_mobile = net(inputs, layer_no)

            #print("Test3")

            # Modify this line
            #server_grad = train_in_server(outputs_mobile, labels, layer_no)
            server_reply = layout_output_sender.transmit_layeout(outputs_mobile, labels, layer_no)

            server_grad = server_reply.get_layer_grad().to(device)

            #print(len(server_reply.get_layer_grad()))

            outputs_mobile.backward(gradient=server_grad, retain_graph=True)

            optimizer.step()

            #print("Current Batch Complete.")

        end_time = datetime.now()
        #print("Server CrossEntropyLoss:", server_reply.get_loss_message())
        training_log = training_log + "Server CrossEntropyLoss:" + server_reply.get_loss_message() + "\n"
        #print("End Time: ", end_time.strftime("%H:%M:%S"))
        training_log = training_log + "End Time: " + end_time.strftime("%H:%M:%S") + "\n"
        #print("Total Time: ", end_time - start_time)
        training_log = training_log + "Total Time: " + str(end_time - start_time) + "\n\n\n"
    return training_log



def execute_train(csv_file_path):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,],std=[0.5,])
    ])


    face_dataset = EmotionFaceTrainDataset(csv_file=join(dirname(__file__), csv_file_path),
                                           root_dir=dirname(__file__), transform=transform)

    train_set_size = int(len(face_dataset) * 0.8)
    valid_set_size = len(face_dataset) - train_set_size
    train_set, valid_set = data.random_split(face_dataset, [train_set_size, valid_set_size])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)


    training_log = train_in_mobile(trainloader)
    return training_log







