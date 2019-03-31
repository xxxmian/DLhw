import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, num_classes)
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model = torch.load('resnet.pkl')
print("model is loaded")
#test_dataset = torchvision.datasets.FashionMNIST(root='./data/',train=False,transform=transforms.ToTensor)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
test_datasets = torchvision.datasets.FashionMNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())
test_loader_ = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=1,
                                          shuffle=True)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
learning_rate = 0.001
to_pil_image = transforms.ToPILImage()




def addinterference():
    choosen = []
    num_choosen = 0
    # choose 1000 correct classified images
    for image, label in test_loader_:
        image = Variable(image.to(device))
        image.requires_grad = True
        label = label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if num_choosen >= 1000:
            break
        if predicted == label:
            num_choosen += 1
            choosen.append((image, label))
    
    num_tointerrupted = 0
    toattactimg = []
    time_bound = 7
    
    for image, label in  choosen:
        # fix the label
        orimage = image[:]
        orlabel = label[:]
        label = label.to(device)  # change the original label to a new false one
        label = (label.data + 1) % 10
        label = label.detach()
        img_mean = torch.mean(image)
        image_var = torch.sum((image - img_mean) ** 2) / (image.shape[-1] * image.shape[-2])
        img_std = (torch.sqrt(image_var) / 8).expand_as(image)
        time = 0
        while time<time_bound:  # do until the image was classified to the new false one
            time+=1
            # add interferrence
            image_ = torch.normal(image, img_std)
            image_= torch.clamp(image_,min=0,max=1)
            # choose a u randomly between [0,1]
            u = torch.rand(1)
            # feed x' to my network,we got c(x')[argmax y'] that is the probabilty of y' (the label we persume)
            image_ = image_.to(device)
            output = model(image_)
            """norm_output = output-torch.min(output)
            norm_output=norm_output/torch.sum(norm_output)
            pp = norm_output[0][label]"""
            predict = torch.argmax(output)
            # distance between image and image_
            dist = torch.sum((orimage-image_)**2)/(image.shape[-1]*image.shape[-2])
            delta_dist = image_var/4
            if dist>delta_dist:
                continue
            if predict == label:
                toattactimg.append((image_, label, orimage, orlabel))
                num_tointerrupted+=1
                break
            image = image_[:]
    print(num_tointerrupted)
    cnt = set()
    for image, label,orimage,orlabel in toattactimg:
        output=model(image)
        predict = torch.argmax(output)
        if predict==label:
            cnt.add(label.item())
    print cnt
    printimg(toattactimg)
    
def printimg(img):
    number = 0
    for image, label, orimage,orlabel in img:
        number += 1
        image = image.cpu().squeeze(0)
        img = to_pil_image(image[:])
        img.convert('RGB')
        img.save('./black_attack_picsave/att/attimg%d.jpg' %(number))

        orimage = orimage.cpu().squeeze(0)
        img = to_pil_image(orimage[:])
        img.convert('RGB')
        img.save('./black_attack_picsave/ori/orimg%d.jpg' % (number))
        
addinterference()
# printimg()