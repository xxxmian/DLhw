
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
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

# frezzing part of grad
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        param.requires_grad = False

test_dataset = torchvision.datasets.FashionMNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
learning_rate = 0.005
to_pil_image = transforms.ToPILImage()

def printimg():
    number = 0
    for image, label in test_loader:
        number += 1
        image = image.reshape((1, 28, 28))
        img = to_pil_image(image[:])
        img.convert('RGB')
        img.save('./white_attack_picsave/tempimg%d.jpg' %(number))
        if number == 10:
            break
def addinterference():
    choosen = []
    num_choosen = 0
    # choose 1000 correct classified images
    for image, label in test_loader:
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
            
    num_interrupted = 0
    attactedimg = []
    time_bound = 10
    for image, label in choosen[0:10]:
        # fix the label
        orimage = image[:]
        orlabel = label[:]
        label = label.to(device)  # change the original label to a new false one
        label = (label.data + 6)%10
        label = label.detach()
        
        time = 0
        while time<time_bound:  # do until the image was classified to the new false one
            time+=1
            image = Variable(image.to(device), requires_grad=True)
            output = model(image)
            predicted = torch.argmax(output.data, 1)
            
            if predicted == label:  # interrupt success
                num_interrupted += 1
                attactedimg.append(((image,label),(orimage,orlabel)))
                
                out_image = image.cpu().squeeze(0)
                orimage = orimage.cpu().squeeze(0)
                img = to_pil_image(out_image[:])
                img2 = to_pil_image(orimage[:])
                print "save image"
                img = img.convert('RGB')
                img2 = img2.convert('RGB')
                print(img.mode)
                img.save('./white_attack_picsave/attackedIMG%d.jpg'%(num_interrupted))
                img2.save('./white_attack_picsave/originalIMG%d.jpg' %(num_interrupted))
                
                break
            else:  # add interferrence
                loss1 = criterion(output, label)
                loss1.backward()
                image_grad = image.grad
                predicted = Variable(predicted.type(torch.cuda.FloatTensor), requires_grad = True)
                templabel = Variable(label.type(torch.cuda.FloatTensor))
                loss2 = criterion2(predicted, templabel)
                loss2.backward()
                image_grad =image_grad + image.grad
                new_image = image - learning_rate*image_grad
                image = new_image[:]
    print(num_interrupted)
addinterference()
#printimg()