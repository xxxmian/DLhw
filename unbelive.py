import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from network import ResNet, ResidualBlock

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('resnet.pkl')

print("model is loaded")

to_pil_image = transforms.ToPILImage()


def image_loader(image_name):
    image = Image.open(image_name)
    image = transforms.ToTensor()(image)
    return image.to(device)




if __name__ == '__main__':
    img1 = image_loader('./test/attimg13.jpg')
    img1 = img1.view((1,1,28,28))
    output1 = model(img1)
    prd1 = torch.argmax(output1)
    img2 = image_loader('./test/orimg13.jpg')
    img2 = img2.view((1, 1, 28, 28))
    output2 = model(img2)
    prd2 = torch.argmax(output2)
    print(prd1,prd2)