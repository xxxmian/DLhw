
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from network import ResNet, ResidualBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model =torch.load('resnet.pkl')
print("model is loaded")

# freezing part of grad
for name, param in model.named_parameters():
    if param.requires_grad:
        #print(name)
        param.requires_grad = False

test_dataset = torchvision.datasets.FashionMNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
learning_rate = 0.001
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
def whiteAttack():
    model.eval()
    f=open('adv_data_w.txt','w')
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
    time_bound = 20
    for image, label in choosen:
        # fix the label
        orimage = image[:]
        orlabel = label[:]
        label = label.to(device)  # change the original label to a new false one
        label = (label.data + 1)%10
        label = label.detach()
        
        time = 0
        pickedP=set()
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
                # print "save image"
                img = img.convert('RGB')
                img2 = img2.convert('RGB')
                img.save('./white_attack_picsave/attackImg/attackedIMG%d.jpg'%(num_interrupted))
                img2.save('./white_attack_picsave/originalImg/originalIMG%d.jpg' %(num_interrupted))
                f.write('./white_attack_picsave/attackImg/attackedIMG%d.jpg'%(num_interrupted)+' %d\n' % orlabel)
                if label not in pickedP:
                    pickedP.add(label)
                    img.save('./submit_w_p/Attimg%d.jpg'%(label))
                    img2.save('./submit_w_p/orimg%d.jpg'%(label))
              
                break
            else:  # add interferrence
                loss1 = criterion(output, label)
                loss1.backward()
                image_grad = image.grad
                predicted = Variable(predicted.type(torch.cuda.FloatTensor), requires_grad = True)
                templabel = Variable(label.type(torch.cuda.FloatTensor))
                loss2 = criterion2(predicted, templabel)
                loss2.backward()
                image_grad =image_grad #+ image.grad
                new_image = image - learning_rate*image_grad
                image = new_image[:]
    print('%d  pictures was attackted attacking rate :%d %%' % (num_interrupted,num_interrupted/10))
    f.close()
whiteAttack()
