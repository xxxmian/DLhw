import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from network import ResNet, ResidualBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model.load_state_dict(torch.load('model_Att.pkl'))
print("model trained with attacked data is loaded")
# freezing part of grad
for name, param in model.named_parameters():
    if param.requires_grad:
        param.requires_grad = False

test_dataset = torchvision.datasets.FashionMNIST(root='./data/',
                                                 train=False,
                                                 transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
to_pil_image = transforms.ToPILImage()

def NwhiteAttack():
    model.eval()
    choosen = []
    num_choosen = 0
    # choose 1000 correct classified images
    for image, label in test_loader:
        image = Variable(image.to(device))
        #image.requires_grad = True
        label = label.to(device)
        output = model(image)
        predicted = torch.argmax(output)
        if num_choosen >= 1000:
            break
        if predicted == label:
            num_choosen += 1
            choosen.append((image, label))
    print('%d correct classified samples was choosen!' % num_choosen)
    
    num_att = 0
    time_bound = 20
    for image, label in choosen:
        orlabel = label[:]
        label = label.to(device)  # change the original label to a new false one
        label = (label.data + 1) % 10
        label = label.detach()
        
        time = 0
        while time < time_bound:  # do until the image was classified to the new false one
            time += 1
            image = Variable(image.to(device), requires_grad=True)
            output = model(image)
            predicted = torch.argmax(output.data, 1)
            
            if predicted == label:  # attack success
                num_att += 1
                break
            else:  # add att
                loss1 = criterion(output, label)
                loss1.backward()
                image_grad = image.grad
                image_grad = image_grad
                new_image = image - learning_rate * image_grad
                image = new_image[:]
    print('%d  pictures was attackted attacking rate :%d %%' % (num_att, num_att / 10))

NwhiteAttack()
