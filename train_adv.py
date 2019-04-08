import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data as data
from network import ResNet, ResidualBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(root='./data/',
                                                  train=True,
                                                  transform=transform,
                                                  download=True)

test_dataset = torchvision.datasets.FashionMNIST(root='./data/',
                                                 train=False,
                                                 transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


def load_img(img_path):
    img = Image.open(img_path)
    img_data = transforms.ToTensor()(img)
    return img_data.to(device)


class mydatasets(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __getitem__(self, idx):
        img, target = self.images[idx], self.labels[idx]
        return img, target
    
    def __len__(self):
        return len(self.images)


model = torch.load('resnet.pkl')


# train model again with adversiral data
def train_adv():
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # prepare
    learning_rate = 0.0001
    num_epochs = 20
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # build my train data
    images = []
    labels = []
    with open('./adv_data.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_path, label = line.strip().split(' ')
        img = load_img(img_path)
        label = int(label)
        label = torch.LongTensor([label]).to(device)
        images.append(img)
        labels.append(label)
    mydata = mydatasets(images, labels)
    mydata_loader = data.DataLoader(dataset=mydata, batch_size=1, shuffle=False)
    # train
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterian(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                for img, label in mydata_loader:
                    output = model(img)
                    loss = criterian(output, label[0])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Decay learning rate
        if (epoch + 1) % 5 == 0:
            curr_lr /= 2
            update_lr(optimizer, curr_lr)
    
    # save model
    torch.save(model.state_dict(), 'model_Att.pkl')


def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    train_adv()
    test()

