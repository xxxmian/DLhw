import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from network import ResNet,ResidualBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('resnet.pkl')
print("model is loaded")

# load test data
test_datasets = torchvision.datasets.FashionMNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())
test_loader_ = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=1,
                                          shuffle=True)

to_pil_image = transforms.ToPILImage()

def blackAttack():
    # choose 1000 correct classified images
    choosen = []
    num_choosen = 0
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
    time_bound = 40
    f=open('./adv_data.txt','w')
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
            # add interference
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
                num_tointerrupted += 1
                toattactimg.append((image_, label, orimage, orlabel))
                path1='./black_attack_picsave/att/attimg%d.jpg' % (num_tointerrupted)
                path2='./black_attack_picsave/ori/orimg%d.jpg' % (num_tointerrupted)
                save_img(image_, path1,orimage,path2)
                f.write(path1+' %d\n' % orlabel)
                
                break
            image = image_[:]
    print("%d images was attacktted" % num_tointerrupted)
    cnt = set()
    for image, label,orimage,orlabel in toattactimg:
        output=model(image)
        predict = torch.argmax(output)
        if predict==label:
            cnt.add(label.item())
    print cnt
    f.close()
    
def save_img(image,path1,orimage,path2):
    
    image = image.cpu().squeeze(0)
    img = to_pil_image(image[:])
    img.convert('RGB')
    img.save(path1)

    orimage = orimage.cpu().squeeze(0)
    img = to_pil_image(orimage[:])
    img.convert('RGB')
    img.save(path2)
        
blackAttack()
