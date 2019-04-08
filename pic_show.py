import matplotlib.pyplot as plt
from PIL import Image
def show_one_picture(path):

    for i in range(len(path)):
        plt.subplot(2,5,1+i)
        plt.title('origin label %c' % path[i][-5])
        plt.imshow(Image.open(path[i]))
    plt.show()
path =[]
for i in range(10):
    if i in(3,5,9):
        continue
    t = './submit_b_p/attimg%d.jpg' % i
    path.append(t)
show_one_picture(path)

path =[]
for i in range(10):
    if i in(3,5,9):
        continue
    t = './submit_b_p/orimg%d.jpg' % i
    path.append(t)
show_one_picture(path)