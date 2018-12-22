import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import *
import torch.optim as optim
import random
from   torch.autograd import Variable


learning_rate = .0001
batch_size = 64
num_epochs = 100

def imshow(img,unnormalize = True):
    if unnormalize:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



#Load in the data from the testing and training datasets
def load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.MNIST('./train/', train =True, transform=transform, target_transform=None, download=True)
    testset = torchvision.datasets.MNIST('./test/', train=False, transform=transform, target_transform=None, download=True)


    return trainset, testset

def train(model,trainloader,learning_rate,num_epochs):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        print('Epoch Number: ', epoch)

        for i, (images, _) in enumerate(trainloader):
            output,code = model(images)

            optimizer.zero_grad()
            loss = criterion(output,images)

            print('Loss at ',i,': ',loss)

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'autoencoder.pth')


def test(model, testloader):

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        output, code = model(images)

    print(output[0].shape)
    imshow(torchvision.utils.make_grid(output),False)
    imshow(torchvision.utils.make_grid(images))

def main():

    #Process the data and create the train and testloaders
    trainset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=0)




    model = AutoEncoder()
    #train(model,trainloader,learning_rate,num_epochs)

    model.load_state_dict(torch.load('autoencoder.pth'))
    model.eval()

    test(model,testloader)





if __name__ == '__main__':
    main()
