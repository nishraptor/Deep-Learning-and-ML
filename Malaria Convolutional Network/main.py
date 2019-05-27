import numpy as np
import torch
import torch.nn as nn
import torchvision
from MalariaDataset import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import *
import torch.optim as optim
import random
from   torch.autograd import Variable


learning_rate = .0001
batch_size = 64
num_epochs = 100

#Load in the data from the testing and training datasets
def load_data():

    dataset = MalariaDataset(csv_file='malaria.csv', transform=transforms.Compose([
                                               Rescale((100,100)),
                                               ToTensor()]))

    return dataset

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

    torch.save(model.state_dict(), 'malaria.pth')


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
    dataset = load_data()
    print(len(dataset))

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].size(), sample['label'])
        dataset.showimage(**sample)
        if i == 3:
            break



    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)

    #estloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True, num_workers=0)

    #model = AutoEncoder()
    #train(model,trainloader,learning_rate,num_epochs)

    #model.load_state_dict(torch.load('autoencoder.pth'))
    #model.eval()

    #test(model,testloader)

if __name__ == '__main__':
    main()
