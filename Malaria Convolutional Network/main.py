import numpy as np
import torch
import torch.nn as nn
import torchvision
from MalariaDataset import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
from PIL import Image
from model import *
import torch.optim as optim
import random
from   torch.autograd import Variable


learning_rate = .00001
batch_size = 128
num_epochs = 3

#Load in the data from the testing and training datasets
def load_data():

    #Create the dataset
    dataset = MalariaDataset(csv_file='malaria.csv', transform=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(.2 * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)


    return trainloader,testloader

def show_batch(images,labels):

        batch_size = len(images)
        im_size = images.size(0)
        grid_border_size = 2


        grid = utils.make_grid(images)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')
        plt.show()


def train(model,trainloader,learning_rate,num_epochs):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        print('Epoch Number: ', epoch)

        for i, (images,labels) in enumerate(trainloader):
            output = model(images)
            labels = labels.view(labels.size(0),1)


            optimizer.zero_grad()

            loss = criterion(output,labels.float())

            print('Loss at ',i,': ',loss)

            loss.backward()
            optimizer.step()


    torch.save(model.state_dict(), 'malaria.pth')

def test(model, testloader):

    total = 0
    correct = 0
    for i, (images, labels) in enumerate(testloader):


        with torch.no_grad():
            output = model(images)

        labels = labels.view(labels.size(0),1)

        t = (torch.eq(torch.round(output),labels.float()))

        total += t.size(0)
        correct += (torch.sum(t)).item()
        print(correct/total)

    print('Accuracy = ',correct/total)
def main():

    #Process the data and create the train and testloaders
    trainloader,testloader = load_data()


    # for i_batch, (images,labels) in enumerate(trainloader):
    #
    #     print(i_batch, images.size(),
    #           labels)
    #
    #     show_batch(images,labels)
    #     if i_batch == 1:
    #         break


    model = CNN()

    #train(model,trainloader,learning_rate,num_epochs)

    model.load_state_dict(torch.load('malaria.pth'))
    model.eval()

    test(model,testloader)

if __name__ == '__main__':
    main()
