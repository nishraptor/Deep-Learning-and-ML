import numpy as np
import torch
import torch.nn as nn
import torchvision
from MalariaDataset import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split

from model import *
import torch.optim as optim
import random
from   torch.autograd import Variable


learning_rate = .0000001
batch_size = 4
num_epochs = 1

#Load in the data from the testing and training datasets
def load_data():

    #Create the dataset
    dataset = MalariaDataset(csv_file='malaria.csv', transform=transforms.Compose([
                                               transforms.Resize((100,100)),
                                               transforms.ToTensor()]))

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

def show_batch(sample_batched):

        images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['label']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')

def train(model,trainloader,learning_rate,num_epochs):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        print('Epoch Number: ', epoch)

        for i, sample in enumerate(trainloader):
            output = model(sample['image'])

            optimizer.zero_grad()

            loss = criterion(output,sample['label'].float())

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
    trainloader,testloader = load_data()


    #
    # for i_batch, sample_batched in enumerate(trainloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['label'])
    #
    #
    #     if i_batch == 2:
    #         plt.figure()
    #         show_batch(sample_batched)
    #         plt.axis('off')
    #         plt.ioff()
    #         plt.show()
    #         break


    model = CNN()
    train(model,trainloader,learning_rate,num_epochs)

    #model.load_state_dict(torch.load('autoencoder.pth'))
    #model.eval()

    #test(model,testloader)

if __name__ == '__main__':
    main()
