import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
from PIL import Image
from torch import optim
from tqdm import tqdm
import torch.optim.lr_scheduler
import os 
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.environ["CUDA_VISIBLE_DEVICES"]=""

object_classes = ['buoys', 'ships']
num_classes = len(object_classes)
print(num_classes)

batch_size = 32
#Define image size
image_size = 8

# Directory of the data
data_dir = "data/raw/2_class_resnet/"

#dataset mean and standard deviation

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]

class MaritimeDataset(Dataset):

    def __init__(self, root_dir="", transform=transforms):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.paths, self.labels = list(map(list, zip(*self.dataset.samples)))
        self.classes = self.dataset.class_to_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        target = self.labels[index]
        img = Image.open(self.paths[index]).convert('RGB')

        if self.transform:
            images = self.transform(img)
        return images, torch.tensor(target)


complete_dataset = MaritimeDataset(root_dir = data_dir,
                                    transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std),
                                        ])
                                    )

#Splitting the data into train and test sets with random_split
train_length = int(0.7* len(complete_dataset))
test_length = len(complete_dataset) - train_length

train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, (train_length, test_length))

workers = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
validation_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

convolutional_network = models.resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()


for param in convolutional_network.parameters():
    param.requires_grad = True
def train(model, train_dataloader, criterion, optimizer, e = 5):
    running_loss =0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for images, labels in tepoch:
            inputs, labels = images.to(device), labels.to(device)
            tepoch.set_description(f"Epoch {int(e)+1}")

            optimizer.zero_grad()
            img = model(inputs)
            
            loss = criterion(img, labels)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            predictions = img.argmax(dim=1, keepdim=True).squeeze()
            #correct_acc += (predictions == labels).float().sum()
            correct_preds = (predictions == labels).float().sum()
            correct_batch = correct_preds/batch_size
            acc =float("{:.4f}".format(100. * correct_batch))
            tepoch.set_postfix(loss=loss.item(), accuracy=float("{:.4f}".format(100. * correct_batch)))
        #print("Epoch : {}/{}..".format(e+1,epochs),
        #"Training Loss: {:.6f}".format(running_loss/len(train_dataloader))) 
    training_loss = running_loss/len(train_dataloader)
        #train_loss.append(running_loss)
    #plt.plot(train_loss,label="Training Loss")
    #plt.show() 
    #tot_acc = 100 * correct_acc / len(train_dataloader)
    #filename_pth = 'models/model_resnet18_fsl_2_class_2.pth'
    #torch.save(model.state_dict(), filename_pth)
    #print(tot_acc)
    return training_loss, acc
def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()
    epochs = 5
    epoch_number = 0
    convolutional_network.train()
    optimizer = optim.Adam(convolutional_network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()    
   # train(convolutional_network,train_dataloader,criterion, optimizer, epochs) 
    #writer = SummaryWriter('runs/fsl_resnet_{}'.format(timestamp))
    best_vloss = 1_000_000.
    for e in range(epochs):

        avg_loss, avg_acc  = train(convolutional_network,train_dataloader,criterion, optimizer, e) 

        convolutional_network.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_dataloader):
            vinputs, vlabels = vdata
            voutputs = convolutional_network(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
            predictions = voutputs.argmax(dim=1, keepdim=True).squeeze()
            correct_preds = (predictions == vlabels).float().sum()
            correct_batch = correct_preds/batch_size

        avg_vloss = running_vloss / (i + 1)
        acf_vacc = float("{:.4f}".format(100. * correct_batch))
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACCURACY train {} valid {}'.format(avg_acc, acf_vacc))

    # Log the running loss averaged per batch
    # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #             epoch_number + 1)
        # writer.flush()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/model_resnet18_fsl_2_class_4.pth'
            torch.save(convolutional_network.state_dict(), model_path)
        epoch_number += 1