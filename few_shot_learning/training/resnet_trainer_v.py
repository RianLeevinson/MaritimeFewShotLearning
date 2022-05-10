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


os.environ["CUDA_VISIBLE_DEVICES"]=""

object_classes = ['buoys', 'ships']
num_classes = len(object_classes)
print(num_classes)
model_name = "resnet"


batch_size = 8


# Number of epochs to train for
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

class MaritimeDataset(Dataset):

    def __init__(self, root_dir="", transform=transforms):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        #self.paths, self.labels = list(zip(*self.dataset.samples))
        self.paths, self.labels = list(map(list, zip(*self.dataset.samples)))
        self.classes = self.dataset.class_to_idx

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        
        #path, target = self.paths[index], self.labels[index]
        target = self.labels[index]
        img = Image.open(self.paths[index]).convert('RGB')

        if self.transform:
            images = self.transform(img)
#             targets = self.transform(target)
        #print(img.size)
        return images, torch.tensor(target)

image_size = 224

data_dir = "data/raw/2_class_resnet/"

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]

complete_dataset = MaritimeDataset(root_dir = data_dir,
                                    transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std),
                                        ])
                                    )
# print(next(iter(complete_dataset)))

#Splittin the data into train and test sets

train_length = int(0.7* len(complete_dataset))

test_length = len(complete_dataset) - train_length

#train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, (train_length, test_length))


workers = 8
train_dataloader = DataLoader(complete_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
#test_dataloader =  DataLoader(test_dataset, batch_size=batch_size,
                     #       shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#opt = torch.optim.Adam(model.parameters(), lr=1e-5)

loss = torch.nn.CrossEntropyLoss()

criterion = nn.NLLLoss()


convolutional_network = models.resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()



for param in convolutional_network.parameters():
    param.requires_grad = True
def train(model, train_dataloader, criterion, optimizer, epochs = 5):
    train_loss =[]
    for e in range(epochs):
        #correct_acc = 0
        running_loss =0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for images, labels in tepoch:
                inputs, labels = images.to(device), labels.to(device)
                tepoch.set_description(f"Epoch {int(e)+1}")

                optimizer.zero_grad()
                img = model(inputs)
                
                loss = criterion(img, labels)
                running_loss+=loss
                loss.backward()
                optimizer.step()
                predictions = img.argmax(dim=1, keepdim=True).squeeze()
                #correct_acc += (predictions == labels).float().sum()
                cor_batch = (predictions == labels).float().sum()
                cor_batch_2 = cor_batch/batch_size
                tepoch.set_postfix(loss=loss.item(), accuracy=float("{:.4f}".format(100. * cor_batch_2)))
            #print("Epoch : {}/{}..".format(e+1,epochs),
            #"Training Loss: {:.6f}".format(running_loss/len(train_dataloader))) 
            
            train_loss.append(running_loss)
        #plt.plot(train_loss,label="Training Loss")
        #plt.show() 
        #tot_acc = 100 * correct_acc / len(train_dataloader)
        filename_pth = 'models/model_resnet18_fsl_2_class_2.pth'
        torch.save(model.state_dict(), filename_pth)
        #print(tot_acc)
def run():
    torch.multiprocessing.freeze_support()

# epochs = 5
# convolutional_network.train()
# optimizer = optim.Adam(convolutional_network.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()    


if __name__ == '__main__':
    run()
    epochs = 5
    convolutional_network.train()
    optimizer = optim.Adam(convolutional_network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()    
    train(convolutional_network,train_dataloader,criterion, optimizer, epochs) 