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
import io

object_classes = ['buoys', 'harbour', 'human', 'large_commercial_vessel', 'leisure_craft', 'sailboats', 'small_medium_fishing_boat']
num_classes = len(object_classes)

data_path = "C:/DTU/master_thesis/fsl/Object-classification-with-few-shot-learning/data/raw/updated_ds/sailboats"

model_name = "resnet"


batch_size = 1


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
        img = self.paths[index]
        with Image.open(img) as img:
                
            if self.transform is not None:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=75)

# You probably want
                img2 = buffer.getbuffer()
                img = self.transform(img2)
                target = self.transform(target)
        #print(img.size)
        return img, target

image_size = 128

data_dir = "C:/DTU/master_thesis/fsl/Object-classification-with-few-shot-learning/data/raw/updated_ds"

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]

complete_dataset = MaritimeDataset(root_dir = data_dir,
                                    transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std),


                                        ])
                                    )

#Splittin the data into train and test sets

train_length = int(0.7* len(complete_dataset))

test_length = len(complete_dataset) - train_length

#train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, (train_length, test_length))


workers = 1
train_dataloader = DataLoader(complete_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
#test_dataloader =  DataLoader(test_dataset, batch_size=batch_size,
                     #       shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


#opt = torch.optim.Adam(model.parameters(), lr=1e-5)

loss = torch.nn.CrossEntropyLoss()

criterion = nn.NLLLoss()


convolutional_network = models.resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()

for param in convolutional_network.parameters():
    param.requires_grad = True
device = torch.device("cuda")
def train(model, train_dataloader, criterion, optimizer, epochs = 5):
    train_loss =[]
    for e in tqdm(range(epochs)):
        print('3')
        running_loss =0
        for images, labels in tqdm(train_dataloader):
            print('4')
            inputs, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            img = model(inputs)
            
            loss = criterion(img, labels)
            running_loss+=loss
            loss.backward()
            optimizer.step()
        print("Epoch : {}/{}..".format(e+1,epochs),
         "Training Loss: {:.6f}".format(running_loss/len(train_dataloader))) 
        train_loss.append(running_loss)
    #plt.plot(train_loss,label="Training Loss")
    #plt.show() 
    filename_pth = 'model_resnet18_fsl.pth'
    torch.save(model.state_dict(), filename_pth)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

epochs = 5
convolutional_network.train()
optimizer = optim.Adam(convolutional_network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()    


if __name__ == '__main__':
    run()
    epochs = 5
    convolutional_network.train()
    optimizer = optim.Adam(convolutional_network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()    
    train(convolutional_network,train_dataloader,criterion, optimizer, epochs) 