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
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from omegaconf import OmegaConf
#from maritime_dataset import MaritimeDataset


import wandb


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

def prog_run():
    USE_WANDB = False

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #os.environ["CUDA_VISIBLE_DEVICES"]=""

    object_classes = ['buoys', 'ships']
    num_classes = len(object_classes)
    print(num_classes)

    batch_size = 32
    #Define image size (Resnet image size is 224 x 224 x 3)
    image_size = 224

    # Directory of the data
    data_dir = "data/raw/2_class_resnet/"
    plot_dir = "few_shot_learning/visualization/"
    model_store_path = 'models/model_partial_resnet18_2_class_cuda_25_1407_bicubic_v1.pth'
    #dataset mean and standard deviation

    data_mean = [0.4609, 0.4467, 0.4413]
    data_std = [0.1314, 0.1239, 0.1198]


    complete_dataset = MaritimeDataset(root_dir = data_dir,
                                        transform = transforms.Compose([
                                            transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
                                            transforms.CenterCrop(image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std),
                                            ])
                                        )

    #Splitting the data into train and test sets with random_split
    train_length = int(0.8* len(complete_dataset))
    test_length = len(complete_dataset) - train_length

    train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, (train_length, test_length))

    workers = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=workers)
    validation_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    convolutional_network = models.resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()

    convolutional_network.to(device)

    #for param in convolutional_network.parameters():
    #    param.requires_grad = True
    # for name, child in convolutional_network.named_children():
    #     print(name)
    for name, child in convolutional_network.named_children():
        if name in ['layer4']:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False
    def train(model, train_dataloader, criterion, optimizer, e = 5):
        running_loss =0
        acc = []
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
                scheduler.step()

                predictions = img.argmax(dim=1, keepdim=True).squeeze()
                correct_preds = (predictions.int() == labels.int()).float()
                acc.append(correct_preds)
                accuracy = torch.cat(acc, dim=0).mean().cpu()
                
                correct_batch = correct_preds/batch_size
                #acc =float("{:.4f}".format(100. * correct_batch))

                tepoch.set_postfix(loss=loss.item(), accuracy= float("{:.4f}".format(100.0 * accuracy)))
            #print("Epoch : {}/{}..".format(e+1,epochs),
            #"Training Loss: {:.6f}".format(running_loss/len(train_dataloader))) 
        training_loss = running_loss/len(train_dataloader)
        accuracy = torch.cat(acc, dim=0).mean().cpu()
        print('accuracy:', float("{:.4f}".format(100 * accuracy)))
            #train_loss.append(running_loss)
        #plt.plot(train_loss,label="Training Loss")
        #plt.show() 
        #tot_acc = 100 * correct_acc / len(train_dataloader)
        #filename_pth = 'models/model_resnet18_fsl_2_class_2.pth'
        #torch.save(model.state_dict(), filename_pth)
        #print(tot_acc)
        return training_loss, accuracy*100
    # def run():
    #     torch.multiprocessing.freeze_support()

    epochs = 25
    epoch_number = 0
    log_frequency = 2
    #optimizer = optim.Adam(convolutional_network.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(
        filter(
            lambda p: p.requires_grad, convolutional_network.parameters()
        ), lr=0.01, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, 
        steps_per_epoch=len(train_dataloader), epochs=epochs
    )
    criterion = nn.CrossEntropyLoss()    
    # train(convolutional_network,train_dataloader,criterion, optimizer, epochs) 
    #writer = SummaryWriter('runs/fsl_resnet_{}'.format(timestamp))
    best_vloss = 1_000_000.

    total_train_acc = []
    total_val_acc = []
    total_train_loss = []
    total_val_loss = []

    for e in range(epochs):
        convolutional_network.train(True)
        avg_loss, avg_acc  = train(convolutional_network,train_dataloader,criterion, optimizer, e) 
        val_acc_total = []
        convolutional_network.eval()
        with torch.no_grad():
            running_vloss = 0.0
            with tqdm(validation_dataloader, unit="batch") as tepoch:
                for images, labels in tepoch:
                    images, labels = images.to(device), labels.to(device)
                    voutputs = convolutional_network(images)
                    vloss = criterion(voutputs, labels)
                    running_vloss += vloss.item()
                    predictions = voutputs.argmax(dim=1, keepdim=True).squeeze()
                    correct_preds = (predictions.int() == labels.int()).float()
                    val_acc_total.append(correct_preds)
                    correct_batch = correct_preds/batch_size
        accuracy = torch.cat(val_acc_total, dim=0).mean().cpu()
        avg_vloss = running_vloss / len(validation_dataloader)
        avg_vacc = float("{:.4f}".format(100. * accuracy))
        print('Training loss {} validation loss {}'.format(avg_loss, avg_vloss))
        print('Train accuracy {} validation accuracy {}'.format(avg_acc, avg_vacc))
        if e % log_frequency == 0:
            if avg_vloss < best_vloss:
                print(f'{avg_vloss} < {best_vloss}')
                print('saving model')
                best_vloss = avg_vloss
                model_path = model_store_path
                torch.save(convolutional_network.state_dict(), model_path)
        epoch_number += 1

        total_train_acc.append(avg_acc)
        total_val_acc.append(avg_vacc)
        total_train_loss.append(avg_loss)
        total_val_loss.append(avg_vloss)
    
    def perf_plot(Metric, train, val):

        plt.figure(figsize=(10,5))
        plt.title(f"Training and Validation {Metric}")
        plt.plot(train,label="train")
        plt.plot(val,label="val")
        plt.xlabel("iterations")
        plt.ylabel(Metric)
        plt.yscale('log')
        plt.legend()
    perf_plot('Loss', total_train_loss, total_val_loss)
    plt.savefig(plot_dir+'resnet_train_val_loss2.png')
    perf_plot('Accuracy', total_train_acc, total_val_acc)
    plt.savefig(plot_dir+'resnet_train_val_accuracy2.png')

if __name__ == '__main__':
    prog_run()
