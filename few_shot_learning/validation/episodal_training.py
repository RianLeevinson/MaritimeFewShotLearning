#!/usr/bin/env python3

import os
import random

from easyfsl.data_tools import TaskSampler
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image_size = 224
new_val_data_dir = r'C:\DTU\master_thesis\MaritimeFewShotLearning\data\processed\new_data_may\val_set'

data_conf = OmegaConf.load(r'few_shot_learning\utils\config.yaml')

plot_dir = r"few_shot_learning/visualization/"

#model_store_path = r'models/protonet_episodal_5_shot_resnet18_1307_norm_v1.pth'
model_store_path_custom_5 = r'models/protonet_episodal_5_shot_resnet18_1407_bicubic_v1.pth'
model_store_path_pre_resnet_5 = r'models/protonet_5_shot_resnet18_pretrained_1407_v1.pth'

model_store_path_pre_sgd_1 = r'models/protonet_1_shot_resnet18_pretrained_1407_v1.pth'
model_store_path_pre_resnet_sgd_5 = r'models/protonet_5_shot_resnet18_pretrained_1407_sgd_v2.pth'
model_store_path_custom_sgd_1 = r'models/protonet_1_shot_resnet18_custom_1407_v1.pth'

model_store_path_custom_sgd_2 = r'models/protonet_2_shot_resnet18_custom_1407_v1.pth'
model_store_path_custom_adam_2 = r'models/protonet_2_shot_resnet18_custom_1407_adam_v1.pth'

model_store_path_custom_sgd_5 = r'models/protonet_5_shot_resnet18_custom_1407_sgd_v1.pth'
model_store_path_pre_adam_5 = r'models/protonet_5_shot_resnet18_pretrained_1407_adam_v1.pth'

model_store_path_custom_adam_5 = r'models/protonet_5_shot_resnet18_custom_1407_adam_v1.pth'
model_store_path_custom_adam_5_v2 = r'models/protonet_5_shot_resnet18_custom_1407_adam_v2.pth'
model_store_path_custom_adam_5_v3 = r'models/protonet_5_shot_resnet18_custom_1407_adam_v3.pth'

model_store_path_custom_adam_1 = r'models/protonet_1_shot_resnet18_custom_1407_adam_v1.pth'
model_store_path_pretrained_adam_1 = r'models/protonet_1_shot_resnet18_pretrained_1407_adam_v1.pth'

model_store_path_pretrained_adam_2 = r'models/protonet_2_shot_resnet18_pretrained_1407_adam_v1.pth'

model_store_path_pretrained_adam_5_v5 = r'models/protonet_5_shot_resnet18_pretrained_1407_adam_v5.pth'

model_store_path = model_store_path_pretrained_adam_5_v5
full_train = r'data\processed\new_data_may\train_set'
n_shot = data_conf.TEST_CONFIG

lat_train = r"data\processed\lat_train"
resnet_path = 'models/model_partial_resnet18_2_class_cuda_100_1307_v1.pth'
resnet_25_path = 'models/model_partial_resnet18_2_class_cuda_25_1407_bicubic_v1.pth'
lat_val = r"data\processed\lat_val"
full_train_sub = r'data\processed\new_data_may\train_set_exp'
dir = full_train_sub
val_dir = lat_val

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]

train_data = datasets.ImageFolder(root = dir, transform = transforms.Compose(
        [
            transforms.Resize(size=image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(data_mean, data_std),
        ]
    ),)



train_classes = os.listdir(dir)


N_WAY_TRAIN = len(train_classes) # Number of classes
#N_WAY_TEST = len(test_classes) # Number of classes
N_SHOT = 5 # Number of images per class
N_QUERY = 5 # Number of images per class in the query set
N_EVALUATION_TASKS = 500

N_TRAINING_EPISODES = 500
#N_VALIDATION_TASKS_2 = 50

#train_dataset.get_labels = lambda: [instance[1] for instance in train_dataset._flat_character_images]
#train_dataset.labels = train_dataset.targets

#test_dataset.labels = test_dataset.targets
train_data.labels = train_data.targets
train_sampler = TaskSampler(
    train_data, n_way=N_WAY_TRAIN, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)



def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


train_loader = DataLoader(
    train_data,
    batch_sampler=train_sampler,
    #num_workers=8,
    pin_memory=True,
    worker_init_fn= seed_worker,
    collate_fn=train_sampler.episodic_collate_fn,
)



from torch import optim

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:

    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )
    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()
    predictions = classification_scores.argmax(dim=1, keepdim=True).squeeze()
    correct_preds = (predictions.int() == query_labels.cuda().int()).float()

    return loss.item(), correct_preds

from easyfsl.utils import sliding_average
log_update_frequency = 20



class PrototypicalNetworkModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworkModel, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)


        n_way = len(torch.unique(support_labels))

        z_proto_median = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].median(dim = 0)[0]
                
                for label in range(n_way)
            ]
        )

        z_proto_mean = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(dim = 0)
                
                for label in range(n_way)
            ]
        )

        z_total = torch.div(torch.add(z_proto_median, z_proto_mean), 2)
        #Eucledian distance metric
        
        def pairwise(z_query, z_proto):
            pdist = nn.PairwiseDistance(p=2)
            d1 = []
            for j in range(0, len(z_query)):
                d2 = []
                for i in range(0,len(z_proto)):
                    d2.append(pdist(z_query[j], z_proto[i]))
                d1.append(d2)
            return(torch.FloatTensor(d1).to(device))

        def cosinesimilarity(z_query, z_proto):
            cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
            d1 = []
            for j in range(0, len(z_query)):
                d2 = []
                for i in range(0,len(z_proto)):
                    d2.append(cos1(z_query[j], z_proto[i]))
                d1.append(d2)
            return(torch.FloatTensor(d1).to(device))
        #dists = torch.cdist(z_query, z_total)
        #dists = pairwise(z_query, z_total)
        dists = torch.cdist(z_query, z_proto_mean)

        
        #dists = cosinesimilarity(z_query, z_total)
        scores = -dists
        
        return scores

def select_model(mode):
    if mode == 1:
        filename_pth = 'models/model_partial_resnet18_fsl_2_class_cuda_100_v1.pth'
        convolutional_network = resnet18(pretrained=False)
        convolutional_network.fc = nn.Flatten()
        convolutional_network.load_state_dict(torch.load(filename_pth))
    else:
        convolutional_network = resnet18(pretrained=True)
        convolutional_network.fc = nn.Flatten()
    return convolutional_network

# 1 - Custom trained ResNet18
# 2 - Pretrained ResNet18 
convolutional_network = select_model(2)

model = PrototypicalNetworkModel(convolutional_network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for name, child in model.named_children():
    print(name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for param in model.parameters():
   param.requires_grad = True
best_vloss = 1_000_000.
acc = []
all_loss = []
all_acc = []
model.train()
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        loss_value, correct_preds = fit(support_images, support_labels, query_images, query_labels)
        all_loss.append(loss_value)
        acc.append(correct_preds)
        accuracy = torch.cat(acc, dim=0).mean().cpu()
        all_acc.append(accuracy)
        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency), accuracy = float("{:.4f}".format(100.0 * accuracy)))
            if loss_value < best_vloss:
                print(f'{loss_value} < {best_vloss}')
                print('saving model')
                best_vloss = loss_value
                model_path = model_store_path
                torch.save(model.state_dict(), model_path)
accuracy = torch.cat(acc, dim=0).mean().cpu()
print(accuracy)



def perf_plot(Metric, train):

    plt.figure(figsize=(10,5))
    ax = plt.subplot(111)
    plt.title(f"Training {Metric}")
    plt.plot(train,label="train", color='blue')
    plt.xlabel("iterations")
    plt.ylabel(Metric)
    plt.yscale('log')
    plt.legend()
    #ax.grid('on')
    plt.grid()
# perf_plot('Loss', all_loss)
# plt.savefig(plot_dir+'episodic_train_loss_2_pretrained_adam_v1.png')
# perf_plot('Accuracy', all_acc)
# plt.savefig(plot_dir+'episodic_train_acc_2_pretrained_adam_v1.png')

# import pickle

# with open('accuracy_custom_resnet_2_pretrained_adam_v1', 'wb') as fp:
#     pickle.dump(all_acc, fp)

# with open('loss_custom_resnet_2_pretrained_adam_v1', 'wb') as fp:
#     pickle.dump(all_loss, fp)

#with open ('outfile', 'rb') as fp:
#    itemlist = pickle.load(fp)
#    print(itemlist)