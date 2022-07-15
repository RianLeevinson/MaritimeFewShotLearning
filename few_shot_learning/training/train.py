import os

import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from torchvision import datasets, transforms
import random
import numpy as np
from easyfsl.data_tools import TaskSampler
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

val_8_data_dir = r"data\processed\fsl_val_8_all"
val_6_buoy_dir = r"data\processed\fsl_val_6_buoys"
val_6_small_vessels_dir = r"data\processed\fsl_val_6_small_vessels"
val_6_sailboats_dir = r"data\processed\fsl_val_6_sailboats"
val_5_dir = r"data\processed\lat_val" 
full_val_data = r'data\processed\new_data_may\val_set'

full_val_sub_data = r'data\processed\new_data_may\val_sub_set'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lat_val = r"data\processed\lat_val"

train_set = r'data\processed\new_data_may\train_set_exp'

full_val_sub_data_exp = r'data\processed\new_data_may\val_sub_set_exp'
full_val_sub_data_exp_v2 = r'data\processed\new_data_may\val_set_all' 
image_size = 224
val_dir = train_set
test_classes = os.listdir(val_dir)



N_WAY_TEST = len(test_classes) # Number of classes
N_SHOT = 5 # Number of images per class
N_QUERY = 5 # Number of images per class in the query set
N_EVALUATION_TASKS = 500

val_data = datasets.ImageFolder(root = val_dir, transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),)


val_data.labels = val_data.targets
test_sampler = TaskSampler(
    val_data, n_way=N_WAY_TEST, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

test_loader = DataLoader(
    val_data,
    batch_sampler=test_sampler,
    #num_workers=8,
    pin_memory=True,
    worker_init_fn= seed_worker,
    collate_fn=test_sampler.episodic_collate_fn,
)

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

        # z_proto_median = torch.cat(
        #     [
        #         z_support[torch.nonzero(support_labels == label)].median(dim = 0)[0]
                
        #         for label in range(n_way)
        #     ]
        # )

        z_proto_mean = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(dim = 0)
                
                for label in range(n_way)
            ]
        )

        # z_total = torch.div(torch.add(z_proto_median, z_proto_mean), 2)
        # #Eucledian distance metric
        
        # def pairwise(z_query, z_proto):
        #     pdist = nn.PairwiseDistance(p=2)
        #     d1 = []
        #     for j in range(0, len(z_query)):
        #         d2 = []
        #         for i in range(0,len(z_proto)):
        #             d2.append(pdist(z_query[j], z_proto[i]))
        #         d1.append(d2)
        #     return(torch.FloatTensor(d1).to(device))

        # def cosinesimilarity(z_query, z_proto):
        #     cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
        #     d1 = []
        #     for j in range(0, len(z_query)):
        #         d2 = []
        #         for i in range(0,len(z_proto)):
        #             d2.append(cos1(z_query[j], z_proto[i]))
        #         d1.append(d2)
        #     return(torch.FloatTensor(d1).to(device))
        # #dists = torch.cdist(z_query, z_total)
        # #dists = pairwise(z_query, z_total)
        dists = torch.cdist(z_query, z_proto_mean)

        
        #dists = cosinesimilarity(z_query, z_total)
        scores = -dists
        
        return scores

model_PATH = 'models/protonet_episodal_5_shot_meta_train_v2.pth'
filename_pth = 'models/model_partial_resnet18_fsl_2_class_cuda_100_v1.pth'
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
#convolutional_network.load_state_dict(torch.load(filename_pth))
model = PrototypicalNetworkModel(convolutional_network)
model.to(device)
#model.load_state_dict(torch.load(model_PATH))
model.eval()

# def evaluate_on_one_task(
#     support_images: torch.Tensor,
#     support_labels: torch.Tensor,
#     query_images: torch.Tensor,
#     query_labels: torch.Tensor,
# ):
#     """
#     evaluation function
#     """
#     class_inf = torch.max(model(support_images, support_labels, query_images).data,1,)[1].tolist()

#     return (torch.max(model(support_images, support_labels, query_images).data, 1,)[1]
#         == query_labels).sum().item(), len(query_labels) , class_inf, query_labels.tolist()

# def evaluate(data_loader: DataLoader):
#     """
#     Evaluates the model 
#     """   

#     total_predictions = 0
#     correct_predictions = 0
#     exact = []
#     predicted = []


#     model.eval()
#     pred_list = []
#     with torch.no_grad():
#         for episode_index, (support_images,support_labels,query_images,query_labels,class_ids,) in tqdm(enumerate(data_loader), total=len(data_loader)):
#             correct, total, predicted_classes, exact_classes = evaluate_on_one_task(support_images.to(device), support_labels.to(device), query_images.to(device), query_labels.to(device))
#             exact.extend(exact_classes)
#             predicted.extend(predicted_classes)
#             pred_list.append(correct)
#             total_predictions += total
#             correct_predictions += correct

#     model_accuracy = (100 * correct_predictions/total_predictions)
#     print(
#         f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
#     )

#     return exact, predicted, model_accuracy


# exact, predicted, model_accuracy = evaluate(test_loader)

# print(model_accuracy)


def find_classes(val_dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(val_dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx

cf_mat = confusion_matrix(exact, predicted)

# def create_cf_plot():
#     '''Creates a confusion matrix of the model predictions '''

#     plt.figure(figsize=(8,8))
#     classes_idx = find_classes(val_dir)


#     cf_mat = confusion_matrix(exact, predicted, normalize='true')
#     class_names = list(classes_idx.keys())
#     tick_class_labels = []
#     for vessel_classes in class_names:        tick_class_labels.append(vessel_classes.replace('_', ' ').capitalize())
#     df_cm = pd.DataFrame(
#         cf_mat, index=tick_class_labels, columns=tick_class_labels
#     ).astype(float)

#     heatmap = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", linewidths=.5, fmt='g')

#     heatmap.yaxis.set_ticklabels(
#         heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=8
#     )
#     heatmap.xaxis.set_ticklabels(
#         heatmap.xaxis.get_ticklabels(), rotation=0,fontsize=8
#     )
#     plt.title(f'Few Shot Learning \n Image size: {image_size} \n Number of shots: {N_SHOT} \n Number of query images: {N_QUERY} \n Model Accuracy: {model_accuracy} ')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

#     return plt

# plt = create_cf_plot()
# plt.show()
import torchvision

def plot_images(test_loader: DataLoader, N_SHOT: int):
    '''Plots the support images in all the classes'''

    (
    example_support_images, example_support_labels,
    example_query_images, example_query_labels,
    example_class_ids,
    ) = next(iter(test_loader))

    _, ax = plt.subplots()
    plt.title("Support Images")
    dummy_val = 0
    classes_list = list(example_support_labels)
    list_classes = list(find_classes(val_dir).keys())
    #plt.yticks(np.arange(0, 1.2, step=0.2)) 
    x_ticks_val = range(N_SHOT)
    ax.set_yticks(np.arange(len(list_classes)), list_classes) 
    ax.set_xticks(x_ticks_val, x_ticks_val) 
    plt.imshow(
        torchvision.utils.make_grid(
            example_support_images, nrow=N_SHOT
        ).permute(1, 2, 0)
    )
    return plt

plot1 = plot_images(test_loader, 5)
plot1.show()