#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, wide_resnet50_2
from tqdm import tqdm
import numpy as np
from easyfsl.data_tools import TaskSampler
from sklearn.metrics import confusion_matrix
import random
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from omegaconf import OmegaConf
import torchvision



image_size = 224  #setting the image size

#Loading the data

old_data_dir = r"few_shot_learning\data\updated_data_fsl"

new_data_dir = r"data\processed\fsl_data_7"



data_conf = OmegaConf.load(r'few_shot_learning\utils\config.yaml')

n_shot = data_conf.TEST_CONFIG

 
dir = new_data_dir

fsl_dataset = datasets.ImageFolder(root = dir, transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),)


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

        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].median(dim = 0)[0]
                
                for label in range(n_way)
            ]
        )

        z_proto2 = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(dim = 0)
                
                for label in range(n_way)
            ]
        )

        z_total = torch.div(torch.add(z_proto, z_proto2), 2)
        #Eucledian distance metric
        pdist = nn.PairwiseDistance(p=2)
        def pairwise(z_query, z_proto):

            d1 = []
            for j in range(0, len(z_query)):
                d2 = []
                for i in range(0,len(z_proto)):
                    d2.append(pdist(z_query[j], z_proto[i]))
                d1.append(d2)
            return(torch.FloatTensor(d1).to(device))
        #dists = torch.cdist(z_query, z_total)
        dists = pairwise(z_query, z_total)
        scores = -dists
        return scores

#loading custom trained model

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
convolutional_network = select_model(1)
model = PrototypicalNetworkModel(convolutional_network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#Selecting the fsl parameters

classes = os.listdir(dir)


N_WAY = len(classes) # Number of classes
N_SHOT = 5 # Number of images per class
N_QUERY = 5 # Number of images per class in the query set
N_EVALUATION_TASKS = 50

#Setting the seed value to 0 to enable reproducability of results

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

fsl_dataset.labels = fsl_dataset.targets
test_sampler = TaskSampler(
    fsl_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)


test_loader = DataLoader(
    fsl_dataset,
    batch_sampler=test_sampler,
    #num_workers=1,
    pin_memory=True,
    worker_init_fn= seed_worker,
    collate_fn=test_sampler.episodic_collate_fn,
)


(
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
) = next(iter(test_loader))
model_PATH = r'C:\DTU\master_thesis\fsl\Object-classification-with-few-shot-learning\models'
torch.save(model.state_dict(), os.path.join(model_PATH,'fsl_model.pth'))
# model.eval()
# example_scores = model(example_support_images, example_support_labels, example_query_images)
# _, example_predicted_labels = torch.max(example_scores.data, 1)


def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
):
    """
    evaluation function
    """
    class_inf = torch.max(model(support_images, support_labels, query_images).data,1,)[1].tolist()

    return (torch.max(model(support_images, support_labels, query_images).data, 1,)[1]
        == query_labels).sum().item(), len(query_labels) , class_inf, query_labels.tolist()

#Evaluate the model

def evaluate(data_loader: DataLoader):
    """
    Evaluates the model 
    """   

    total_predictions = 0
    correct_predictions = 0
    exact = []
    predicted = []


    model.eval()
    pred_list = []
    with torch.no_grad():
        for episode_index, (support_images,support_labels,query_images,query_labels,class_ids,) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total, predicted_classes, exact_classes = evaluate_on_one_task(support_images.to(device), support_labels.to(device), query_images.to(device), query_labels.to(device))
            exact.extend(exact_classes)
            predicted.extend(predicted_classes)
            pred_list.append(correct)
            total_predictions += total
            correct_predictions += correct

    model_accuracy = (100 * correct_predictions/total_predictions)
    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )

    return exact, predicted, model_accuracy

exact, predicted, model_accuracy = evaluate(test_loader)


def find_classes(dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx

cf_mat = confusion_matrix(exact, predicted)

def create_cf_plot():
    '''Creates a confusion matrix of the model predictions '''

    plt.figure(figsize=(6,6))
    classes_idx = find_classes(dir)


    cf_mat = confusion_matrix(exact, predicted, normalize='true')
    class_names = list(classes_idx.keys())
    tick_class_labels = []
    for vessel_classes in class_names:
        tick_class_labels.append(vessel_classes.replace('_', ' ').capitalize())
    df_cm = pd.DataFrame(
        cf_mat, index=tick_class_labels, columns=tick_class_labels
    ).astype(float)

    heatmap = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", linewidths=.5, fmt='g')

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=8
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=0,fontsize=8
    )
    plt.title(f'Few Shot Learning \n Image size: {image_size} \n Number of shots: {N_SHOT} \n Number of query images: {N_QUERY} \n Model Accuracy: {model_accuracy} ')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return plt

plt = create_cf_plot()
plt.show()

def create_hist(tensor_mat):
    '''Creates a histogram with all the class accuracies'''

    plt.figure(figsize=(12,8))
    classes_idx = find_classes(dir)


    cf_mat = confusion_matrix(exact, predicted)
    class_names = list(classes_idx.keys())
    df_cm = pd.DataFrame(
        tensor_mat, index=class_names, columns=class_names
    ).astype(int)
    hist = sns.histplot(df_cm, binwidth=3)
    plt.title(f'Few Shot Learning \n Image size: {image_size} \n Number of shots: {N_SHOT} \n Number of query images: {N_QUERY} \n Model Accuracy: {model_accuracy} ')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

#plt = create_hist(cf_mat)
#plt.show()

def plot_images():
    '''Plots the support images in all the classes'''

    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids, = next(iter(test_loader))
    _, ax = plt.subplots()
    plt.title("Support Images")
    plt.imshow(
        torchvision.utils.make_grid(
            example_support_images, nrow=N_SHOT
        ).permute(1, 2, 0)
    )
    plt.ylabel('Vessel classes')
    dummy_val = 0
    classes_list = list(example_support_labels)
    classes_list2 = classes_list.insert(0,dummy_val)
    ax.set_yticklabels(classes_list2)

    return plt

#plt = plot_images()
#plt.show()


cf_report = pd.DataFrame(
    classification_report(
        exact, predicted, target_names = list(
            find_classes(dir).keys()
        ), output_dict = True
    )
)

cf_report_short = cf_report.drop(
    ['macro avg', 'weighted avg'], axis = 1
).drop(['support'], axis = 0)
report_df = pd.DataFrame(cf_report_short.loc['precision'])
report_df['recall'] = cf_report_short.loc['recall']
report_df['f1_score'] = cf_report_short.loc['f1-score']
report_df['vessel_class'] = report_df.index
print(report_df)
x = pd.melt(report_df.drop(['accuracy'], axis = 0), ['vessel_class'])
sns.lineplot(x='vessel_class', y='value', hue='variable', data=x)
plt.ylabel('Value')
plt.xlabel('Vessel Classes')

plt.show()
