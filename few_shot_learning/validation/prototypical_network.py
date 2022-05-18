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

from few_shot_learning.utils.util_functions import (
    compute_protoype_mean,
    reproducability_config,
    seed_worker,
    find_classes, 
    plot_images,
    compute_protoype_median,
    compute_protoype_mean
)


seed_value = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrototypicalNetworkModel(nn.Module):
    '''
    Class for the prototypical network, adapted from 
    https://github.com/sicara/easy-few-shot-learning
    '''

    def __init__(self, backbone: nn.Module):
        '''Initializing with backbone model'''

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
        z_proto_median = compute_protoype_median(z_support, support_labels)
        z_proto_mean = compute_protoype_mean(z_support, support_labels)

        #dists = pairwise(z_query, z_proto_mean, device)
        dists = torch.cdist(z_query, z_proto_mean)
        #dists = cosinesimilarity(z_query, z_proto_mean)
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


def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    model: nn.Module,
):
    """
    evaluation function
    """
    class_inf = torch.max(
        model(support_images, support_labels, query_images).data, 1
    )[1].tolist()
    return (torch.max(model(support_images, support_labels, query_images).data, 1,)[1]
        == query_labels).sum().item(), len(query_labels) , class_inf, query_labels.tolist()


def evaluate(data_loader: DataLoader, model: nn.Module):
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
        for episode_index, (
            support_images,support_labels,query_images,query_labels,class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):
            (
                correct, total,
                predicted_classes,
                exact_classes
            ) = evaluate_on_one_task(
                    support_images.to(device), 
                    support_labels.to(device), 
                    query_images.to(device),
                    query_labels.to(device),
                    model
                )

            exact.extend(exact_classes)
            predicted.extend(predicted_classes)
            pred_list.append(correct)
            total_predictions += total
            correct_predictions += correct

    model_accuracy = (100 * correct_predictions/total_predictions)
    print(
        f'Model tested on {len(data_loader)} tasks.'
        f'Accuracy: {(100 * correct_predictions/total_predictions):.2f}%'
    )
    return exact, predicted, model_accuracy


def create_cf_plot(cf_mat, image_size, N_SHOT, N_QUERY, model_accuracy, data_dir):
    '''Creates a confusion matrix of the model predictions'''

    plt.figure(figsize=(6,6))
    classes_idx = find_classes(data_dir)
    class_names = list(classes_idx.keys())
    tick_class_labels = []
    for vessel_classes in class_names:
        tick_class_labels.append(vessel_classes.replace('_', ' ').capitalize())
    df_cm = pd.DataFrame(
        cf_mat, index=tick_class_labels, columns=tick_class_labels
    ).astype(float)

    heatmap = sns.heatmap(
        df_cm, annot=True, cmap="YlGnBu", linewidths=.5, fmt='g'
    )

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=8
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=0,fontsize=8
    )
    plt.title(
        f'Few Shot Learning \n '
        f'Image size: {image_size} \n '
        f'Number of shots: {N_SHOT} \n '
        f'Number of query images: {N_QUERY} \n '
        f'Model Accuracy: {model_accuracy} '
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


# cf_report = pd.DataFrame(
#     classification_report(
#         exact, predicted, target_names = list(
#             find_classes(dir).keys()
#         ), output_dict = True
#     )
# )

# cf_report_short = cf_report.drop(
#     ['macro avg', 'weighted avg'], axis = 1
# ).drop(['support'], axis = 0)
# report_df = pd.DataFrame(cf_report_short.loc['precision'])
# report_df['recall'] = cf_report_short.loc['recall']
# report_df['f1_score'] = cf_report_short.loc['f1-score']
# report_df['vessel_class'] = report_df.index
# print(report_df)
# x = pd.melt(report_df.drop(['accuracy'], axis = 0), ['vessel_class'])
# sns.lineplot(x='vessel_class', y='value', hue='variable', data=x)
# plt.ylabel('Value')
# plt.xlabel('Vessel Classes')

# #plt.show()


def main():

    reproducability_config(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)

    image_size = 224  #setting the image size

    #Loading the data

    old_data_dir = r"few_shot_learning\data\updated_data_fsl"
    new_data_dir = r"data\processed\fsl_data_6"
    custom_data_dir = r"data\processed\fsl_data_6_custom"

    data_conf = OmegaConf.load(r'few_shot_learning\utils\config.yaml')
    n_shot = data_conf.TEST_CONFIG
    classes = os.listdir(custom_data_dir)

    fsl_dataset = datasets.ImageFolder(
        root = custom_data_dir, transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    # 1 - Custom trained ResNet18
    # 2 - Pretrained ResNet18 
    convolutional_network = select_model(2)
    model = PrototypicalNetworkModel(convolutional_network)
    model.to(device)

    N_WAY = len(classes) # Number of classes
    N_SHOT = 5 # Number of images per class
    N_QUERY = 5 # Number of images per class in the query set
    N_EVALUATION_TASKS = 50

    fsl_dataset.labels = fsl_dataset.targets
    test_sampler = TaskSampler(
        fsl_dataset, 
        n_way=N_WAY, 
        n_shot=N_SHOT, 
        n_query=N_QUERY, 
        n_tasks=N_EVALUATION_TASKS
    )

    test_loader = DataLoader(
        fsl_dataset,
        batch_sampler=test_sampler,
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
    model_PATH = 'models/'
    torch.save(model.state_dict(), os.path.join(model_PATH,'fsl_protonet.pth'))

    exact, predicted, model_accuracy = evaluate(test_loader, model)
    cf_mat = confusion_matrix(exact, predicted)

    plt = create_cf_plot(cf_mat, image_size, N_SHOT, N_QUERY, model_accuracy, custom_data_dir)
    plt.show()

    #plt = create_hist(cf_mat)
    #plt.show() 

    #plt = plot_images(test_loader, N_SHOT)
    #plt.show()

if __name__ == '__main__':
    main()
