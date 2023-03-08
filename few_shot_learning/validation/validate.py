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

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

val_5_dir = r"data\processed\lat_val" 
full_val_data = r'data\processed\new_data_may\val_set'

full_val_sub_data = r'data\processed\new_data_may\val_sub_set'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lat_val = r"data\processed\lat_val"

train_set = r'data\processed\new_data_may\train_set_exp'

full_val_sub_data_exp = r'data\processed\new_data_may\val_sub_set_exp'
full_val_sub_data_exp_v2 = r'data\processed\new_data_may\val_set_all' 
image_size = 128

resnet_2_shot_path = 'models/protonet_episodal_2_shot_resnet18_1307_v1.pth'
episodal_5_shot_path = r'models/protonet_episodal_5_shot_resnet18_1307_norm_v1.pth'

#model_PATH = 'models/protonet_episodal_2_shot_meta_train_v2.pth'
model_PATH = 'models/protonet_episodal_5_shot_meta_train_v2.pth'
filename_pth = 'models/model_partial_resnet18_fsl_2_class_cuda_100_v1.pth'
resnet_pth = 'models/model_partial_resnet18_2_class_cuda_100_1307_v1.pth'
norm_model = r'models/protonet_episodal_2_shot_resnet18_1307_norm_v1.pth'
model_store_path22 = r'models/protonet_episodal_miniimagenet_5_v1.pth'
data_dir = r'models/protonet_episodal_2_shot_meta_train_v2.pth'
proto_dir = r'data\processed\output\train'
vali_dir = r'data\processed\validate_images'
shot_5_path = r'models/protonet_episodal_5_shot_meta_train_v2.pth'

resnet_25_path = 'models/model_partial_resnet18_2_class_cuda_25_1407_bicubic_v1.pth'
model_store_path = r'models/protonet_episodal_5_shot_resnet18_1407_bicubic_v1.pth'

pretrained_proto_net_sgd_1 = r'models/protonet_1_shot_resnet18_pretrained_1407_v1.pth'

pretrained_proto_net_sgd_5 = r'models/protonet_5_shot_resnet18_pretrained_1407_sgd_v2.pth'

model_store_path_custom_sgd_1 = r'models/protonet_1_shot_resnet18_custom_1407_v1.pth'

model_store_path_custom_sgd_2 = r'models/protonet_2_shot_resnet18_custom_1407_v1.pth'
model_store_path_custom_adam_2 = r'models/protonet_2_shot_resnet18_custom_1407_adam_v1.pth'

model_store_path_custom_sgd_5 = r'models/protonet_5_shot_resnet18_custom_1407_sgd_v1.pth'
model_store_path_pre_adam_5 =  r'models/protonet_5_shot_resnet18_custom_1407_sgd_v1.pth'

model_store_path_custom_adam_5 = r'models/protonet_5_shot_resnet18_custom_1407_adam_v1.pth'

model_store_path_custom_adam_5_v2 = r'models/protonet_5_shot_resnet18_custom_1407_adam_v2.pth'
model_store_path_custom_adam_5_v3 = r'models/protonet_5_shot_resnet18_custom_1407_adam_v3.pth'

model_store_path_custom_adam_1 = r'models/protonet_1_shot_resnet18_custom_1407_adam_v1.pth'
model_store_path_pretrained_adam_1 = r'models/protonet_1_shot_resnet18_pretrained_1407_adam_v1.pth'

model_store_path_pretrained_adam_2 = r'models/protonet_2_shot_resnet18_pretrained_1407_adam_v1.pth'

model_store_path_pretrained_adam_5_v5 = r'models/protonet_5_shot_resnet18_pretrained_1407_adam_v5.pth'

resnet_path = resnet_25_path
proto_model_PATH = model_store_path22

N_SHOT = 5 # Number of images per class
N_QUERY = 5 # Number of images per class in the query set
N_EVALUATION_TASKS = 100

val_dir = full_val_data

test_classes = os.listdir(val_dir)
N_WAY_TEST = len(test_classes) # Number of classes

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]


val_data = datasets.ImageFolder(root = val_dir, transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(data_mean, data_std),
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


    def get_prototypes(self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor):
        
        z_support = self.backbone.forward(support_images)
        n_way = len(torch.unique(support_labels))

        mean_prototypes = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(dim = 0)
                
                for label in range(n_way)
            ]
        )
        return mean_prototypes

    def calculate_distance(self, mean_prototypes, query_images):
        z_query = self.backbone.forward(query_images)
        dist1 = torch.cdist(z_query, mean_prototypes)
        return -dist1
        
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        mean_prototypes: None
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        z_query = self.backbone.forward(query_images)

        if not mean_prototypes:
            mean_prototypes = self.get_prototypes(support_images,support_labels)
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
        dists = torch.cdist(z_query, mean_prototypes)

        
        #dists = cosinesimilarity(z_query, z_total)
        scores = -dists
        
        return scores, mean_prototypes


convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
#convolutional_network.load_state_dict(torch.load(resnet_path))
model = PrototypicalNetworkModel(convolutional_network)
model.to(device)
model.load_state_dict(torch.load(proto_model_PATH))
model.eval()

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    class_prototypes: torch.Tensor
):
    """
    evaluation function
    """
    model_scores, comp_prototypes =  model(
            support_images,
            support_labels,
            query_images,
            class_prototypes
        )
    class_inf = torch.max(model_scores.data
       ,1,
    )[1].tolist()

    return (
        torch.max(model_scores, 1,)[1] == query_labels
    ).sum().item(), len(query_labels), class_inf, query_labels.tolist(), comp_prototypes

def evaluate(data_loader: DataLoader, class_prototypes: torch.Tensor):
    """
    Evaluates the model 
    """   
    best_accuracy = 0
    total_predictions = 0
    correct_predictions = 0
    exact = []
    predicted = []
    pred_list = []
    best_prototypes = None

    if class_prototypes is not None:
        class_prototypes.to(device)
    model.eval()
    log_frequency = 10
    with torch.no_grad():
        for episode_index, (support_images,support_labels,query_images,query_labels,class_ids,) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total, predicted_classes, exact_classes, comp_prototypes = evaluate_on_one_task(
                support_images.to(device),
                support_labels.to(device),
                query_images.to(device),
                query_labels.to(device), 
                class_prototypes)
            exact.extend(exact_classes)
            predicted.extend(predicted_classes)
            pred_list.append(correct)
            total_predictions += total
            correct_predictions += correct
            
            if episode_index % log_frequency == 0:
                
                model_accuracy = (100 * correct_predictions/total_predictions)
                if model_accuracy > best_accuracy:
                    best_accuracy = model_accuracy
                    best_prototypes = comp_prototypes
                    print(
                        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
                    )

    return exact, predicted, model_accuracy


exact, predicted, model_accuracy = evaluate(test_loader, None)

print(model_accuracy)


def find_classes(val_dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(val_dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx

#cf_mat = confusion_matrix(exact, predicted)

def create_cf_plot():
    '''Creates a confusion matrix of the model predictions '''

    plt.figure(figsize=(10,10))
    classes_idx = find_classes(val_dir)


    cf_mat = confusion_matrix(exact, predicted, normalize='true')
    class_names = list(classes_idx.keys())
    tick_class_labels = []
    for vessel_classes in class_names:        tick_class_labels.append(vessel_classes.replace('_', ' ').capitalize())
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
    #plt.xticks(rotation=45)
    return plt

plt = create_cf_plot()
plt.show()
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
    #ax.set_yticks(np.arange(len(list_classes)), list_classes) 
    #ax.set_xticks(x_ticks_val, x_ticks_val)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False) 
    plt.imshow(
        torchvision.utils.make_grid(
            example_support_images, nrow=N_SHOT
        ).permute(1, 2, 0)
    )
    return plt

#plot1 = plot_images(test_loader, 5)
#plot1.show()


# proto_dataset = datasets.ImageFolder(root = proto_dir, transform = transforms.Compose(
#         [
#             transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#     ),)


# val_dataset = datasets.ImageFolder(root = vali_dir, transform = transforms.Compose(
#         [
#             transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#     ),)

# proto_dataset.labels = proto_dataset.targets
# proto_sampler = TaskSampler(
#     proto_dataset, n_way=N_WAY_TEST, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
# )

# val_dataset.labels = val_dataset.targets

# proto_loader = DataLoader(
#     proto_dataset,
#     batch_sampler=proto_sampler,
#     #num_workers=8,
#     pin_memory=True,
#     worker_init_fn= seed_worker,
#     collate_fn=test_sampler.episodic_collate_fn,
# )

# val_loader = DataLoader(
#     val_dataset,
#     #batch_size= 1,
#     #num_workers=8,
#     pin_memory=True,
#     worker_init_fn= seed_worker,
#     #collate_fn=test_sampler.episodic_collate_fn,
# )

def compute_prototypes(data_loader):
    best_prototypes = None
    best_accuracy = 0
    total_predictions = 0
    correct_predictions = 0
    exact = []
    predicted = []
    pred_list = []
    class_prototypes = None
    model.eval()
    log_frequency = 10
    with torch.no_grad():
        for episode_index, (support_images,support_labels,query_images,query_labels,class_ids,) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total, predicted_classes, exact_classes, comp_prototypes = evaluate_on_one_task(support_images.to(device), support_labels.to(device), query_images.to(device), query_labels.to(device), class_prototypes)
            #opt_prototypes = comp_prototypes
            exact.extend(exact_classes)
            predicted.extend(predicted_classes)
            pred_list.append(correct)
            total_predictions += total
            correct_predictions += correct

            if episode_index % log_frequency == 0:
            
                model_accuracy = (100 * correct_predictions/total_predictions)
                print(model_accuracy)
                if model_accuracy > best_accuracy:
                    best_accuracy = model_accuracy
                    best_prototypes = comp_prototypes
                    print(
                        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
                    )

        return best_prototypes


# def get_label(val):
#     my_dict = find_classes(proto_dir)
#     for key, value in my_dict.items():
#          if val == value:
#              return key
 
#     return "There is no such Class"

# def predict_class(img):

#     x = model.calculate_distance(best_prototypes, img.to(device))
#     y = torch.max(x, 1,)[1]
    
#     return get_label(y)

# torch.load('prototypes.pt')
# iterator1 = iter(val_loader)

# for i in range (len(val_loader)):
#     train_features, train_labels = next(iterator1)
    
#     img = train_features
#     predicted_class = predict_class(img)
#     plt.imshow(train_features[0].permute(1, 2, 0))
#     plt.title(predicted_class)
#     plt.show()
#     print(predicted_class)



if __name__ == "__main__":
    #best_prototypes = compute_prototypes(test_loader)
    #torch.save(best_prototypes, 'prototypes_5_shot_mini.pt')
    pass

