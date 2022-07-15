
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
from torch import optim
from easyfsl.datasets import MiniImageNet

data_dir = r'C:\DTU\master_thesis\MaritimeFewShotLearning\data\mini_imagenet'

model_store_path = r'models/protonet_episodal_miniimagenet_5_v1.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 112

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# model_PATH = 'models/protonet_episodal_5_shot_meta_train_v2.pth'
# filename_pth = 'models/model_partial_resnet18_fsl_2_class_cuda_100_v1.pth'
def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


train_set = datasets.ImageFolder(root = data_dir, transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),)


# train_set = MiniImageNet(root='C:/DTU/master_thesis/MaritimeFewShotLearning/data/mini_imagenet', split="train",  transform = transforms.Compose(
#         [
#             transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()]), training=True)
# test_set = MiniImageNet(root='C:/DTU/master_thesis/MaritimeFewShotLearning/data/mini_imagenet', split="test", transform = transforms.Compose([
#             transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()]), training=False)
train_classes = os.listdir(data_dir)
N_WAY_TEST = len(train_classes) # Number of classes
N_SHOT = 5 # Number of images per class
N_QUERY = 5 # Number of images per class in the query set
N_EVALUATION_TASKS = 1000
print(len(train_set))

train_set.labels = train_set.targets
train_sampler = TaskSampler(
    train_set, n_way=N_WAY_TEST, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    #num_workers=8,
    pin_memory=True,
    worker_init_fn= seed_worker,
    collate_fn=train_sampler.episodic_collate_fn,
)

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:

    optimizer.zero_grad()
    classification_scores = model(
        support_images.to(device), support_labels.to(device), query_images.to(device), None
    )
    loss = criterion(classification_scores[0], query_labels.to(device))
    loss.backward()
    optimizer.step()
    predictions = classification_scores[0].argmax(dim=1, keepdim=True).squeeze()
    correct_preds = (predictions.int() == query_labels.to(device).int()).float()

    return loss.item(), correct_preds

from easyfsl.utils import sliding_average
log_update_frequency = 10


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
#convolutional_network.load_state_dict(torch.load(filename_pth))
model = PrototypicalNetworkModel(convolutional_network)
model.to(device)
#model.load_state_dict(torch.load(model_PATH))
model.train()

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


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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


def find_classes(val_dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(val_dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx

#cf_mat = confusion_matrix(exact, predicted)

def create_cf_plot():
    '''Creates a confusion matrix of the model predictions '''

    plt.figure(figsize=(8,8))
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

    return plt

#plt = create_cf_plot()
#plt.show()