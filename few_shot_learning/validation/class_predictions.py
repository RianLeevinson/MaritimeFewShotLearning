
import os
import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from validate import PrototypicalNetworkModel

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

image_size = 224

model_PATH = 'models/protonet_episodal_5_shot_meta_train_v2.pth'
model_PATH = 'models/protonet_episodal_5_shot_meta_train_v2.pth'
filename_pth = 'models/model_partial_resnet18_fsl_2_class_cuda_100_v1.pth'

episodal_2shot_path = r'models/protonet_episodal_2_shot_resnet18_1307_v1.pth'
resnet_path = 'models/model_partial_resnet18_2_class_cuda_100_1307_v1.pth'

pretrained_proto_net_sgd_1 = r'models/protonet_1_shot_resnet18_pretrained_1407_v1.pth'

proto_dir = r'data\processed\output\train'
vali_dir = r'data\processed\validate_images'

def find_classes(val_dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(val_dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


overleaf_dir = r'data\dataset_sample_images_overleaf'

val_dataset = datasets.ImageFolder(root = overleaf_dir, transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),)

val_dataset.labels = val_dataset.targets
val_loader = DataLoader(
    val_dataset,
    #batch_size= 1,
    #num_workers=8,
    pin_memory=True,
    worker_init_fn= seed_worker,
    #collate_fn=test_sampler.episodic_collate_fn,
)


def get_label(val):
    my_dict = find_classes(proto_dir)
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "There is no such Class"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

convolutional_network = resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()
convolutional_network.load_state_dict(torch.load(resnet_path))
model = PrototypicalNetworkModel(convolutional_network)
model.to(device)
model.load_state_dict(torch.load(model_PATH))
model.eval()

def predict_class(img, best_prototypes):

    x = model.calculate_distance(best_prototypes[:9], img.to(device))
    y = torch.topk(x, 3,)[1]
    top3_preds = y[0]
    return get_label(top3_preds[0]), get_label(top3_preds[1]), get_label(top3_preds[2])

def main():

    best_prototypes = torch.load('prototypes.pt')
    iterator1 = iter(val_loader)

    for i in range (len(val_loader)):
        train_features, train_labels = next(iterator1)
        
        img = train_features
        predicted_class1, class2, class3 = predict_class(img, best_prototypes)
        plt.imshow(train_features[0].permute(1, 2, 0))
        plt.title('Top prediction: ' + predicted_class1 + '\n' + 'Second closest class: '+ class2 +'\n'+ 'Third closest class: ' + class3, loc='left')
        plt.show()
        print(predicted_class1)

def display_img():

    #best_prototypes = torch.load('prototypes.pt')
    iterator1 = iter(val_loader)

    for i in range (len(val_loader)):
        train_features, train_labels = next(iterator1)
        plt.imshow(train_features[0].permute(1, 2, 0))
        plt.show()

if __name__ == "__main__":
    display_img()
    #main()