from prototypical_model import PrototypicalNetworkModel
import torch

model_PATH = 'C:\DTU\master_thesis\fsl\Object-classification-with-few-shot-learning\models'

model = PrototypicalNetworkModel(*args, **kwargs)
model.load_state_dict(torch.load(model_PATH))
model.eval()

