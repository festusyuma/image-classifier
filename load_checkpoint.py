import torch

from torch import optim
from custom_model import pretrained_model


def checkpoint(checkpoint):
    checkpoints = torch.load('checkpoint.pth', map_location='cpu')
    model = pretrained_model(checkpoints['arch'], checkpoints['hidden_units'])
    model.load_state_dict(checkpoints['state_dict'])
    
    if checkpoints['arch'] == 'resnet':
        optimiser = optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        optimiser = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimiser.load_state_dict(checkpoints['optimiser_dict'])
    
    labels = [None] * len(checkpoints['class_to_idx'])
    for key in checkpoints['class_to_idx']:
        labels[checkpoints['class_to_idx'][key]] = key
    model.idx_to_class = labels
    
    return model, optimiser