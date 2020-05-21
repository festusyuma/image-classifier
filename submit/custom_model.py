from torch import nn, optim
from torchvision import models

def pretrained_model(arch='vgg', hidden_units=list()): 
    
    model = get_model(arch, hidden_units)
    return model


def get_model(arch, hidden_units, pretrained=True):
     
    if arch == 'resnet':
        model = models.resnet50(pretrained=pretrained)
        input_size = 2048
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        input_size = 9216
    elif arch == 'squeezenet':
        model = models.squeezenet1_0(pretrained=pretrained)
        input_size = 512
    elif arch == 'densenet':
        model = models.densenet121(pretrained=pretrained)
        input_size = 1024
    else:
        model = models.vgg19(pretrained=pretrained)
        input_size = 25088
        
    # Turns off gradient for existing trained model
    for p in model.parameters():
        p.requires_grad = False
        
    classifier = generate_classifier(input_size, hidden_units)
     
    if arch == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    return model


def generate_classifier(input_size, hidden_units):
    
    modules = []
    
    if len(hidden_units) > 0:
        hidden_units = list(map(int, hidden_units))
        
        modules.append(nn.Linear(input_size, hidden_units[0]))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.3))
        
        for i in range(1, len(hidden_units)):
            modules.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.3))
            
        modules.append(nn.Linear(hidden_units[-1], 102))
    else:
        modules.append(nn.Linear(input_size, 102))
    
    modules.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*modules)