import argparse
import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from load_dataset import get_dataset
from custom_model import pretrained_model

parser = argparse.ArgumentParser(description='sample argument parser')
parser.add_argument('data_dir', help='Enter data directory', type=str)
parser.add_argument('--save_dir', help='Enter save directory', type=str)
parser.add_argument('--arch', help='Enter architecture', type=str)
parser.add_argument('--learning_rate', help='Enter Learning rate', type=float)
parser.add_argument('--hidden_units', help='Enter hidden units', nargs='+')
parser.add_argument('--epochs', help='Enter epochs', type=int)
parser.add_argument('--gpu', help='Enter GPU usage', type=bool, default=False, nargs='?', const=True)

# Sets important arguments
args = parser.parse_args()
data_dir = args.data_dir
arch = args.arch if args.arch else 'vgg'
hidden_units = args.hidden_units if args.hidden_units else list()
cuda = True if args.gpu else False
save_dir = args.save_dir.strip().strip('/') if args.save_dir else ''
learning_rate = args.learning_rate if args.learning_rate else 0.002
epochs = args.epochs if args.epochs else 20
datasets, dataloaders = get_dataset(data_dir)
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

if len(save_dir) > 0:
    save_dir += '/checkpoint.pth'
else:
    save_dir = 'checkpoint.pth'
    
    
# Gets custom built model to train
model = pretrained_model(arch, hidden_units)
criterion = nn.NLLLoss()
if arch == 'resnet':
    optimiser = optim.Adam(model.fc.parameters(), lr=learning_rate)
else:
    optimiser = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Train network
model.to(device)
print('Device: ',device, '\n')

for i in range(epochs):
    running_loss = 0
    
    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()  # Reset gradient to zero
        
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
    else:
        model.eval()
        with torch.no_grad():
            train_loss = running_loss/len(dataloaders['train'])
            validation_loss = 0
            accuracy = 0

            for images, labels in dataloaders['validate']:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                validation_loss += criterion(output, labels).item()

                output = torch.exp(output)
                top_prb, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        validation_loss= validation_loss/len(dataloaders['validate'])
        accuracy = accuracy/len(dataloaders['validate'])

        print('Epoch {}/{}'.format(i+1, epochs))
        print('Train Loss: {}'.format(train_loss))
        print('Accuracy: {}%'.format(accuracy * 100))
        print('Validation Loss: {}'.format(validation_loss))
        print('-----------------------------------------------\n')
        model.train()
    
    
# TODO: Save the checkpoint
checkpoint = {
    'hidden_units': hidden_units,
    'class_to_idx': datasets['train'].class_to_idx,
    'state_dict': model.state_dict(),
    'optimiser_dict': optimiser.state_dict(),
    'epochs': epochs,
    'arch': arch
}

torch.save(checkpoint, save_dir)
    