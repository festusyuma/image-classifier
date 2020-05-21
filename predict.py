import argparse
import numpy as np
import torch
import json

from torch import nn
from PIL import Image
from load_checkpoint import checkpoint

parser = argparse.ArgumentParser(description='sample argument parser')
parser.add_argument('image_path', help='Enter Image path', type=str)
parser.add_argument('checkpoint_path', help='Enter checkpoint path', type=str)
parser.add_argument('--top_k', help='Enter save directory', type=int)
parser.add_argument('--category_names', help='Enter architecture', type=str)
parser.add_argument('--gpu', help='Enter GPU usage', type=bool, default=False, nargs='?', const=True)

# Sets important arguments
args = parser.parse_args()
image_path = args.image_path
checkpoint_path = args.checkpoint_path
top_k = args.top_k if args.top_k else 1
cuda = True if args.gpu else False
category_names = args.category_names
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image.thumbnail((256, 256))
    width, height = pil_image.size
    
    top_crop = (height - 224) / 2
    right_crop = (width + 224) / 2
    bottom_crop = (height + 224) / 2
    left_crop = (width - 224) / 2
    crop_size = (left_crop, top_crop, right_crop, bottom_crop)
    pil_image = pil_image.crop(crop_size)
    
    np_image = np.array(pil_image) / 255
    np_image = np_image - np.array([0.485, 0.456, 0.406])
    np_image = np_image / np.array([0.229, 0.224, 0.225])
    np_image = np.transpose(np_image, (2, 0, 1))

    t_image = torch.from_numpy(np_image)
    t_image = t_image.float()
    t_image.unsqueeze_(0)
    
    return t_image


model, optimiser = checkpoint('checkpoint.pth')
criterion = nn.NLLLoss()
image = process_image(image_path)

model.to(device)
image = image.to(device)

model.eval()
with torch.no_grad():
    output = torch.exp(model(image))
    top_p, top_class = output.topk(top_k, dim=1)
    
    top_p = top_p.cpu().numpy().flatten()
    top_class = [model.idx_to_class[i] for i in top_class.cpu().numpy().flatten()]
    
model.train()

if category_names:
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        top_class = [cat_to_name[i] for i in top_class]
    
print('Result')
print('--------------------------')
for i in range(len(top_p)):
    print('Class: {} ,Probability: {}'.format(top_class[i], top_p[i]))
    print('--------------------------')