from torchvision import datasets, transforms, models
import torch

def get_dataset(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_data_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_data_transforms)
    image_datasets['validate'] = datasets.ImageFolder(valid_dir, transform=test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    dataloaders['validate'] = torch.utils.data.DataLoader(
        image_datasets['validate'],
        batch_size=64, shuffle=True
    )
    
    return image_datasets, dataloaders