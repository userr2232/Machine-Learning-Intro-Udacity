import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def load_dataloader(data, batch_size):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

def load_transform(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {'train_transform': transforms.Compose([transforms.Resize(255),
                                                          transforms.CenterCrop(224),
                                                          transforms.RandomRotation(30),
                                                          transforms.RandomResizedCrop(224),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean, std)]),
                  'test_valid_transform': transforms.Compose([transforms.Resize(255),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean, std)])}
    
    image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transform']),
                  'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_valid_transform']),
                  'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['test_valid_transform'])}
    
    batch_size = 64
    dataloaders = {'trainloader': load_dataloader(image_datasets['train_data'], batch_size),
              'testloader': load_dataloader(image_datasets['test_data'], batch_size),
              'validloader': load_dataloader(image_datasets['valid_data'], batch_size)}
    
    return dataloaders, image_datasets

def process_image(image_path):
    image = Image.open(image_path)
    size = image.size
    new_size = (256, size[1] * 256 // size[0]) if size[0] <= size[1] else (size[0] * 256 // size[1], 256)
    image = image.resize(new_size)
    x, y = (new_size[0]-224)//2, (new_size[1]-224)//2
    box = (x, y, x + 224, y + 224)
    image = image.crop(box)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np.array(image) / 255

    np_image = (np_image - mean) / std
    
    return torch.from_numpy(np_image.transpose((2,0,1))).float().unsqueeze(0)