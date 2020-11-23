import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

def build_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
                                                  ('relu1', nn.ReLU()),
                                                  ('dropout1', nn.Dropout(0.2)),
                                                  ('fc2', nn.Linear(hidden_units, 102)),
                                                  ('output', nn.LogSoftmax(dim=1))
                                                 ]))
    return model

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path , map_location=device)

    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                                                  ('relu1', nn.ReLU()),
                                                  ('dropout1', nn.Dropout(0.2)),
                                                  ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                                                  ('output', nn.LogSoftmax(dim=1))
                                                 ]))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch