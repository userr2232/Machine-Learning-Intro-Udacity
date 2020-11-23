import argparse
import torch
from torch import nn, optim
from utils import load_transform
from modelutils import build_model

parser = argparse.ArgumentParser()
parser.add_argument('data_directory')
parser.add_argument('--save_dir')
parser.add_argument('--arch', default='vgg13', choices=['vgg13', 'vgg16'])
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_units', default=4096, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

dataloaders, datasets = load_transform(args.data_directory)
model = build_model(args.arch, args.hidden_units)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


if args.gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = args.epochs
trainloader = dataloaders['trainloader']
validloader = dataloaders['validloader']

for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
    else:
        with torch.no_grad():
            model.eval()
            test_loss = 0
            accuracy = 0
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                logps = model.forward(images)
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                
                equals = top_class == labels.view(*top_class.shape)
                test_loss += criterion(logps, labels).item()
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()
            print("Epoch: {}/{}".format(epoch+1, epochs), 
                  "Training Loss: {:.3f}..".format(running_loss/len(trainloader)), 
                  "Test Loss: {:.3f}..".format(test_loss/len(validloader)), 
                  "Accuracy: {:3f}..".format(accuracy/len(validloader)*100))

if args.save_dir:
    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': datasets['train_data'].class_to_idx
    }, args.save_dir + '.pth')