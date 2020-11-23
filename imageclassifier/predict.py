import argparse
import torch
from modelutils import load_checkpoint
from utils import process_image
import json

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('checkpoint_dir')
parser.add_argument('--top_k', default=1, type=int)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--category_names')

args = parser.parse_args()

device = torch.device('cpu')
device_name = 'cpu'
if args.gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
model, optimizer, epoch = load_checkpoint(args.checkpoint_dir + '.pth', device_name)

with torch.no_grad():
    image = process_image(args.image_path)
    image.to(device)
    model.eval()
    logp = model(image)
    model.train()
    p = torch.exp(logp)
    
    top_ps, top_idxs = p.topk(args.top_k, dim=1)
    top_ps, top_idxs = top_ps.squeeze(0).numpy(), top_idxs.squeeze(0).numpy()

    idx_to_class = {v : k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_idxs]
    top_names = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            top_names = [cat_to_name[cat] for cat in top_classes]

    print("Top Probabilities:", top_ps, "Top Classes: ", top_classes, "Top Flowers: ", top_names)