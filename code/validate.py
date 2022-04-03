#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:14:52 2022

@author: edbw
"""

#%% Standard modules
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

import torch

from torch import nn
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as F

from pathlib import Path
from datetime import datetime

from IPython.core.debugger import set_trace



#%% Our modules

up = lambda pth: pth.parent.resolve()

project_dir = up(Path(__file__))

sys.path.append( str(project_dir / 'common') )

from common.read_data import prepare_images_att
from datasets import ATTImages as att
from models import Discriminator, Generator



#%% Functions for handling configuration options
def get_config(**kwargs):
    opt = {}

    opt['device'] = 'cpu'

    # Project directories
    opt['project_dir'] = str(project_dir) # Directory containing this script and common/ folder
    opt['checkpoint_dir'] = str( project_dir / 'checkpoints' / datetime.now().strftime("%d%b%y_%H%M%S") )
    opt['data_dir'] = str(project_dir / '../data/ORL-DATABASE')

    # Pre-trained models
    opt['saved_G'] = None
    opt['saved_D'] = None

    opt.update(kwargs)
    return opt

def to_device(opts, *args):
    send = lambda x: x
    if opts['device'] != 'cpu':
        dev = torch.device(opts['device'])
        send = lambda x: x.to(dev)
    return (send(x) for x in args)



#%% Set and log run configuration to checkpoint directory

# Load models from training with MSE then with VGG5.4 for 1000 epochs
# saved_dir = project_dir / 'checkpoints' / '03Apr22_182838'
saved_dir = project_dir / 'checkpoints' / '05Apr22_232516'
opt = get_config(saved_G=str(saved_dir / 'models' / 'G_9999.pth'),
                 saved_D=str(saved_dir / 'models' / 'D_9999.pth'),
                 batch_size=8, device='cuda:0')

# Load generator from training with MSE (no VGG) only for 800 epochs
#saved_dir = project_dir / 'checkpoints' / '03Apr22_182838'
#opt = get_config(saved_G=str(project_dir / 'checkpoints' / '03Apr22_175646' / 'models' / 'G_799.pth'),
#                 saved_D=str(saved_dir / 'models' / 'D_999.pth'),
#                 batch_size=8)

# Create timestamped directory with images and logged config for this validation run
Path(opt['checkpoint_dir'] + '/images').mkdir(parents=True, exist_ok=True)

with open(opt['checkpoint_dir'] + '/config.txt', 'w') as f:
    f.write('opt = ' + str(opt).replace(', \'', ',\n\t\''))



#%% Create datasets and data loaders
seen_people_tr, seen_people_te, unseen_people = prepare_images_att(opt['data_dir'])

train_data = att(seen_people_tr)
test_data = att(seen_people_te)
validate_data = att(unseen_people)

# plt.imshow(train_data.hr[0])
# plt.imshow(train_data.lr[0])


train_loader = DataLoader(train_data, batch_size = opt['batch_size'],
                          shuffle = True, num_workers = 1)
test_loader = DataLoader(test_data, batch_size = opt['batch_size'],
                          shuffle = False, num_workers = 1)
validate_loader = DataLoader(validate_data, batch_size = opt['batch_size'],
                          shuffle = False, num_workers = 1)


#%%
from datasets import ATTImages as att


mu = np.mean(seen_people_tr.flatten())
s = np.std(seen_people_tr.flatten())
imshow((seen_people_tr[0]-mu)/s)

train_data = att(seen_people_tr)

_, imgs = next(enumerate(train_loader))

imgs_lr = make_grid(nn.functional.interpolate(imgs['lr'], scale_factor=4), nrow=1, normalize=True)
imgs_hr = make_grid(imgs['hr'], nrow=1, normalize=True)
img_grid = torch.cat((imgs_hr, imgs_lr), -1)
imshow(F.to_pil_image(img_grid), vmin=0, vmax=1)

# save_image(img_grid, (opt['checkpoint_dir'] + "/images/%d.png") % i, normalize=True)


#%% Create models, optimizers, and loss functions

G = Generator(in_channels=1, out_channels=1)
if opt['saved_G'] is not None:
    G.load_state_dict(torch.load(opt['saved_G'], map_location=torch.device(opt['device'])))

D = Discriminator(in_channels=1)
if opt['saved_D'] is not None:
    D.load_state_dict(torch.load(opt['saved_D'], map_location=torch.device(opt['device'])))



#%% Run model on test and validate datasets

is_valid = torch.ones(opt['batch_size'], 1, requires_grad=False)
is_fake = torch.zeros(opt['batch_size'], 1, requires_grad=False)

# Move everything to CUDA if necessary
G, D = to_device(opt, G, D)
is_valid, is_fake = to_device(opt, is_valid, is_fake)

total = 0
count_fake = 0
count_real= 0
for i, imgs in enumerate(validate_loader):
    imgs_lr, imgs_hr = to_device(opt, imgs['lr'], imgs['hr'])

    verify = [int(v) for v in torch.sigmoid(D(imgs_hr)) > 0.5]
    count_real = count_real + sum(verify)

    gen_hr = G(imgs_lr)
    pred = [int(v) for v in torch.sigmoid(D(G(imgs_lr).detach())) > 0.5]
    count_fake = count_fake + sum(pred)

    total = total + imgs_lr.shape[0]

    # Save image grid with upsampled inputs and SRGAN outputs
    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=False)
    gen_hr = make_grid(gen_hr, nrow=1, normalize=False)
    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=False)
    img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
    save_image(img_grid, (opt['checkpoint_dir'] + "/images/%d.png") % i, normalize=True)

sys.stdout.write('\n')
sys.stdout.write('%d generated images labeled as real out of %d\n' % (count_fake, total))
sys.stdout.write('%d real images labeled as real out of %d\n' % (count_real, total))
