#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:01:50 2022

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
from models import Discriminator, Generator, VGG19Grayscale



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

    # Image and model feature, channel, size options
    opt['batch_size'] = 4
    opt['hr_shape'] = (64, 64)
    opt['channels'] = 64

    # Content loss
    opt['content_loss'] = 'vgg' # 'mse'
    opt['vgg_feature_layer'] = 29 # VGG/5.4
    # opt['vgg_feature_layer'] = 35 # VGG/5.1

    # Training and logging options
    opt['n_epochs'] = 2000 # Number of passes through the training dataset
    opt['n_save'] = 50    # Save models and images every n_save epochs
    opt['n_print'] = 32    # Print to screen every Xth batch in current epoch

    # Optimizer options
    opt['lr'] = 0.0001
    opt['beta1'] = 0.5
    opt['beta2'] = 0.999

    opt.update(kwargs)
    return opt

def to_device(opts, *args):
    send = lambda x: x
    if opts['device'] != 'cpu':
        dev = torch.device(opts['device'])
        send = lambda x: x.to(dev)
    return (send(x) for x in args)



#%% Set and log run configuration to checkpoint directory

# opt = get_config()

# assert(torch.cuda.is_available())
# opt = get_config(device='cuda')

# opt = get_config(content_loss='vgg', n_print=1, n_save=1, n_epochs=2, batch_size=8, device='cpu')

# opt = get_config(content_loss='vgg', n_print=8, n_save=10, n_epochs=500, batch_size=8, device='cuda')

# opt = get_config(content_loss='mse', n_print=8, n_save=100, n_epochs=1000,
#                   batch_size=8, device='cuda')

#opt = get_config(content_loss='vgg', n_print=8, n_save=10, n_epochs=1000,
#                  batch_size=8, device='cuda',
#                  saved_G='checkpoints/03Apr22_175646/models/G_799.pth')

opt = get_config(content_loss='vgg', n_print=8, n_save=100, n_epochs=10000,
                  batch_size=8, device='cuda', lr=0.00001,
                  saved_G='checkpoints/05Apr22_221204/models/G_999.pth')

# Create timestamped directory with images, models, and logged config
Path(opt['checkpoint_dir'] + '/models').mkdir(parents=True, exist_ok=True)
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
                          shuffle = True, num_workers = 1)
validate_loader = DataLoader(validate_data, batch_size = opt['batch_size'],
                          shuffle = True, num_workers = 1)



#%% Create models, optimizers, and loss functions

G = Generator(in_channels=1, out_channels=1)
if opt['saved_G'] is not None:
    G.load_state_dict(torch.load(opt['saved_G']))

D = Discriminator(in_channels=1)
if opt['saved_D'] is not None:
    D.load_state_dict(torch.load(opt['saved_D']))

VGG19 = VGG19Grayscale(opt['vgg_feature_layer'])

# Losses
mse = nn.MSELoss(reduction='sum')
bce = nn.BCEWithLogitsLoss()
l1_norm = torch.nn.L1Loss()

# Optimizers
g_opt = torch.optim.Adam(G.parameters(), lr=opt['lr'], betas=(opt['beta1'], opt['beta2']))
d_opt = torch.optim.Adam(D.parameters(), lr=opt['lr'], betas=(opt['beta1'], opt['beta2']))

def reset_grad():
    g_opt.zero_grad()
    d_opt.zero_grad()



#%% Training

# Labels
is_valid = torch.ones(opt['batch_size'], 1, requires_grad=False)
is_fake = torch.zeros(opt['batch_size'], 1, requires_grad=False)

# Move everything to CUDA if necessary
G, D, VGG19, bce, l1_norm = to_device(opt, G, D, VGG19, bce, l1_norm)
is_valid, is_fake = to_device(opt, is_valid, is_fake)

for epoch in range(0, opt['n_epochs']):
    for i, imgs in enumerate(train_loader):
        imgs_lr = imgs["lr"]
        imgs_hr = imgs["hr"]

        imgs_lr, imgs_hr = to_device(opt, imgs_lr, imgs_hr)

        #######################
        # Train discriminator #
        reset_grad()

        # Classification loss on real and fake images
        loss_real = bce(D(imgs_hr), is_valid)
        loss_fake = bce(D(G(imgs_lr).detach()), is_fake)
        d_loss = loss_real + loss_fake #TODO: discriminator loss is too small, maybe because we are training on grayscale

        d_loss.backward()
        d_opt.step()

        #####################
        ## Train generator ##
        reset_grad()

        gen_hr = G(imgs_lr)

        # Content loss
        if opt['content_loss'] == 'mse':
            loss_content = mse(imgs_hr, gen_hr) #TODO: do we need imgs_hr.detach() here?
        elif opt['content_loss'] == 'vgg':
            gen_features = VGG19(gen_hr)
            real_features = VGG19(imgs_hr)
            loss_content = mse(gen_features, real_features.detach())/real_features.shape[1]**2

        # Adversarial loss
        loss_adversarial = bce(D(gen_hr), is_fake)

        # Perceptual loss
        g_loss = loss_content + 1e-3 * loss_adversarial

        g_loss.backward()
        g_opt.step()

        ###################
        # Training status #
        if (i+1) %opt['n_print'] == 0:
            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                % (epoch, opt['n_epochs'], i+1, len(train_loader), d_loss.item(), g_loss.item())
            )

    if opt['n_save'] != -1 and (epoch+1) % opt['n_save'] == 0:
        # Save image grid with upsampled inputs and SRGAN outputs
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, gen_hr), -1)
        save_image(img_grid, (opt['checkpoint_dir'] + "/images/%d.png") % epoch, normalize=False)

        # Save model checkpoints
        torch.save(G.state_dict(), (opt['checkpoint_dir'] + "/models/G_%d.pth") % epoch)
        torch.save(D.state_dict(), (opt['checkpoint_dir'] + "/models/D_%d.pth") % epoch)

        # Echo to stdout
        sys.stdout.write('[Epoch %d/%d] checkpoint.\n' % (epoch, opt['n_epochs']))

