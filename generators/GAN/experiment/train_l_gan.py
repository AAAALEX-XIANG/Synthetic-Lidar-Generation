import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists


import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel

import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True


def compute_gradient_penalty(real_samples, fake_samples, D):
        """Calculates the gradient penalty loss for WGAN GP"""
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def main(config):
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    results_dir = prepare_results_dir(config)
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger(__name__)

    device = cuda_setup(config['cuda'], config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'nuscenes':
        from dataset.nuScenes import NuScenesData
        dataset = NuScenesData(root_dir=config['data_dir'], points=config["n_points"])
    else:
        raise ValueError(f'Invalid dataset name. Expected `nuScenes` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.debug("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'],
                                   drop_last=True, pin_memory=True)

    #
    # Models
    #
    arch = import_module(f"model.{config['arch']}")
    E = arch.Encoder(config).to(device)
    G = arch.Generator(config).to(device)
    D = arch.Discriminator(config).to(device)

    E.load_state_dict(torch.load(config["encoder"]))
    G.apply(weights_init)
    D.apply(weights_init)
    E.eval()
    #
    # Float Tensors
    #
    noise = torch.FloatTensor(config['batch_size'], config['prior_size'])
    noise = noise.to(device)
    log.debug(f"The Normal Std is set: {config['normal_std']}")
    #
    # Optimizers
    #
    G_optim = getattr(optim, config['optimizer']['G']['type'])
    G_optim = G_optim(G.parameters(), lr = config['optimizer']['G']['hyperparams']['lr'])
    D_optim = getattr(optim, config['optimizer']['D']['type'])
    D_optim = D_optim(D.parameters(), lr = config['optimizer']['D']['hyperparams']['lr'])

    if starting_epoch > 1:
        G.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_G.pth')))
        D.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_E.pth')))

        G_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Go.pth')))
        D_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Do.pth')))
    
    num_batch = len(dataset)/config['batch_size']

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        total_D_loss = 0.0
        total_G_loss = 0.0
        log.debug('-' * 20)
        for i, data in enumerate(points_dataloader, 1):
            #Real Latent Variables
            X, _ = data
            X = X.to(device)
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)
            real_z = E(X)
            #Fake Latent Variables
            noise.normal_(config["normal_mu"], config["normal_std"])
            fake_z = G(noise)

            #Train D
            D_optim.zero_grad()
            real_pred = D(real_z)
            fake_pred = D(fake_z)

            gradient_penalty = compute_gradient_penalty(real_z, fake_z, D)
            loss_D = -torch.mean(real_pred) + torch.mean(fake_pred) + config["gp_lambda"] * gradient_penalty

            loss_D.backward()
            D_optim.step()
            total_D_loss += loss_D.item()
            
            #Train G
            G_optim.zero_grad()
            noise.normal_(config["normal_mu"], config["normal_std"])
            fake_z = G(noise)
            fake_pred = D(fake_z)
            loss_G = -torch.mean(fake_pred)

            loss_G.backward()
            G_optim.step()

            total_G_loss += loss_G.item()
        
        log.debug(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_D: {total_D_loss / i:.6f} '
            f'Loss_G: {total_G_loss / i:.6f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )
        
        #
        # Save intermediate results
        #
        if epoch % config['save_frequency'] == 0:

            torch.save(G.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(D.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))

            torch.save(D_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_Do.pth'))
            torch.save(G_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_Go.pth'))

    log.debug("Training Finished")


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)

    