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
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True


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
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
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
    G = arch.Generator(config).to(device)
    E = arch.Encoder(config).to(device)

    G.apply(weights_init)
    E.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')
    #
    # Float Tensors
    #
    std_assumed = torch.tensor(config['normal_std'])
    std_assumed = std_assumed.to(device)
    log.debug(f"The Normal Std is set: {config['normal_std']}")

    #
    # Optimizers
    #
    EG_optim = getattr(optim, config['optimizer']['EG']['type'])
    EG_optim = EG_optim(chain(E.parameters(), G.parameters()),
                        **config['optimizer']['EG']['hyperparams'])

    if starting_epoch > 1:
        G.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_G.pth')))
        E.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_E.pth')))

        EG_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_EGo.pth')))
    
    E.train()
    G.train()
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        log.debug('-' * 20)
        start_epoch_time = datetime.now()
        total_eg_loss = 0.0
        total_e_loss = 0.0
        total_kld_loss = 0.0
        for i, point_data in enumerate(points_dataloader, 1):
            X, _ = point_data
            X = X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)
            codes, mu, logvar = E(X)
            X_rec = G(codes)
            loss_e = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(X.permute(0, 2, 1) + 0.5, X_rec.permute(0, 2, 1) + 0.5))

            loss_kld = -0.5 * torch.mean(
                1 - 2.0 * torch.log(std_assumed) + logvar -
                (mu.pow(2) + logvar.exp()) / torch.pow(std_assumed, 2))

            loss_eg = loss_e + loss_kld
            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss_eg.backward()
            total_eg_loss += loss_eg.item()
            total_e_loss += loss_e.item()
            total_kld_loss += loss_kld.item()
            EG_optim.step()

        log.debug(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_EG: {total_eg_loss/i: .6f} '
            f'(REC: {total_e_loss/i: .6f}'
            f' KLD: {total_kld_loss/i: .6f})'
            f'Time: {datetime.now() - start_epoch_time}'
        )

        #
        # Save intermediate results
        #
        if epoch % config['save_frequency'] == 0:

            torch.save(G.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(E.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))

            torch.save(EG_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_EGo.pth'))

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