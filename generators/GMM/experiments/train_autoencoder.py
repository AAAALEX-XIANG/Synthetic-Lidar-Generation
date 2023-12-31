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

from losses.champfer_loss import ChamferLoss
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
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
        reconstruction_loss = ChamferLoss().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    print("finished model")

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

    G.train()
    E.train()
    all_loss = []
    all_epoch = []
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        log.debug('-' * 20)
        start_epoch_time = datetime.now()
        total_loss = 0.0
        for i, point_data in enumerate(points_dataloader, 1):
            X, _ = point_data
            X = X.to(device)
            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)
            X_rec = G(E(X))
            loss = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))
            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()
            loss.backward()
            total_loss += loss.item()
            EG_optim.step()
        log.debug(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss: {total_loss / i:.4f} '
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
            all_loss.append(total_loss / i)
            all_epoch.append(epoch)

    best_loss = min(all_loss)
    best_epoch = all_epoch[all_loss.index(best_loss)]
    log.debug('Best Loss: ' + str(best_loss))
    log.debug('Best Epoch: ' + str(best_epoch))
    
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
