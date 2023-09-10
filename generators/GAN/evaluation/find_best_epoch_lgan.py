import argparse
import json
import logging
import random
import re
from datetime import datetime
from importlib import import_module
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import jsd_between_point_cloud_sets
from utils.util import cuda_setup, setup_logging


def _get_epochs_by_regex(path, regex):
    reg = re.compile(regex)
    return {int(w[:5]) for w in listdir(path) if reg.match(w)}


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    log.debug('Evaluating JensenShannon divergences on validation set on all '
              'saved epochs.')

    weights_path = join(train_results_path, 'weights')

    # Find all epochs that have saved model weights
    e_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_E\.pth')
    g_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_G\.pth')
    epochs = sorted(e_epochs.intersection(g_epochs))
    log.debug(f'Testing epochs: {epochs}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'nuscenes':
        from dataset.nuScenes import NuScenesData
        dataset = NuScenesData(root_dir=eval_config['data_dir'], points=train_config["n_points"])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    # Priors
    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    num_samples = len(dataset)
    jsd_noise = torch.FloatTensor(num_samples,train_config['prior_size']).to(device)
    log.debug(f"The Normal Std is set: {train_config['normal_std']}")
    #
    # Models
    #
    arch = import_module(f"model.{train_config['arch']}")
    G = arch.Generator(train_config).to(device)
    G.eval()
    De = arch.Decoder(train_config).to(device)
    De.eval()

    De.load_state_dict(torch.load(train_config["decoder"]))
    
    real_data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=eval_config['num_workers'],
                             drop_last=False, pin_memory=True)

    X, _ = next(iter(real_data_loader))
    X = X.to(device)

    results = {}
    for epoch in reversed(epochs):
        try:
            # Load Decoder
            G.load_state_dict(torch.load(
                join(weights_path, f'{epoch:05}_G.pth')))

            start_clock = datetime.now()
            #Encode data
            batch_size = eval_config['batch_size']
            # We average JSD computation from 3 independent trials.
            js_results = []
            for _ in range(3):
                jsd_noise.normal_(train_config["normal_mu"], train_config["normal_std"])
                with torch.no_grad():
                    fake_latent = G(jsd_noise)
                    X_g = De(fake_latent)
                if X_g.shape[-2:] == (3, 2048):
                     X_g.transpose_(1, 2)
                if X.shape[-2:] == (3,2048):
                    X.transpose_(1,2)
                jsd = jsd_between_point_cloud_sets(X_g.cpu().numpy(), X.cpu().numpy())
                js_results.append(jsd)
            js_result = np.mean(js_results)
            log.debug(f'Epoch: {epoch} JSD: {js_result: .6f}' 
                      f' Time: {datetime.now() - start_clock}')
            results[epoch] = js_result
            if epoch == 5000:
                break
        except KeyboardInterrupt:
            log.debug(f'Interrupted during epoch: {epoch}')
            break

    results = pd.DataFrame.from_dict(results, orient='index', columns=['jsd'])
    best_epoch = results.idxmin()['jsd']
    log.debug(f"Minimum JSD at epoch {best_epoch}: "
              f"{results.min()['jsd']: .6f} ")


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)