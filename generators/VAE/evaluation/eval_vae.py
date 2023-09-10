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


from metrics import compute_all_metrics, jsd_between_point_cloud_sets
from utils.util import cuda_setup
import time

def main(eval_config, args):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])
    
    log = logging.getLogger(__name__)

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

    #
    # Models
    #
    arch = import_module(f"model.{train_config['arch']}")
    G = arch.Generator(train_config).to(device)

    G.eval()

    num_samples = len(dataset)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=eval_config['num_workers'],
                             drop_last=False, pin_memory=True)
    X, _ = next(iter(data_loader))
    X = X.to(device)
    
    # Load Decoder
    G.load_state_dict(torch.load(args.model_g))

    noise = torch.FloatTensor(num_samples, train_config['z_size'], 1)
    noise = noise.to(device)
    print(f"The Normal Std is setted: {train_config['normal_std']}")
    train_start_time = time.time()
    with torch.no_grad():
        noise.normal_(train_config["normal_mu"], train_config["normal_std"])
        with torch.no_grad():
            X_g = G(noise).to(torch.device('cuda'))
        if X_g.shape[-2:] == (3, 2048):
                X_g.transpose_(1, 2)
        if X.shape[-2:] == (3,2048):
            X.transpose_(1,2)
        results = compute_all_metrics(X_g, X,eval_config['batch_size'] )
    results = {k:v.item() for k, v in results.items()}
    jsd = jsd_between_point_cloud_sets(X_g.cpu().numpy(), X.cpu().numpy())
    results['jsd'] = jsd
    for k, v in results.items():
        print('%s: %.12f' % (k, v))
    print('Evaluating VAE finished, took {:.2f}s'.format(time.time() - train_start_time))

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    parser.add_argument('--model_g', type=str, required=True, help='generator path')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None
    print(args)

    main(evaluation_config, args)
