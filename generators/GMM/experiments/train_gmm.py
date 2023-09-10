import argparse
import json
import logging
import random
import re
import time
from importlib import import_module
from os.path import join
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.util import cuda_setup, setup_logging
from sklearn.mixture import GaussianMixture
import joblib


def main(eval_config, args):

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

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
    E = arch.Encoder(train_config).to(device)
    E.eval()
    num_samples = len(dataset)
    data_loader = DataLoader(dataset, batch_size=eval_config['batch_size'],
                             shuffle=False, num_workers=eval_config['num_workers'],
                             drop_last=False, pin_memory=True)
    latent_variables = np.zeros([num_samples, eval_config['z_size']])
    # Load Encoder
    E.load_state_dict(torch.load(args.model_e))
    #Encode data
    batch_size = eval_config['batch_size']
    train_start_time = time.time()
    for i, point_data in enumerate(data_loader, 0):
        X_e, _ = point_data
        X_e = X_e.to(device)
        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X_e.size(-1) == 3:
            X_e.transpose_(X_e.dim() - 2, X_e.dim() - 1)
        z = E(X_e).cpu().detach().numpy()
        latent_variables[i*batch_size:i*batch_size+batch_size] = z
    #Create GMM model
    print("Start Training GMM")
    gm = GaussianMixture(n_components=int(args.n_component)).fit(latent_variables)

    print('Training GMM finished, took {:.2f}s'.format(time.time() - train_start_time))
    joblib.dump(gm, args.outf + "/GMM_" + args.training_class + "_" + args.n_component + ".model")

if __name__ == '__main__':
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    parser.add_argument('--model_e', type=str, required=True, help='Encoder path')
    parser.add_argument('--n_component', type=str, required=True, help='Number of GMM Sub-Distributions')
    parser.add_argument('--training_class', type=str, required=True, help='Class of Samples')
    parser.add_argument('--outf', type=str, required=True, help='output folder')
    args = parser.parse_args()
    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None
    print(args)
    main(evaluation_config, args)
