import argparse
import json
import logging
import random
import re
from datetime import datetime
from importlib import import_module
import os
from os.path import join

import numpy as np
import pandas as pd
import torch

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

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    print(f'Device variable: {device}')
    if device.type == 'cuda':
        print(f'Current CUDA device: {torch.cuda.current_device()}')

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
    print(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    #
    # Models
    #
    arch = import_module(f"model.{train_config['arch']}")
    G = arch.Generator(train_config).to(device)
    G.eval()
    De = arch.Decoder(train_config).to(device)
    De.eval()
    G.load_state_dict(torch.load(args.model_g))
    De.load_state_dict(torch.load(train_config["decoder"]))

    num_samples = len(dataset)
    

    try:
        os.makedirs(join(args.outf,eval_config['experiment_name']))
    except OSError:
        pass

    # Load Gaussian Noise and Decoder
    noise = torch.FloatTensor(num_samples,train_config['prior_size']).to(device)
    noise = noise.to(device)
    print(f"The Normal Std is setted: {train_config['normal_std']}")
    train_start_time = time.time()
    print("Start Generation!")
    with torch.no_grad():
        noise.normal_(train_config["normal_mu"], train_config["normal_std"])
        fake_latent = G(noise)
        X_g = De(fake_latent)
    X_g = X_g.cpu().numpy()
    count = 0
    for pc in X_g:
        new_sample = {}
        new_sample["points"] = pc
        new_sample["semantic_label"] = args.name
        torch.save(new_sample, join(args.outf,eval_config['experiment_name'], "synthetic_2_" + eval_config['experiment_name'] + "_" + str(count)))
        count += 1
    print('Synthetic Generation finished, took {:.2f}s'.format(time.time() - train_start_time))
    

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    parser.add_argument('--model_g', type=str, required=True, help='generator path')
    parser.add_argument('--outf', type=str, required=True, help='output folder')
    parser.add_argument('--name', type=str, required=True, help='Name of Category for Generation')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None
    print(args)

    main(evaluation_config, args)
