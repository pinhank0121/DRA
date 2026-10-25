"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed



# Setup

parse = parser_eval()
parse.add_argument('--eps', type=float, default=-1.0, help='Evaluate Eps, default to original training eps if not passed.')
parse.add_argument(
    '--aa-norm',
    type=str,
    default=None,
    choices=['Linf', 'L2'],
    help='Override AutoAttack norm. Defaults to the norm implied by the training attack.',
)
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'
elif args.data in ['tiny-imagenet', 'tiny-imagenets']:
    da = '/tiny-imagenet/'
elif args.data in ['imagenet']:
    da = '/imagenet/'


# DATA_DIR = os.path.join(args.data_dir, da)
DATA_DIR = args.data_dir + da
WEIGHTS = LOG_DIR + '/weights-best.pt'

# log_path = LOG_DIR + '/log-aa.log'
if args.eps > 0:
    attack_eps = args.eps
else:
    attack_eps = args.attack_eps

default_aa_norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
aa_norm = args.aa_norm if args.aa_norm is not None else default_aa_norm

log_path = LOG_DIR + f'/log-aa-{attack_eps:.3f}'
if aa_norm != default_aa_norm:
    log_path += f'-{aa_norm.lower()}'
log_path += '.log'
if os.path.exists(log_path):
    raise Exception(f'Log file {log_path} already exists. Please remove it or use a different log directory.')
logger = Logger(log_path)
info = get_data_info(DATA_DIR)

BATCH_SIZE = 256
BATCH_SIZE_VALIDATION = 256
# get number of gpu and multiply the batch size
if torch.cuda.is_available():
    # for using DataParallel
    num_gpus = torch.cuda.device_count()
    BATCH_SIZE *= num_gpus
    BATCH_SIZE_VALIDATION *= num_gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation="base", 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)


# Model
print(args.model)
if hasattr(args, 'repa_dim'):
    repa_dim = args.repa_dim
else:
    repa_dim = 256
use_repa = args.feat_reg_coeff is not None and args.feat_reg_coeff > 0.0
DP = not (args.data == 'imagenet')
model = create_model(args.model, args.normalize, info, device, repa_dim=repa_dim, use_repa=use_repa, DataParallel=DP)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
if not DP:
    model = torch.nn.DataParallel(model)
model.eval()
del checkpoint


# AA Evaluation

seed(args.seed)
logger.log(f'Using AutoAttack norm: {aa_norm} (training attack: {args.attack})')
adversary = AutoAttack(model, norm=aa_norm, eps=attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

if 'cifar10-binary' in DATA_DIR:
    # AA does not support DLR for binary classification
    adversary.attacks_to_run = ['apgd-ce', 'fab', 'square']

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')