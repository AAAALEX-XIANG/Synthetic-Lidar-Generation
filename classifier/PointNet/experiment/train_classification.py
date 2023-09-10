from __future__ import print_function
import argparse
from ast import Return
from logging import root
import os
import time
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset.nuScenes import NuScenesData
from model.pointnet import PointNetCls, feature_transform_regularizer
from tqdm import tqdm
from pytorchtools import EarlyStopping
import torch.nn.functional as F
import sklearn.metrics as metrics
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2048, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--val_dataset', type=str, required=True, help="validate dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', type=str, required=True, help="use feature transform")
parser.add_argument('--patience', type=int, default = 50, help="Early Stopping Patience")
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

print("Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

device = torch.device('cuda:0')

if opt.dataset_type == 'nuscenes':
    dataset = NuScenesData(
        root_dir=opt.dataset,points=opt.num_points)
    val_dataset = NuScenesData(
        root_dir=opt.val_dataset,points=opt.num_points)

else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

valdataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.feature_transform == "True":
    opt.feature_transform = True
elif opt.feature_transform == "False":
    opt.feature_transform = False
else:
    print("Wrong Feature Transform")
    exit

print(opt.feature_transform)
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
num_validate_batch = len(val_dataset) / opt.batchSize

early_stopping = EarlyStopping(patience=opt.patience)
for epoch in range(opt.nepoch):
    train_start_time = time.time()
    train_loss = 0.0
    count = 0.0
    train_pred = []
    train_true = []
    for i, data in enumerate(dataloader, 0):
        points, target = data
        batch_size = points.size()[0]
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        count += batch_size
        train_loss += loss.item() * batch_size
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        train_true.append(target.cpu().numpy())
        train_pred.append(pred_choice.detach().cpu().numpy())
        correct = pred_choice.eq(target.data).cpu().sum()
        #print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(len(pred_choice))))
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_accuracy = metrics.accuracy_score(train_true, train_pred)
    train_loss = train_loss*1.0/count
    print(str(epoch) + ' Epoch finished, took {:.2f}s'.format(time.time() - train_start_time))
    print(str(epoch) + ' Loss: ' + str(train_loss)) 
    print(str(epoch) + ' Accuracy: ' + str(train_accuracy)) 
    # Save model parameters
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    scheduler.step()
    print("Start Validation")
    test_pred = []
    test_true = []
    test_loss = 0.0
    count = 0.0
    validate_start_time = time.time()
    for i, data in enumerate(valdataloader, 0):
        points, target = data
        batch_size = points.size()[0]
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()

        pred, _, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        count += batch_size
        test_loss += loss.item() * batch_size
        pred_choice = pred.data.max(1)[1]
        test_true.append(target.cpu().numpy())
        test_pred.append(pred_choice.detach().cpu().numpy())
        correct = pred_choice.eq(target.data).cpu().sum()
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_loss = test_loss * 1.0 / count
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    print(str(epoch) + ' Epoch validated, took {:.2f}s'.format(time.time() - validate_start_time))
    print(str(epoch) + ' Validation Loss: ' + str(test_loss)) 
    print(str(epoch) + ' Validation Accuracy: ' + str(test_accuracy)) 
    print("-"*50, flush = True)

    early_stopping(-test_accuracy, epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Best Validation Accuracy: ", early_stopping.best_score)
        print("Best Epoch on Validation: ",early_stopping.best_epoch)
        break

