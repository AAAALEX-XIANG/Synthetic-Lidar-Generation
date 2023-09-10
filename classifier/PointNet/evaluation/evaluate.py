from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from model.pointnet import PointNetCls
import torch.nn.functional as F
from dataset.nuScenes import NuScenesData
import sklearn.metrics as metrics
import numpy as np
import random


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--test_dataset', type=str, required=True, help="test_dataset path")
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--feature_transform', type=str, required=True, help="use feature transform")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


opt = parser.parse_args()
print(opt)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
if opt.feature_transform == "True":
    opt.feature_transform = True
elif opt.feature_transform == "False":
    opt.feature_transform = False
else:
    print("Wrong Feature Transform")
    exit

test_dataset = NuScenesData(
        root_dir=opt.test_dataset,points=opt.num_points)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

classifier = PointNetCls(k=len(test_dataset.classes), feature_transform = opt.feature_transform)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

#Accuracy
test_pred = []
test_true = []
for points, target in test_dataloader:
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    test_true.append(target.cpu().numpy())
    test_pred.append(pred_choice.detach().cpu().numpy())

test_true = np.concatenate(test_true)
test_pred = np.concatenate(test_pred)
test_acc = metrics.accuracy_score(test_true, test_pred)
avg_precision = metrics.precision_score(test_true, test_pred, average = 'macro')
avg_recall = metrics.recall_score(test_true, test_pred,average = 'macro')
avg_f1 = metrics.f1_score(test_true, test_pred,average = 'macro')
outstr = 'Test :: test acc: %.6f, test avg precision: %.6f , test avg recall: %.6f, test avg f1: %.6f' % (test_acc, avg_precision, avg_recall, avg_f1)
print(outstr)