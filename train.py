import argparse
from symbol import parameters
import numpy as np
import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
import pandas as pd
from ETFHead import ETFHead
from network import ResNet_regression
from mlp import MLPFFNNeck
from dataset import IMDBWIKI
from utils import AverageMeter, accuracy, adjust_learning_rate,shot_metric, shot_metric_balanced, shot_metric_cls, \
    setup_seed, balanced_metrics, gmean
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--dataset', type=str, default='imdb_wiki',
                    choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data', help='data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=0,
                    help='number of workers used in data loading')
parser.add_argument('--model_depth', type = int, default=50)
#############################################
#
# Here is the param you need to tune
#
#############################################
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--etf_weight', default=1, type=float)
parser.add_argument('--ce_weight', default=0, type=float)



def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    ##### how to orgnize the datastes
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                             split='train', group_num=args.groups,  reweight=None)
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size,
                           split='val', group_num=args.groups)
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size,
                            split='test', group_num=args.groups)
    #
    #train_group_cls_num = train_dataset.get_group()
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    train_labels = df_train['age']
    return train_loader, test_loader, val_loader, train_labels


def train_one_epoch(model, train_loader, opt, args, etf, e):
    etf_weight, ce_weight = args.etf_weight, args.ce_weight
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    model.train()
    #print('---------epoch: ', e, '---------')
    for idx, (x,y,g,_) in enumerate(train_loader):
        x, y, g = x.to(device), y.to(device), g.to(device)
        y_hat, z = model(x)
        #y_chunk = torch.chunk(y_output, 2, dim=1)
        #y_hat, g_hat = y_chunk[0], y_chunk[1]
        y_predicted = torch.gather(y_hat, dim=1, index=g.to(torch.int64))
        loss = 0
        loss_mse = mse(y_predicted, y)
        loss_etf = etf(z, g)
        loss_ce = ce(z, g.squeeze().long())
        loss = loss_mse + etf_weight*loss_etf + ce_weight*loss_ce
        opt.zero_grad()
        loss.backward()
        opt.step()
        #if idx % 200 == 0:
        #    print('batch: ', idx)

    return model


def test(model, test_loader, train_labels, etf, args):
    model.eval()
    mse = nn.MSELoss()
    mse_gt = AverageMeter()
    mse_pred = AverageMeter()
    acc_g = AverageMeter()
    mae_gt = AverageMeter()
    mae_pred = AverageMeter()
    criterion_gmean_gt = nn.L1Loss(reduction='none')
    criterion_gmean_pred = nn.L1Loss(reduction='none')
    gmean_loss_all_gt, gmean_loss_all_pred = [], [] 
    pred_gt, pred, labels,  pred_g_gt, pred_g = [], [], [], [], []
    with torch.no_grad():
        for idx, (inputs, targets, group) in enumerate(test_loader):
            bsz = targets.shape[0]
            inputs, targets, group = inputs.to(device), targets.to(device), group.to(device)
            labels.extend(targets.data.cpu().numpy())
            pred_g_gt.extend(group.data.cpu().numpy())
            y_hat, z = model(inputs.to(torch.float32))
            z_norm = etf.pre_logits(z)
            cls_score = z_norm @ etf.etf_vec 
            #y_chunk = torch.chunk(y_output, 2, dim=1)
            #g_hat, y_hat = y_chunk[0], y_chunk[1]
            g_hat = torch.topk(cls_score, dim=-1, k=1).indices
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            #print(f' etf {etf.etf_vec}')
            #print(f' z_norm {z_norm}')
            #print(f' the cls score {cls_score}')
            #print(f' the g_hat is {g_hat}')
            group = group.to(torch.int64)
            y_gt = torch.gather(y_hat, dim=1, index=group)
            pred_gt.extend(y_gt.data.cpu().numpy())
            y_pred = torch.gather(y_hat, dim=1, index=g_index)
            pred.extend(y_pred.data.cpu().numpy())
            #
            mse_y_gt = mse(y_gt, targets)
            mse_y_pred = mse(y_pred, targets)
            mae_gt_loss = torch.mean(torch.abs(y_gt - targets))
            mae_pred_loss = torch.mean(torch.abs(y_pred - targets))
            acc3 = accuracy(g_hat, group, topk=(1,))
            loss_all_gt = criterion_gmean_gt(y_gt, targets)
            loss_all_pred = criterion_gmean_pred(y_pred, targets)
            gmean_loss_all_gt.extend(loss_all_gt.cpu().numpy())
            gmean_loss_all_pred.extend(loss_all_pred.cpu().numpy())
            mse_gt.update(mse_y_gt.item(), bsz)
            mse_pred.update(mse_y_pred.item(), bsz)
            acc_g.update(acc3[0].item(), bsz)
            mae_gt.update(mae_gt_loss.item(), bsz)
            mae_pred.update(mae_pred_loss.item(), bsz)
        gmean_gt = gmean(np.hstack(gmean_loss_all_gt), axis=None).astype(float)
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        shot_dict_pred = shot_metric(pred, labels, train_labels)
        shot_dict_gt = shot_metric(pred_gt, labels, train_labels)
        # shot_dict_cls = shot_metric_cls(pred_g, pred_g_gt, train_labels,  labels)
        #with open('group.pkl', 'wb') as f:
        #    pickle.dump(pred_g_gt, f)
            #
        return [mse_gt.avg,  mse_pred.avg, acc_g.avg, mae_gt.avg, mae_pred.avg,\
                                    shot_dict_pred, shot_dict_gt, gmean_gt, gmean_pred]   #shot_dict_cls,



def load_model(args):
    model = ResNet_regression(args).to(device)
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, test_loader, val_loader,  train_labels = get_dataset(
        args)
    model = load_model(args)
    print(model)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    etf = ETFHead(args.groups, model.output_shape, device)
    for e in range(args.epoch):
        model = train_one_epoch(model, train_loader, opt, args, etf, e)
        if e % 10 == 0:
            mse_gt,  mse_pred, acc_g, mae_gt, mae_pred,\
                                    shot_dict_pred, shot_dict_gt, gmean_gt, gmean_pred = test(model, test_loader, train_labels, etf, args)  #shot_dict_cls,
        print(f'current epoch is {e}')
        print(f' group prediction is {acc_g}')
        print(f' the result of gt is : mse of gt is {mse_gt}, mae of gt is {mae_gt}')
        print(' Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_gt['many']['l1'],shot_dict_gt['median']['l1'], shot_dict_gt['low']['l1']))
        print(f' the result of pred is : mse of gt is {mse_pred}, mae of gt is {mae_pred}')
        print(' Prediction Many: MAE {} Median: MAE {} Low: MAE {} '.format(shot_dict_pred['many']['l1'],shot_dict_pred['median']['l1'], shot_dict_pred['low']['l1']))









        
