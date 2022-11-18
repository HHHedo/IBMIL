import enum
import re
from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
import redis
import pickle
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from train_tcga import BagDataset
import os
import time
import numpy as np
import faiss
import torch
import sys




def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

    
def reduce(args, feats, k):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    prototypes = []
    semantic_shifts = []
    feats = feats.cpu().numpy()

    kmeans = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0)
                          for i in range(k)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(feats[assignments == i].T)
                           for i in range(k)])

    os.makedirs(f'datasets_deconf/{args.dataset}', exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes =  prototypes.reshape(-1, args.feats_size)
    print(prototypes.shape)
    print(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy')
    np.save(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy', prototypes)

    del feats


def main():
    parser = argparse.ArgumentParser(description='Clutering for abmil/dsmil/transmil')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [admil, dsmil]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--load_path', default='./', type=str, help='load path for Stage 2')
    # parser.add_argument('--dir', type=str,help='directory to save logs')
    #dsmil
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')

    args = parser.parse_args()
    # args = parser.parse_args(['--feats_size', '512','--num_classes','2', '--dataset','tcga_Img_nor'])
    '''
    ['--feats_size','512', '--num_classes','1', '--dataset','Camelyon16_Img_nor']
    ['--feats_size', '512','--num_classes','2', '--dataset','tcga_Img_nor']
    '''

    if args.model == 'dsmil':
        import dsmil as mil
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    elif args.model == 'abmil':
        import abmil as mil
        milnet = mil.Attention(in_size=args.feats_size, out_size=args.num_classes).cuda()
    elif args.model == 'transmil':
        import Models.TransMIL.net as mil
        milnet = mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes).cuda()

    # if args.dataset.startswith("tcga"):
    #     if args.feats_size == 512:
    # #         load_path = 'abmil_tcga_resnet18.pth'
    #         load_path = 'baseline/11172022/tcga_Img_nor_abmil_no_fulltune/0/1.pth'

    # elif args.dataset.startswith('Camelyon16'):
    #     if args.feats_size == 512:
    #         load_path = 'abmil_c16_resnet18'

    state_dict_weights = torch.load(args.load_path) 
    msg = milnet.load_state_dict(state_dict_weights, strict=False)
    print("***********loading init from {}*******************".format(args.load_path))
    print(msg.missing_keys)
    milnet.eval()
        
    #DATASET
    # if args.database:
    #     database = redis.Redis(host='localhost', port=6379)
    #     print('************************using database************************************')
    # else:
    #     database = None

    if args.dataset.startswith("tcga"):
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]
    elif args.dataset.startswith('Camelyon16'):
        # bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_off.csv') #offical train test
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:270, :]
        test_path = bags_path.iloc[270:, :]
            
    trainset =  BagDataset(train_path, args)
    train_loader = DataLoader(trainset,1, shuffle=True, num_workers=16)
    # testset =  BagDataset(test_path, args)
    # test_loader = DataLoader(testset,1, shuffle=False, num_workers=16)
        
    # forward
    feats_list = []
    for i,(bag_label,bag_feats) in enumerate(train_loader):
        with torch.no_grad():
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim  
            if args.model == 'abmil':
                bag_prediction, bag_feats, attention = milnet(bag_feats)
            elif args.model == 'dsmil':
                ins_prediction, bag_prediction, attention, bag_feats= milnet(bag_feats)
            elif args.model == 'transmil':
                output = milnet(bag_feats)
                bag_prediction, bag_feats ,attention=  output['logits'], output["Bag_feature"], output["A"]
         
            feats_list.append(bag_feats.cpu())
    bag_tensor = torch.cat(feats_list,dim=0)



    # bag_tensor=torch.load(f'datasets/{args.dataset}/abmil/ft_feats.pth')
    bag_tensor_ag = bag_tensor.view(-1,args.feats_size)
    for i in [2,4,8,16]:
        reduce(args, bag_tensor_ag, i)


if __name__ == '__main__':
    main()