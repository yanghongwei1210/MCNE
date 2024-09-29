import torch
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix

from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.node_classfication_evaluate import node_classification_evaluate

import logging

args = get_citation_args()

net_path =r'/home/ljx/Ljx/Graduate/MHGCN-Bern-Tong/data/citationv1.mat'
eval_name =r'/home/ljx/Ljx/Graduate/MHGCN-Bern-Tong/data/citationv1.mat'

mat = loadmat(net_path)

labels=mat['group']
features=mat['attrb']
A=mat['network']

labels=np.argmax(labels,axis=1)
feature = csc_matrix(features) if type(features) != csc_matrix else features

logging.basicConfig(filename='result/TONG_Cheb_LabelSmooth111.txt',level=logging.DEBUG)

model = get_model(args.model, features.shape[1], labels.max()+1, A, args.hidden, args.out,args)

f1_ma, f1_mi,best_micro = node_classification_evaluate(model, feature, A,labels, file_type='mat', device=torch.device('cuda:0'))

print('Test F1-ma: {:.10f}, F1-mi: {:.10f}'.format(f1_ma, f1_mi))

logging.info(args)  
logging.info(best_micro)  