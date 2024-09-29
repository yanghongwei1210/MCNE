import torch
from scipy.io import loadmat

from src.Utils import *
from src.args import get_citation_args
from src.node_classfication_evaluate import node_classification_evaluate

import os
import os.path as osp
from train_test import *
args = get_citation_args()


names = ['acmv9', 'dblpv7', 'citationv1']

folder = './data/'
args.s_dset_path = folder + names[args.s] + '.mat'
args.output_dir_src = osp.join(args.output, names[args.s][0].upper())
args.name_src = names[args.s][0].upper()
if not osp.exists(args.output_dir_src):
    os.system('mkdir -p ' + args.output_dir_src)
if not osp.exists(args.output_dir_src):
    os.mkdir(args.output_dir_src)
args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'a')
args.out_file.write(print_args(args)+'\n')
args.out_file.flush()


train_source(args)
args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'a')
for i in range(len(names)):
    if i == args.s:
        continue
    args.t = i
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    folder = 'data/'
    args.s_dset_path = folder + names[args.s] + '.mat'
    args.test_dset_path = folder + names[args.t] + '.mat'

    test_target(args)


