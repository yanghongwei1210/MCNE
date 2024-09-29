import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix

from src.Utils import *
from src.args import get_citation_args
from src.node_classfication_evaluate1 import node_classification_evaluate

import os
import os.path as osp
from src.logreg import LogReg
from sklearn.metrics import f1_score
from scipy.sparse import csc_matrix

def train_source(args):
    mat = loadmat(args.s_dset_path)

    labels=mat['group']
    features=mat['attrb']
    A=mat['network']

    labels=np.argmax(labels,axis=1)
    feature = csc_matrix(features) if type(features) != csc_matrix else features
    
    model = get_model(args.model, features.shape[1], labels.max()+1, A, args.hidden, args.out,args)
    
    node_classification_evaluate(model, feature, A,labels, args, device=torch.device('cuda:0'))

def test_target(args):
    mat = loadmat(args.test_dset_path)

    labels=mat['group']
    features=mat['attrb']
    A=mat['network']

    labels=np.argmax(labels,axis=1)
    feature = csc_matrix(features) if type(features) != csc_matrix else features

    A=csc_matrix.toarray(A)
    A=torch.tensor(A,dtype=torch.float32)
    feature=torch.tensor(feature.toarray(),dtype=torch.float32)
    A,feature=A.cuda() ,feature.cuda()

    netF = get_model(args.model, features.shape[1], labels.max()+1, A, args.hidden, args.out,args)
    netC = LogReg(args.out, labels.max()+1).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netC.eval()


    embeds = netF(feature, A)
    logits = netC(embeds)

    preds = torch.argmax(logits, dim=1)
    labels=torch.tensor(labels)
    # val_acc = torch.sum(preds.cpu().detach().numpy() == labels.cpu().detach().numpy()).float() / labels.shape[0]
    val_f1_macro = f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
    val_f1_micro = f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

    log_str = '\nTraining: {}, Task: {}, Accuracy = {:.4f}%'.format(args.trte, args.name, val_f1_micro)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)   