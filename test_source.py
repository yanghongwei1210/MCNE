import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import numpy as np
from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.node_classfication_evaluate import node_classification_evaluate
from src.logreg import LogReg
from sklearn.metrics import f1_score

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

args = get_citation_args()


net_path =r'/Users/ljx/Desktop/Graduate/ACDNE-master/ACDNE_codes/input/citationv1.mat'
eval_name =r'/Users/ljx/Desktop/Graduate/ACDNE-master/ACDNE_codes/input/citationv1.mat'

# net_path =r'/Users/ljx/Desktop/Graduate/ACDNE-master/ACDNE_codes/input/acmv9.mat'
# eval_name =r'/Users/ljx/Desktop/Graduate/ACDNE-master/ACDNE_codes/input/acmv9.mat'

# net_path =r'/Users/ljx/Desktop/Graduate/ACDNE-master/ACDNE_codes/input/dblpv7.mat'
# eval_name =r'/Users/ljx/Desktop/Graduate/ACDNE-master/ACDNE_codes/input/dblpv7.mat'
mat = loadmat(net_path)


labels=mat['group']
features=mat['attrb']
A=mat['network']


labels=np.argmax(labels,axis=1)
feature = csc_matrix(features) if type(features) != csc_matrix else features

model = get_model(args.model, features.shape[1], labels.max()+1, A, args.hidden, args.out,args)

# #测试目标域

modelpath = 'source_citationv1_F.pt'
model.load_state_dict(torch.load(modelpath))
modelpath = 'source_citationv1_C.pt'
log = LogReg(200, 5)
log.load_state_dict(torch.load(modelpath))
model.eval()

embeds = model(feature, A)
logits=log(embeds)
preds = torch.argmax(logits, dim=1)

val_f1_macro = f1_score(labels, preds.cpu().detach().numpy(), average='macro')
val_f1_micro = f1_score(labels, preds.cpu().detach().numpy(), average='micro')
print("{:.4f}\t{:.4f}".format( val_f1_macro,val_f1_micro))
# print("{:.4f}\t{:.4f}\t{:.4f}".format(val_acc, val_f1_macro,val_f1_micro))

X_tsne = TSNE(n_components=2,random_state=33).fit_transform(embeds.cpu().detach().numpy())
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels,label="t-SNE")
plt.legend()
plt.show()
print('----------------')