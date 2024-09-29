import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb
from src.Bernpro import Bern_prop
from torch_geometric.utils import dense_to_sparse
from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge

from scipy.sparse import csc_matrix

class BernNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, out, dropout,dprate,K):
        super(BernNet, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, out)
        self.m = torch.nn.BatchNorm1d(out)
        self.prop1 = Bern_prop(K)
 
        self.dprate = dprate
        self.dropout = dropout

        # # Alibaba
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)

        # DBLP
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        
        # # Aminer
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=1)
        
        #IMDB
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, feature, A):
        # final_A = adj_matrix_weight_merge(A, self.weight_b)
        # final_A=(A[0][0]+A[0][1]).todense()
        # final_A=torch.tensor(final_A+final_A.transpose(),dtype=torch.float32) 
        final_A=A
        # final_A=A.toarray()
        final_A=torch.tensor(final_A,dtype=torch.float32)
        try:
            feature = torch.tensor(feature.astype(float32).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        x = feature.float()
        edge_index, edge_weight = dense_to_sparse(final_A)
        # edge_weight=abs(edge_weight)
        # edge_weight=(edge_weight+1)/2
        # AA=final_A.detach().numpy()
        # aa=np.max(AA)
        # np.savetxt('weight.txt',edge_weight.detach().numpy())

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index,edge_weight)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index,edge_weight)
            # return F.log_softmax(x, dim=1)
            return x