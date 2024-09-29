import torch
from scipy.io import loadmat

from src.Utils import *
from src.args import get_citation_args

import os
import os.path as osp
import random
import torch.nn as nn
import loss
from scipy.spatial.distance import cdist
import logging

from train_test import *
from src.logreg import LogReg
from src.Model import scalar

from sklearn.metrics import f1_score

def train_target(args):
    mat = loadmat(args.t_dset_path)

    # labels=mat['label']
    # features=mat['feature']
    # A=mat['edges']

    # labels= labels.ravel().astype(int)
    # labels=torch.tensor(labels)
    # features = csc_matrix(features) if type(features) != csc_matrix else features

    labels=mat['group']
    features=mat['attrb']
    A=mat['network']

    labels=np.argmax(labels,axis=1)
    labels=torch.tensor(labels)
    feature = csc_matrix(features) if type(features) != csc_matrix else features

    A=csc_matrix.toarray(A)
    A=torch.tensor(A,dtype=torch.float32)
    feature=torch.tensor(feature.toarray(),dtype=torch.float32)
    A,feature=A.to(device),feature.to(device)
     ## set base network

    netF_list = [get_model(args.model, features.shape[1], labels.max()+1, A, args.hidden, args.out,args) for i in range(len(args.src))]
    
    w = 2*torch.rand((len(args.src),))-1
    print(w)

    netC_list = [LogReg(args.out, labels.max()+1).cuda() for i in range(len(args.src))]
    netG_list = [scalar(w[i]).cuda() for i in range(len(args.src))]

    param_group = []
    # 加载预训练模型
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        # for k, v in netF_list[i].named_parameters():
        #     param_group += [{'params':v, 'lr':0.0001}]
        # for k, v in netF_list[i].named_parameters():
        #     param_group += [{'params':v, 'lr':0.001}]
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params':v, 'lr':0.002}]
        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        # 固定分类器参数
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        # for k, v in netG_list[i].named_parameters():
        #     param_group += [{'params':v, 'lr':0.001}]
        for k, v in netG_list[i].named_parameters():
            param_group += [{'params':v, 'lr':0.001}]
    
    optimizer = torch.optim.Adam(param_group, weight_decay=0.0005)
    max_iter = args.max_epoch


    iter_num = 0
    best_micro=0
    best_macro=0
    # 开始域适应
    while iter_num < max_iter:
        # 获取聚类中心
        initc = []
        all_feas = []
        for i in range(len(args.src)):
            netF_list[i].eval()
            temp1, temp2 = obtain_label(A,feature,labels, netF_list[i], netC_list[i], args)
            temp1 = torch.from_numpy(temp1).cuda()
            temp2 = torch.from_numpy(temp2).cuda()
            initc.append(temp1)
            all_feas.append(temp2)
            netF_list[i].train()

        iter_num += 1
        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(args.src), labels.shape[0], args.class_num)
        weights_all = torch.ones(labels.shape[0], len(args.src))
        outputs_all_w = torch.zeros(labels.shape[0], args.class_num)
        init_ent = torch.zeros(1,len(args.src))

        for i in range(len(args.src)):
            features_test = (netF_list[i](feature, A))[:labels.shape[0]]
            outputs_test = netC_list[i](features_test)
            softmax_ = nn.Softmax(dim=1)(outputs_test)
            ent_loss = torch.mean(loss.Entropy(softmax_))
            init_ent[:,i] = ent_loss
            weights_test = netG_list[i](features_test)
            outputs_all[i] = outputs_test
            weights_all[:, i] = weights_test.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)
        
        z_2 = torch.sum(weights_all)
        z_ = z_/z_2

        for i in range(labels.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])
        
        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp.size()).cuda()
            for i in range(len(args.src)):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu())
        else:
            classifier_loss = torch.tensor(0.0)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        
        for i in range(len(args.src)):
            netF_list[i].eval()

        acc, _,f1_macro,f1_micro = cal_acc_multi(A,feature,labels, netF_list, netC_list, netG_list, args)
        log_str = 'Iter:{}/{}; Accuracy = {:.4f}%; micro = {:.4f}, macro = {:.4f}'.format(iter_num, max_iter, acc,f1_micro,f1_macro)
        print(log_str+'\n')
        # for i in range(len(args.src)):
        #     torch.save(netF_list[i].state_dict(), osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"))
        #     torch.save(netC_list[i].state_dict(), osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"))
        #     torch.save(netG_list[i].state_dict(), osp.join(args.output_dir, "target_G_" + str(i) + "_" + args.savename + ".pt"))
        if f1_micro>best_micro:
            best_micro=f1_micro
        if f1_macro>best_macro:
            best_macro=f1_macro
    print('---------------')
    print("best_micro:",best_micro)
    print("best_macro:",best_macro)
    logging.info(args) 
    logging.info(best_micro)
    logging.info(best_macro)


def obtain_label(A,features,labels, netF, netC, args):
    # 计算预训练源模型在目标域上的准确率，执行基于聚类的标签获取
    with torch.no_grad():
        embeds = netF(features, A)
        logits = netC(embeds)
    all_fea=embeds.float().cpu()
    all_output=logits.float().cpu()
    all_label=labels.float()
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    # 对all_fea进行归一化处理，并将结果转换为Numpy数组
    # 将all_fea转置并除以其2范数，然后再次转置回去，这个操作可以将all_fea中的每个行向量除以其2范数，即将它们归一化为单位向量。
    # 拼接上的全1的那列作用是什么？
    '''
    这个操作是为了实现平移不变性。因为平移不变性是图像识别中一个很重要的性质，如果一张图像平移一下，它的内容并没有改变，只是位置发生了改变。
    因此，我们希望在特征提取后的特征向量中也具有这种平移不变性，即在向量中增加一个常数项可以达到平移不变性的效果。
    这也就是在 all_fea 中拼接上全为 1 的一列的原因，使得所有的特征向量都增加了一个常数项，从而在一定程度上实现了平移不变性。
    '''
    # all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    # 计算聚类中心
    # initc 每一行对应一个聚类中心，也即是每个类别的均值特征向量
    initc = aff.transpose().dot(all_fea)
    '''
    将初始的聚类中心矩阵 initc 进行归一化。
    归一化的目的是将所有的聚类中心向量的模长缩放到相同的尺度上，使得它们的重要性相同，从而提高聚类的准确性
    '''
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        macro = f1_score(all_label.float().numpy(), pred_label, average='macro')
        micro = f1_score(all_label.float().numpy(), pred_label, average='micro')

    log_str = 'Accuracy = {:.2f}% -> {:.4f}%  {:.4f}%'.format(accuracy*100, micro,macro)

    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    return initc,all_fea




def cal_acc_multi(A,features,labels, netF_list, netC_list, netG_list, args):
    # start_test = True
    with torch.no_grad():
        # iter_test = iter(loader)
        # for _ in range(len(loader)):
            # data = iter_test.next()
            # inputs = data[0]
            # labels = data[1]
            # inputs = inputs.cuda()
        outputs_all = torch.zeros(len(args.src), labels.shape[0], args.class_num)
        weights_all = torch.ones(labels.shape[0], len(args.src))
        outputs_all_w = torch.zeros(labels.shape[0], args.class_num)
        
        for i in range(len(args.src)):
            feature = netF_list[i](features, A)[:labels.shape[0]]
            outputs = netC_list[i](feature)
            # features = netB_list[i](netF_list[i](inputs))
            # outputs = netC_list[i](features)
            weights = netG_list[i](feature)
            outputs_all[i] = outputs
            weights_all[:, i] = weights.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        print(weights_all.mean(dim=0))
        outputs_all = torch.transpose(outputs_all, 0, 1)

        for i in range(labels.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])

        # if start_test:
        #     all_output = outputs_all_w.float().cpu()
        #     all_label = labels.float()
        #     start_test = False
        # else:
        #     all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
        #     all_label = torch.cat((all_label, labels.float()), 0)
        all_output = outputs_all_w.float().cpu()
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == labels).item() / float(labels.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    
    f1_macro = f1_score(labels, predict, average='macro')
    f1_micro = f1_score(labels, predict, average='micro')
    
    return accuracy*100, mean_ent,f1_macro,f1_micro




args = get_citation_args()

# if args.dset == 'acm':
#     names = ['a', 'b', 'c']
#     args.class_num = 3
# if args.dset == 'imdb':
#     names = ['a', 'b', 'c']
#     args.class_num = 3
names = ['acmv9', 'dblpv7', 'citationv1']
args.class_num = 5

args.src = []
for i in range(len(names)):
    if i == args.t:
        continue
    else:
        args.src.append(names[i])

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

for i in range(len(names)):
    if i != args.t:
        continue
    folder = './data/'

    args.t_dset_path = folder +  names[args.t] + '.mat'
    args.test_dset_path = folder +  names[args.t] + '.mat'

    print(args.t_dset_path)

args.output_dir_src = []
for i in range(len(args.src)):
    args.output_dir_src.append(osp.join(args.output_src, args.src[i][0].upper()))
print(args.output_dir_src)
args.output_dir = osp.join(args.output_adapt,  names[args.t][0].upper())

if not osp.exists(args.output_dir):
    os.system('mkdir -p ' + args.output_dir)
if not osp.exists(args.output_dir):
    os.mkdir(args.output_dir)

args.savename = 'par_' + str(args.cls_par)
logging.basicConfig(filename='result/{}.txt'.format(names[args.t]),level=logging.DEBUG)

train_target(args)