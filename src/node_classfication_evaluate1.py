import numpy as np
import scipy.io as sio
import pickle as pkl
import torch.nn as nn
from sklearn.metrics import f1_score
import time
import random
import torch
import logging
from src.logreg import LogReg
import os.path as osp
from scipy.sparse import csc_matrix
from loss import CrossEntropyLabelSmooth

def node_classification_evaluate(model, feature, A,labels,  args, device, isTest=True):
    """Node classification training process"""
    seed = 812
    random.seed(seed)

    A=csc_matrix.toarray(A)
    A=torch.tensor(A,dtype=torch.float32)
    feature=torch.tensor(feature.toarray(),dtype=torch.float32)
    A,feature=A.to(device),feature.to(device)
    # labels=labels.ravel()
    embeds = model(feature, A)
    
    num_nodes=labels.shape[0]
    a=list(np.arange(num_nodes))
    random.shuffle(a)
    idx_train=sorted(a[:int(0.6*num_nodes)])
    idx_val=sorted(a[int(0.6*num_nodes):int(0.8*num_nodes)])
    idx_test=sorted(a[int(0.8*num_nodes):num_nodes])
    
    try:
        labels = labels.todense()
    except:
        pass
    labels = labels.astype(np.int16)
    embeds = torch.FloatTensor(embeds[np.newaxis].cpu().detach().numpy()).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    hid_units = embeds.shape[2]
    # nb_classes = labels.shape[2]
    labels=labels.long()
    nb_classes = 5
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    # train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    # val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    # test_lbls = torch.argmax(labels[0, idx_test], dim=1)
    train_lbls = labels[0, idx_train]
    val_lbls = labels[0, idx_val]
    test_lbls = labels[0, idx_test]
    labels=labels.ravel()
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []

    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        # opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.05}, {'params': log.parameters()}], lr=0.005, weight_decay=0.0005)
        opt = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': 0.0005, 'lr': 0.01},
        {'params': model.lin2.parameters(), 'weight_decay': 0.0005, 'lr': 0.01},
        {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': 0.01},
        # {'params': model.weight_b, 'weight_decay': 0.0005, 'lr': 0.011},
        {'params': log.parameters(), 'weight_decay': 0.0005, 'lr': 0.005}])
        # opt = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': 0.0005, 'lr': 0.01},
        # {'params': model.lin2.parameters(), 'weight_decay': 0.0005, 'lr': 0.01},
        # {'params': model.prop1.parameters(), 'weight_decay': 0.0005, 'lr': 0.01},
        # {'params': model.weight_b, 'weight_decay': 0.0005, 'lr': 0.011},
        # {'params': log.parameters(), 'weight_decay': 0.0005, 'lr': 0.005}])
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        best_micro=0

        starttime = time.time()
        for iter_ in range(200):
            embeds = model(feature, A)
            
            
            # embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
            train_embs = embeds[idx_train]
            val_embs = embeds[idx_val]
            test_embs = embeds[idx_test]

            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            classifier_loss = CrossEntropyLabelSmooth(num_classes=labels.max()+1, epsilon=0.1)(logits, train_lbls)
            classifier_loss.backward()
            # loss.backward()
            
            opt.step()

            logits_val = log(val_embs)
            preds = torch.argmax(logits_val, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), val_acc, val_f1_macro,
                                                              val_f1_micro))
            print("weight_b:{}".format(model.weight_b))

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            #BernNet参数
            TEST = model.prop1.temp.clone()
            theta = TEST.detach().cpu()
            theta = torch.relu(theta).numpy()
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta])

            # test
            logits_test = log(test_embs)
            preds = torch.argmax(logits_test, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("test_f1-ma: {:.4f}\ttest_f1-mi: {:.4f}".format(test_f1_macro, test_f1_micro))

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            if test_f1_micro>best_micro:
                best_netF = model.state_dict()
                best_netC = log.state_dict()
                best_micro=test_f1_micro
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}%'.format(args.name_src, iter_ , args.epochs, best_micro)
                args.out_file.write(log_str + '\n')
                args.out_file.flush()

        endtime = time.time()
        print('time: {:.10f}'.format(endtime - starttime))
        print('best_micro:',best_micro)
        # np.save('test_micro.npy',test_micro_f1s)
        np.save('embedding/{}.npy'.format(args.name_src),test_micro_f1s)
        torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
        torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])


    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s), np.mean(micro_f1s)

    return np.mean(macro_f1s), np.mean(micro_f1s),best_micro