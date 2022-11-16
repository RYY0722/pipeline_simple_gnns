from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from models.layers import GraphConvolution
from utils import *
from torch.nn import functional as F
from collections import defaultdict
from importlib import import_module
import networkx as nx
import json
from dataset import shot_way_info, load_data
from pathlib import Path
args = get_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
num_repeat = args.num_repeat

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return F.log_softmax(x, dim=1)


def train(class_selected, id_support, id_query, n_way, k_shot):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)[id_query]
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train

def test(class_selected, id_support, id_query, n_way, k_shot):
    model.eval()
    output = model(features, adj)[id_query]
    
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':
    Path('results').mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("Using GPU acceleration... ")
       
    results=defaultdict(dict)
    meta_test_acc_total = np.zeros((num_repeat))
    meta_test_f1_total = np.zeros((num_repeat))
    # for dataset in ['email']:#['Amazon_eletronics','Amazon_clothing','dblp']:
    dataset = args.dataset
    adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict, id_by_class, degrees = load_data(dataset)
    class_list_valid = list(class_valid_dict)
    class_list_test = list(class_test_dict)
    class_list_train = list(class_train_dict)


    shot_way_pairs = shot_way_info[dataset]['pairs']

    for N, K in shot_way_pairs:
        args.way = N
        args.shot = K

        meta_test_acc_total = np.zeros((num_repeat))
        meta_test_f1_total = np.zeros((num_repeat))
        n_way = args.way
        k_shot = args.shot
        n_query = shot_way_info[dataset]['Q']
        meta_test_num = 50
        meta_valid_num = 50
        print("Training %s for on %s (%d-way %d-shot) Q: %d" % (args.model, dataset, N, K, n_query))
        for repeat in range(num_repeat):
            model =  GCN(features.shape[1], args.hidden, args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if args.cuda:
                model.cuda()
                features = features.cuda()
                adj = adj.cuda()
                labels = labels.cuda()
                degrees = degrees.cuda()
            print("Repeat %d: Training %s for on %s (%d-way %d-shot)" % (repeat, args.model, dataset, N, K))
            # Sampling a pool of tasks for validation/testing
            valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
            test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]

            # Train model
            t_total = time.time()
            meta_train_acc = []
            best_valid_acc = 0
            for episode in range(args.epochs):
                id_support, id_query, class_selected = \
                    task_generator(id_by_class, class_list_train, n_way, k_shot, n_query)
                acc_train, f1_train = train(class_selected, id_support, id_query, n_way, k_shot)
                meta_train_acc.append(acc_train)
                if (episode > 0 and episode % 100 == 0) or (episode == args.epochs-1):    
                    print("-------Episode {}-------".format(episode))
                    print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

                    # validation
                    meta_test_acc = []
                    meta_test_f1 = []
                    for idx in range(meta_valid_num):
                        id_support, id_query, class_selected = valid_pool[idx]
                        acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                        meta_test_acc.append(acc_test)
                        meta_test_f1.append(f1_test)
                    print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                                np.array(meta_test_f1).mean(axis=0)))
                    # testing
                    meta_test_acc = []
                    meta_test_f1 = []
                    for idx in range(meta_test_num):
                        id_support, id_query, class_selected = test_pool[idx]
                        acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                        meta_test_acc.append(acc_test)
                        meta_test_f1.append(f1_test)
                    valid_acc = np.array(meta_test_acc).mean(axis=0)
                    if valid_acc > best_valid_acc:
                        # best_test_accs = temp_accs
                        best_valid_acc = valid_acc
                        meta_test_acc_total[repeat] = np.array(meta_test_acc).mean(axis=0)
                        meta_test_f1_total[repeat] = np.array(meta_test_f1).mean(axis=0)
                    print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(meta_test_acc_total[repeat], meta_test_f1_total[repeat]))
                    # meta_test_acc_total[repeat] = np.array(meta_test_acc).mean(axis=0)
                    # meta_test_f1_total[repeat] = np.array(meta_test_f1).mean(axis=0)
                    print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(meta_test_acc_total[repeat], meta_test_f1_total[repeat]))

            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            print("---------- F1 ------------")
            for repeat in range(num_repeat):
                print(meta_test_f1_total[repeat])
            print("---------- Acc ------------")
            for repeat in range(num_repeat):
                print(meta_test_acc_total[repeat])
                results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)]= meta_test_acc_total[repeat]

                json.dump(results[dataset],open('./results/{}-result_{}.json'.format(args.model, dataset),'w'), indent=4) 
            accs=[]
            for repeat in range(num_repeat):
                accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)])

        results[dataset]['{}-way {}-shot'.format(N,K)]=np.mean(accs)
        results[dataset]['{}-way {}-shot_print'.format(N,K)]='acc: {:.4f}'.format(np.mean(accs))

        json.dump(results[dataset],open('./results/{}-result_{}.json'.format(args.model, dataset),'w'), indent=4)   
    print("Done :)")