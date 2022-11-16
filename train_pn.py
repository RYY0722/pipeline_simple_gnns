from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
from importlib import import_module
from models.GCN import model as gcn
from utils import *
from torch.nn import functional as F
from collections import defaultdict
import networkx as nx
import json
from dataset import shot_way_info, load_data
from torch.autograd import Variable
from pathlib import Path
args = get_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
num_repeat = args.num_repeat



def loop(class_selected, id_support, id_query, n_way, k_shot, train=True):
    model.train()
    optimizer.zero_grad()
    result = model(features, adj)
    embeddings, scores = result['emb'], result['score']
    z_dim = embeddings.size()[1]

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    n_query = len(id_query)
    
    # construct prototype
    prototypes = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototypes)
    p = k_shot * n_way
    data_shot, data_query = support_embeddings, query_embeddings

    proto = data_shot
    proto = proto.reshape(k_shot, n_way, -1).mean(dim=0)

    label = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])

    logits = euclidean_dist(data_query, proto)
    loss = F.cross_entropy(logits, label)
    acc = count_acc(logits, label)
    return loss,  acc


if __name__ == '__main__':
    Path('results').mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("Using GPU acceleration... ")
       
    # Model and optimizer

    # N_set=[5,10]
    # K_set=[3,5]

    results=defaultdict(dict)
    meta_test_loss_total = np.zeros((num_repeat))
    meta_test_acc_total = np.zeros((num_repeat))
    # for dataset in ['email']:#['Amazon_eletronics','Amazon_clothing','dblp']:
    dataset = args.dataset
    adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict, id_by_class, degrees = load_data(dataset)
    class_list_valid = list(class_valid_dict)
    class_list_test = list(class_test_dict)
    class_list_train = list(class_train_dict)
    # adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset)
    # adj = adj.to_dense()
    if args.model in ['GAT', 'GraghSage']:
        D = nx.DiGraph(adj.to_dense().numpy())
        edge_lst = nx.to_pandas_edgelist(D)
        edge_lst = [edge_lst['source'], edge_lst['target']]
        adj = torch.Tensor(edge_lst).long()
        del D, edge_lst
    shot_way_pairs = shot_way_info[dataset]['pairs']

    for N, K in shot_way_pairs:
        args.way = N
        args.shot = K

        meta_test_loss_total = np.zeros((num_repeat))
        meta_test_acc_total = np.zeros((num_repeat))
        n_way = args.way
        k_shot = args.shot
        n_query = shot_way_info[dataset]['Q']
        meta_test_num = 50
        meta_valid_num = 50
        print("Training %s for on %s (%d-way %d-shot) Q: %d" % (args.model, dataset, N, K, n_query))
        for repeat in range(num_repeat):
            model =  gcn(features.shape[1], args.hidden, args.dropout)
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
                model.train()
                optimizer.zero_grad()
                _loss, _acc = loop(class_selected, id_support, id_query, n_way, k_shot)
                _loss.backward()
                optimizer.step()
                meta_train_acc.append(_acc)
                if (episode > 0 and episode % 10 == 0) or (episode == args.epochs-1):    
                    print("-------Episode {}-------".format(episode))
                    print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

                    # validation
                    meta_val_acc = []
                    meta_val_loss = []
                    for idx in range(meta_valid_num):
                        id_support, id_query, class_selected = valid_pool[idx]
                        model.eval()
                        loss_val, acc_val = loop(class_selected, id_support, id_query, n_way, k_shot)
                        meta_val_loss.append(loss_val.item())
                        meta_val_acc.append(acc_val)
                    print("Meta-valid_Loss: {}, Meta-valid_Accuracy: {}".format(np.array(meta_val_loss).mean(axis=0),
                                                                                np.array(meta_val_acc).mean(axis=0)))
                    # testing
                    meta_test_loss = []
                    meta_test_acc = []
                    for idx in range(meta_test_num):
                        id_support, id_query, class_selected = test_pool[idx]
                        loss_test, acc_test = loop(class_selected, id_support, id_query, n_way, k_shot, train=False)
                        meta_test_loss.append(loss_test.item())
                        meta_test_acc.append(acc_test)
                    valid_acc = np.array(meta_val_acc).mean(axis=0)
                    if valid_acc > best_valid_acc:
                        # best_test_accs = temp_accs
                        best_valid_acc = valid_acc
                        meta_test_loss_total[repeat] = np.array(meta_test_loss).mean(axis=0)
                        meta_test_acc_total[repeat] = np.array(meta_test_acc).mean(axis=0)
                    print("Meta-Test_Loss: {}, Meta-Test_Accuracy: {}".format(meta_test_loss_total[repeat], meta_test_acc_total[repeat]))

            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            print("---------- F1 ------------")
            for repeat in range(num_repeat):
                print(meta_test_acc_total[repeat])
            print("---------- Acc ------------")
            for repeat in range(num_repeat):
                print(meta_test_loss_total[repeat])
                results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)]= meta_test_acc_total[repeat]

                json.dump(results[dataset],open('./results/{}-result_{}.json'.format(args.model, dataset),'w'), indent=4) 


            accs=[]
            for repeat in range(num_repeat):
                accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)])


        results[dataset]['{}-way {}-shot'.format(N,K)]=np.mean(accs)
        results[dataset]['{}-way {}-shot_print'.format(N,K)]='acc: {:.4f}'.format(np.mean(accs))

        json.dump(results[dataset],open('./results/{}-result_{}.json'.format(args.model, dataset),'w'), indent=4)   
    print("Done :)")