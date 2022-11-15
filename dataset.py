import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from utils import *
import json
from collections import defaultdict
import tqdm
from importlib import import_module
from pathlib import Path
DATASET_DIR = Path(r'D:\OneDrive - HKUST Connect\Courses\COMP5331\Project\final_data')
valid_num_dic = {'Amazon_clothing': 20, 'Amazon_eletronics': 36, 'dblp': 27}
dgl_dataset = ['CoauthorCSDataset', 'AmazonCoBuyComputerDataset', 'WikiCSDataset']
## N, K --> N-way & K-shot
shot_way_info = {'Amazon_clothing':{'pairs':[(5,3), (5,5), (3,3), (3,2) ], 'Q':10},
                 'Amazon_eletronics':{'pairs':[(5,3), (5,5), (10,5), (10,3)],'Q':10},
                 'dblp':{'pairs':[(5,3), (5,5), (10,5), (10,3)],'Q':10},
                 'email':{'pairs':[(5,3), (5,5), (7,3), (7,5)],'Q':5},
                 'reddit':{'pairs':[(5,3), (5,5),  (10,3), (10,5)],'Q':10},
                 'ogbn-arxiv':{'pairs':[(5,3), (5,5),  (10,3), (10,5)],'Q':10},
                 'cora-full':{'pairs':[(5,3), (5,5), (10,3), (10,5)],'Q':10},
                 'CoauthorCSDataset':{'pairs':[(5,3), (5,5), (3,2), (3,3)],'Q':10},
                 'AmazonCoBuyComputerDataset':{'pairs':[(3,2), (3,3)],'Q':10},
                 'WikiCSDataset':{'pairs':[(3,2), (3,3)],'Q':10},}

def load_data(dataset_source):
    class_list_train,class_list_valid,class_list_test=json.load(open(DATASET_DIR / '{}_class_split.json'.format(dataset_source)))
    if dataset_source in dgl_dataset:
        m = import_module('dgl.data')
        dataset_class = getattr(m, dataset_source)
        data = dataset_class(raw_dir=DATASET_DIR / '{}'.format(dataset_source))
        g = data[0]
        num_class = data.num_classes
        num_nodes = g.num_nodes()
        features = g.ndata['feat'].numpy()  # get node feature
        labels = g.ndata['label'].unsqueeze(-1).numpy()  # get node labels
        raw_edge = g.edges()
        ns = np.asarray([item.numpy() for item in raw_edge]).T
        ns = ns.astype(np.int32)
        n1s = ns[:,0]
        n2s = ns[:,1]
        # adj = g.adj()
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_nodes, num_nodes))    

        
        class_list = []
        for cla in labels:
            if cla not in class_list:
                class_list.append(cla)  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i[0]] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        print(num_nodes, 'nodes')
    elif dataset_source == 'email':
        ns = np.load(DATASET_DIR / "{}/{}-edge.npy".format(dataset_source, dataset_source))
        ns = ns.astype(np.int32)
        n1s = ns[:,0]
        n2s = ns[:,1]
        node_info = pkl2dic(DATASET_DIR / "{}/{}-node.pkl".format(dataset_source, dataset_source))
        id_lst, class_lst, feat_lst = node_info['id'], node_info['class'], node_info['feat']

        ## get labels
        num_nodes = len(set(id_lst))
        labels = np.zeros((num_nodes,1))
        labels[id_lst,0] = class_lst
        ## features
        feat_dim = len(feat_lst[0])
        features = np.zeros((num_nodes,feat_dim))
        features[id_lst,:] = np.asarray(feat_lst)
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    

        class_list = []
        for cla in labels:
            if cla not in class_list:
                class_list.append(cla)  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i[0]] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        # stats = {}
        # for cla, lst in id_by_class.items():
        #     stats[cla] = len(lst)
        # cnt_lst = list(stats.values())
        # cnt_lst = np.asarray(cnt_lst)
        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_source == 'reddit':

        # ns = np.loadtxt(DATASET_DIR / "{}/{}_network".format(dataset_source, dataset_source))
        ns = np.load(DATASET_DIR / "{}/{}-edge.npy".format(dataset_source, dataset_source))
        ns = ns.astype(np.int32)
        n1s = ns[:,0]
        n2s = ns[:,1]
        # num_nodes = max(max(n1s),max(n2s)) + 1
        # labels = np.zeros((num_nodes,1))
        # all_feats = np.load(r'dataset\reddit.npy')
        id_class_map = np.load(DATASET_DIR / "{}/{}-label.npy".format(dataset_source,dataset_source)).astype(np.int32) #### !!! use int32
        num_nodes = len(set(id_class_map[0]))
        labels = np.zeros((num_nodes,1))
        labels[id_class_map[0], 0] = id_class_map[1]
        # labels = np.reshape(labels, (num_nodes, 1))
        features = np.load((DATASET_DIR / "{}/{}-feats.npy".format(dataset_source, dataset_source)))
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    

        class_list = []
        for cla in labels:
            if cla not in class_list:
                class_list.append(cla)  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i[0]] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_source in valid_num_dic.keys():

        n1s = []
        n2s = []
        for line in open(DATASET_DIR / "{}/{}_network".format(dataset_source,dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        data_train = sio.loadmat(DATASET_DIR / "{}/{}_train.mat".format(dataset_source,dataset_source))
        data_test = sio.loadmat(DATASET_DIR / "{}/{}_test.mat".format(dataset_source,dataset_source))

        num_nodes = max(max(n1s),max(n2s)) + 1
        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()


        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_source=='cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph(DATASET_DIR / 'cora-full/cora_full.npz')
             
        sparse_mx = adj.tocoo().astype(np.float32)
        indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        
        n1s=indices[0].tolist()
        n2s=indices[1].tolist()
        
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        
        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj= sparse_mx_to_torch_sparse_tensor(adj)
        features=features.todense()
        features = torch.FloatTensor(features)
        labels=torch.LongTensor(labels).squeeze()
                
            
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)
        
    elif dataset_source=='ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name = dataset_source)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object

        n1s=graph['edge_index'][0]
        n2s=graph['edge_index'][1]

        num_nodes = graph['num_nodes']
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze()

        
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)

    idx_train,idx_valid,idx_test=[],[],[]
    print('Gettng index')
    for idx_,class_list_ in zip([idx_train,idx_valid,idx_test],[class_list_train,class_list_valid,class_list_test]):
        for class_ in tqdm.tqdm(class_list_):
            idx_.extend(id_by_class[class_])

    print('Gettng class_train_dict')
    class_train_dict=defaultdict(list)
    for one in tqdm.tqdm(class_list_train):
        # for i,label in tqdm.tqdm(list(enumerate(labels.numpy().tolist()))):
        for i,label in enumerate(labels.numpy().tolist()):
            if label==one:
                class_train_dict[one].append(i)
    print('Gettng class_valid_dict')
    class_valid_dict = defaultdict(list)
    for one in tqdm.tqdm(class_list_valid):
        for i, label in enumerate(labels.numpy().tolist()):
        # for i, label in tqdm.tqdm(list(enumerate(labels.numpy().tolist()))):
            if label == one:
                class_valid_dict[one].append(i)

    print('Gettng class_test_dict')
    class_test_dict = defaultdict(list)
    for one in tqdm.tqdm(class_list_test):
        # for i, label in tqdm.tqdm(list(enumerate(labels.numpy().tolist()))):
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)


    return adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict, id_by_class, degree


def neighborhoods_(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # return (adj@(adj.to_dense())+adj).to_dense().cpu().numpy().astype(int)

    hop_adj = adj + torch.sparse.mm(adj, adj)

    hop_adj = hop_adj.to_dense()
    # hop_adj = (hop_adj > 0).to_dense()

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    # prev_hop_adj = hop_adj
    # hop_adj = hop_adj + power_adj
    # hop_adj = (hop_adj > 0).float()

    hop_adj = hop_adj.cpu().numpy().astype(int)

    return (hop_adj > 0).astype(int)

    # return hop_adj.cpu().numpy().astype(int)

# def neighborhoods(adj, n_hops, use_cuda):
#     """Returns the n_hops degree adjacency matrix adj."""
#     # adj = torch.tensor(adj, dtype=torch.float)
#     # adj=adj.to_dense()
#     # print(type(adj))
#     if n_hops == 1:
#         return adj.cpu().numpy().astype(int)

#     if use_cuda:
#         adj = adj.cuda()
#     # hop_adj = power_adj = adj

#     # for i in range(n_hops - 1):
#     # power_adj = power_adj @ adj
#     hop_adj = adj + adj @ adj
#     hop_adj = (hop_adj > 0).float()

#     np.save(hop_adj.cpu().numpy().astype(int), './neighborhoods_{}.npy'.format(dataset))

#     return hop_adj.cpu().numpy().astype(int)