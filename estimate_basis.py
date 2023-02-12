import torch
import torch_geometric
from learner import estimate_graphon
from load_utils import MoleculeDataset
import argparse
import os
import math
import networkx as nx
from sklearn.cluster import SpectralClustering,KMeans
import numpy as np
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
parser.add_argument('--num-graphs', type=int,
                    default=10,
                    help='the number of synthetic graphs')
parser.add_argument('--num-nodes', type=int, default=200,
                    help='the number of nodes per graph')
parser.add_argument('--graph-size', type=str, default='random',
                    help='the size of each graph, random or fixed')
parser.add_argument('--threshold-sba', type=float, default=0.1,
                    help='the threshold of sba method')
parser.add_argument('--threshold-usvt', type=float, default=0.1,
                    help='the threshold of usvt method')
parser.add_argument('--alpha', type=float, default=0.0003,
                    help='the weight of smoothness regularizer')
parser.add_argument('--beta', type=float, default=5e-3,
                    help='the weight of proximal term')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='the weight of gw term')
parser.add_argument('--inner-iters', type=int, default=50,
                    help='the number of inner iterations')
parser.add_argument('--outer-iters', type=int, default=20,
                    help='the number of outer iterations')
parser.add_argument('--n-trials', type=int, default=2,
                    help='the number of trials')
parser.add_argument('--rw_hops', type=int, default=1,
                    help='the number of trials')
parser.add_argument('--ego_hops', type=int, default=2,
                    help='the number of ego_hops')
parser.add_argument('--pre_data', type=str, default="imdb_academia",
                    help='pretrain data path')
parser.add_argument('--down_data', type=str, default="",
                    help='downstream data path')
parser.add_argument('--split_num', type=int, default=2, help='number of splits for pre-training datasets',
                    choices=[1,2, 3, 4])
parser.add_argument('--split_ids', type=list, default="12")
args = parser.parse_args()
method = 'LG'
file_path ="data/dataset/"
save_path ="data/graphons/"
load_path ="data/graphons/"
pre_dataset= ['zinc_standard_agent']
def estimate_basis(file_path,dataname):
    dataset = MoleculeDataset(file_path + dataname, dataset=dataname)
    predata_splits = ""
    for i in range(args.split_num):
        predata_splits += args.split_ids[i]
    graph_ids = []
    for i in range(args.split_num):
        graph_ids+=np.load("data/dataset/"+dataname+"/split"+args.split_ids[i]+".npy").tolist()
    datas = dataset[graph_ids]
    graphs = []
    for j in range(len(datas)):
        matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,num_nodes=datas[j].num_nodes).toarray()
        graphs.append(matrix_graph)
    splits = ["trivial","topo"]
    for split in (splits):
        if split=="trivial":
            if not os.path.exists(save_path + "trivial/" + dataname+ predata_splits):
                os.makedirs(save_path + "trivial/" + dataname+ predata_splits)
            step_func,non_para_graphon  =estimate_graphon(np.array(graphs), method=method, args=args)
            np.save(save_path + "trivial/" + dataname+ predata_splits + "/graphon.npy", non_para_graphon)
            np.save(save_path + "trivial/" + dataname+ predata_splits + "/func.npy", step_func)
        elif split=="domain":
            for pre in predata_splits:
                if not os.path.exists(save_path + "domain/" + dataname + pre):
                    os.makedirs(save_path + "domain/" + dataname + pre)
                graph_ids=np.load("data/dataset/"+dataname+"/split"+pre+".npy").tolist()
                datas = dataset[graph_ids]
                graphs = []
                for j in range(len(datas)):
                    matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                                num_nodes=datas[j].num_nodes).toarray()
                    graphs.append(matrix_graph)
                step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=method, args=args)
                np.save(save_path + "domain/" + dataname + pre + "/graphon.npy", non_para_graphon)
                np.save(save_path + "domain/" + dataname + pre + "/func.npy", step_func)
        elif split=="topo":
            if not os.path.exists(save_path + "topo/" + dataname+ predata_splits):
                os.makedirs(save_path + "topo/" + dataname+ predata_splits)
            topos = []
            for pre in predata_splits:
                topo = np.load(load_path + "topo/" + dataname + pre + ".npy")
                topos.append(topo)
            topo_feats = np.concatenate((topos), axis=0)
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(topo_feats)
            y_kmeans = kmeans.predict(topo_feats)
            cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(5)]
            np.save(save_path + "topo/" + dataname + predata_splits + "/cluster_log.npy", cluster_graphs)
            for k, deg_graphs in enumerate(cluster_graphs):
                step_func, non_para_graphon = estimate_graphon(np.array(graphs)[deg_graphs], method=method, args=args)
                np.save(save_path + "topo/" + dataname + predata_splits + "/graphon" + str(k) + ".npy", non_para_graphon)
                np.save(save_path + "topo/" + dataname + predata_splits + "/func" + str(k) + ".npy", step_func)
            print("Kmeans topo Done!")
for i in range(len(pre_dataset)):
    estimate_basis(file_path,pre_dataset[i])
