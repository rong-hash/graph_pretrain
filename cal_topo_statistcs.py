import torch
import torch_geometric
from learner import estimate_graphon
from load_utils import MoleculeDataset
import argparse
from tqdm import tqdm
import os
import math
import networkx as nx
from sklearn.cluster import SpectralClustering,KMeans
import numpy as np
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--split_num', type=int, default=1, help='number of splits for pre-training datasets',
                    choices=[0,1,2,3,4])
parser.add_argument('--split_ids', type=list, default="0")
args = parser.parse_args()
file_path ="dat/dataset/"
save_path ="data/graphons/"
topo_dataset = ['zinc_standard_agent']
def z_norm(sequence):
    return (np.array(sequence) - np.mean(sequence)) / np.std(sequence)
def cal_topo(file_path,dataname):
    dataset = MoleculeDataset(file_path + dataname, dataset=dataname)
    graph_ids=[]
    construct_num = args.split_num
    for i in range(construct_num):
        graph_ids+=np.load("data/dataset/"+dataname+"/split"+args.split_ids[i]+".npy").tolist()
    datas = dataset[graph_ids]
    predata_splits = ""
    for i in range(args.split_num):
        predata_splits+=args.split_ids[i]
    graphs = []
    deg_dises = []
    for j in tqdm(range(len(datas))):
        num_nodes = datas[j].num_nodes
        degree_dis = np.array(torch_geometric.utils.degree(datas[j].edge_index[0], num_nodes=num_nodes))
        deg_dises.append(degree_dis)
        matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,num_nodes=datas[j].num_nodes).toarray()
        graphs.append(matrix_graph)
        if not os.path.exists(save_path + "topo/" + dataname+ predata_splits):
            os.makedirs(save_path + "topo/" + dataname + predata_splits)
            graph_feats =[]
            for j in tqdm(range(len(datas))):
                avg_degree = float(sum(deg_dises[j]) / datas[j].num_nodes)
                std = np.std(deg_dises[j])
                nxgraph = nx.from_numpy_array(graphs[j])
                density = nx.density(nxgraph)
                closeness_centrality = sum(nx.closeness_centrality(nxgraph).values()) / datas[j].num_nodes
                degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(nxgraph)
                degree_assortativity_coefficient = nx.degree_assortativity_coefficient(nxgraph)
                transitivity = nx.transitivity(nxgraph)
                avg_clu_co = nx.average_clustering(nxgraph)
                topo_vec = [avg_degree,std,density,closeness_centrality,degree_pearson_correlation_coefficient,degree_assortativity_coefficient,transitivity,avg_clu_co]
                graph_feats.append(topo_vec)
            np.save(save_path + "topo/" + dataname + predata_splits + "pre.npy", graph_feats)
            graph_feats = np.squeeze(graph_feats)
            graph_feats = [list(z_norm(graph_feats[:, i])) for i in range(len(graph_feats[0]))]
            graph_feats=np.squeeze(graph_feats).reshape(-1,len(topo_vec)).tolist()
            np.save(save_path + "topo/" + dataname + predata_splits + ".npy", graph_feats)
for i in range(len(topo_dataset)):
    cal_topo(file_path,topo_dataset[i])

