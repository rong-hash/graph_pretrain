import torch_geometric
from utils.learner import estimate_graphon
from utils.load_utils import MoleculeDataset
import argparse
import os
from cal_topo import cal_topo_graphs
from sklearn.cluster import KMeans
import numpy as np
def construct_basis(args):
    dataname = args.pre_data
    file_path = args.file_path
    save_path = args.save_path
    load_path = args.load_path
    method = args.method
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
    splits = ["trivial","topo","domain"]
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
            topo_feats = cal_topo_graphs(graphs)
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
