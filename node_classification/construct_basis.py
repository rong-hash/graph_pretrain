from utils.data_util import create_node_classification_dataset, _create_dgl_graph
import os
import dgl
import torch
import numpy as np
import torch_geometric
from utils.learner import estimate_graphon
from sklearn.cluster import KMeans
from cal_topo import cal_topo_graphs
def load_down_data(name):
    data =create_node_classification_dataset(name).data
    num_nodes = len(data.y)
    edge_index = data.edge_index
    down_dgl = _create_dgl_graph(edge_index)
    return edge_index,num_nodes,down_dgl
def load_single_pre_data(name,load_dir = "data/single/"):
    load_path = os.path.join(load_dir, "single_{}.bin".format(name))
    g = dgl.data.utils.load_graphs(load_path)
    num_nodes = int(g[1]['graph_sizes'].item())
    edge_index = torch.stack(g[0][0].edges())
    return edge_index, num_nodes
def load_pre_data(name,load_dir):
    dataset = name
    load_dir = "data/caoyuxuan/gcc/two_data"
    load_path = os.path.join("merge_{}.bin".format(dataset))
    print(load_path)
    g = dgl.data.utils.load_graphs(load_path)
    graphs = g[0]
    edge_indices = []
    for i in range(len(graphs)):
        edge_indices.append(torch.stack(g[0][i].edges()))
    return graphs, edge_indices
def get_subgraphs(name):
    data, size_graphs, prob = load_pre_data(name)
    pre_datas = name.split("_")
    total_subgraphs = []
    for pre_data in pre_datas:
        subgraphs=[]
        edge_index,num_nodes =load_single_pre_data(pre_data)
        prob = prob.numpy()
        candidates = np.random.choice(num_nodes, size=int(num_nodes * 0.1), replace=False, p=prob)
        for t in range(len(candidates)):
            n_id = candidates[t]
            new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id),
                                                                                           num_hops=args.ego_hops,
                                                                                           edge_index=edge_index,
                                                                                           relabel_nodes=True)
            adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
            subgraphs.append(adj)
            total_subgraphs.append(adj)
        np.save("data/subgraphs/"+pre_data+".npy",np.array(subgraphs))
        return np.array(total_subgraphs)
def construct_basis(dataname):
    graphs = get_subgraphs(dataname)
    method = 'LG'
    splits = ["trivial", "topo","domain"]
    save_path = "data/graphons"
    for split in (splits):
        if split == "trivial":
            if not os.path.exists(save_path + "trivial/" + dataname):
                os.makedirs(save_path + "trivial/" + dataname)
            step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=method, args=args)
            np.save(save_path + "trivial/" + dataname+ "/graphon.npy", non_para_graphon)
            np.save(save_path + "trivial/" + dataname+ "/func.npy", step_func)
        elif split == "domain":
            pre_datas = dataname.split("_")
            for pre in pre_datas:
                if not os.path.exists(save_path + "domain/" + dataname + pre):
                    os.makedirs(save_path + "domain/" + dataname + pre)
                graphs = np.load("data/subgraphs/" +pre + ".npy").tolist()
                step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=method, args=args)
                np.save(save_path + "domain/" +  pre + "/graphon.npy", non_para_graphon)
                np.save(save_path + "domain/" +  pre + "/func.npy", step_func)
        elif split == "topo":
            if not os.path.exists(save_path + "topo/" + dataname):
                os.makedirs(save_path + "topo/" + dataname)
            topo_feats = cal_topo_graphs(graphs)
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(topo_feats)
            y_kmeans = kmeans.predict(topo_feats)
            cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(5)]
            np.save(save_path + "topo/" + dataname+ "/cluster_log.npy", cluster_graphs)
            for k, deg_graphs in enumerate(cluster_graphs):
                step_func, non_para_graphon = estimate_graphon(np.array(graphs)[deg_graphs], method=method, args=args)
                np.save(save_path + "topo/" + dataname+ "/graphon" + str(k) + ".npy",
                        non_para_graphon)
                np.save(save_path + "topo/" + dataname+ "/func" + str(k) + ".npy", step_func)
            print("Kmeans topo Done!")
  
