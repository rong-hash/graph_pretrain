import argparse
import numpy as np
import os
import torch_geometric
import torch
import time
import networkx
import math

import dgl
from sklearn.cluster import KMeans
from torch_geometric.datasets import TUDataset
from utils import create_node_classification_dataset, learner

parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--f-result', type=str,
                    default='results/merge/small',
                    help='the root path saving learning results')
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
parser.add_argument('--down_data', type=str, default="h-index",
                    help='downstream data path')
parser.add_argument('--node_threshold', type=int, default=1000,
                    help='subgraph size threshold')
parser.add_argument('--edge_threshold', type=int, default=10000,
                    help='subgraph size threshold')
parser.add_argument('--data_type', type=str, default="down_node",choices=["pre_single","pre_merge","down_node","down_graph"],
                    help='data type ')
parser.add_argument('--load_dir', type=str, default="", help='load data.bin directory ')
parser.add_argument('--save_dir', type=str, default="", help='save sampled subgraphs directory ')

parser.add_argument('--split', type=str, default="Kmeans")
parser.add_argument('--resume_graphs', action="store_true")
parser.add_argument('--domain_num', type=int, default=3)
args = parser.parse_args()


methods ='LG'

def load_pretrain_data(args):
    dataset = args.pre_data
    data_type =args.data_type
    load_dir = args.load_dir
    print(load_dir)
    if "single" in data_type:
        load_dir = "data/single/"
        load_path = os.path.join(load_dir,"single_{}.bin".format(dataset))
        print(load_path)
        g = dgl.data.utils.load_graphs(load_path)
        num_nodes = int(g[1]['graph_sizes'].item())
        edge_index = torch.stack(g[0][0].edges())
        return edge_index,num_nodes
    elif "merge" in data_type:
        load_path =os.path.join(load_dir, "merge_{}.bin".format(dataset))
        print(load_path)
        g = dgl.data.utils.load_graphs(load_path)
        graphs = g[0]
        degrees = torch.cat([graph.in_degrees().double() ** 0.75 for graph in graphs])
        print(degrees.shape)
        prob = degrees / torch.sum(degrees)
        sizes_graphs = g[1]['graph_sizes']
        return graphs, sizes_graphs,prob
def load_down_graph_data(dataset_name):
    name = {
        "imdb-binary": "IMDB-BINARY",
        "imdb-multi": "IMDB-MULTI",
        "rdt-b": "REDDIT-BINARY",
        "rdt-5k": "REDDIT-MULTI-5K",
        "collab": "COLLAB",
        "dd": "DD",
        "msrc": "MSRC_21"
    }[dataset_name]
    dataset = TUDataset('data/TUDataset',name)
    graphs = []
    for i in range(len(dataset)):
        edge_index = dataset[i].edge_index
        adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index).toarray()
        graphs.append(adj)
    return graphs

def load_down_node_data(name,save_path = None):
    save_path = "/data"
    gcc_data = ['Texas','Cornell','Wisconsin','brazil_airport','DD242','DD68','DD687','usa_airport', 'europe_airport','h-index','squirrel','chameleon']
    data =create_node_classification_dataset(name).data
    num_nodes = len(data.y)
    edge_index = data.edge_index
    return edge_index,num_nodes
def z_norm(sequence):
    return (np.array(sequence) - np.mean(sequence)) / np.std(sequence)
def cal_satatics(edge_index,num_nodes,subgraph=None):
    degrees = np.array(torch_geometric.utils.degree(edge_index[0]))
    edge_num = sum(degrees)/2
    entropy = float(sum([degrees[i] * math.log(degrees[i]) for i in range(num_nodes)]) / 2 / edge_num)
    avg_degree = float(sum(degrees) / num_nodes)
    std = np.std(degrees)
    nxgraph = networkx.from_numpy_array(subgraph)
    density = networkx.density(nxgraph)
    closeness_centrality = sum(networkx.closeness_centrality(nxgraph).values())/num_nodes
    degree_pearson_correlation_coefficient = networkx.degree_pearson_correlation_coefficient(nxgraph)
    degree_assortativity_coefficient = networkx.degree_assortativity_coefficient(nxgraph)
    transitivity = networkx.transitivity(nxgraph)
    # diameter = networkx.diameter(nxgraph)
    avg_clu_co = networkx.average_clustering(nxgraph)
    print("d_t:{}".format(d_t))
    return [entropy,avg_degree,std,density,closeness_centrality,degree_pearson_correlation_coefficient,
            degree_assortativity_coefficient,transitivity,avg_clu_co]
def split_subgraphs(num_nodes_ls,split_criterions,edge_indices=None,subgraphs = None):
    # entroys,avg_degrees,stds,densities = [],[],[],[]

    static_list = []


    bases_num=5
    split = "entropy"
    for i in range(len(edge_indices)):
        edge_index = edge_indices[i]
        num_nodes = num_nodes_ls[i]
        statistics = cal_satatics(edge_index,num_nodes,subgraphs[i])
        static_list.append(statistics)
        if i%100 ==0:
            print(i)
    static_list = np.squeeze(static_list)
    static_list = [list(z_norm(static_list[:,i])) for i in range(len(static_list[0]))]
    syn_list = np.sum(static_list,0)
    cluster_graph_dic = {}
    for i in range(len(split_criterions)):
        cluster_graphs = []
        split = split_criterions[i]
        criteria_list = []
        if split == "syn":
            criteria_list = syn_list
        elif split == "Kmeans":
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(static_list)
            #normalize
            y_kmeans = kmeans.predict(static_list)
            print(y_kmeans)
            cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(bases_num)]
            print("Kmeans Done!")
        if len(cluster_graphs) ==0:
            sorted_criteria_list = sorted(range(len(criteria_list)), key=lambda k: criteria_list[k], reverse=True)
            s = np.array_split(sorted_criteria_list, bases_num)
            cluster_graphs = [list(item) for item in s]
        cluster_graph_dic[split] = cluster_graphs
    return cluster_graph_dic
# graphon = simulator.synthesize_graphon(r=args.r, type_idx=i)
# simulator.visualize_graphon(graphon, save_path=os.path.join(args.f_result, 'graphon_{}.pdf'.format(i)))


def estimate_graphon(clusters,method,args,subgraphs,split):
    graphons = []
    for i in range(len(clusters)):
        estimation,_ = learner.estimate_graphon(subgraphs[clusters[i]], method=method, args=args)
        graphons.append(estimation)
        np.save(os.path.join(args.f_result, "ego2_graphons_{}_{}".format(split,i)),estimation)

    # graphon_entropy  = sum([entropy(graphons[i].flatten()) for i in range(len(clusters))])
    # corre = pearsonr
    # pre_graphon = np.mean(graphons,axis = 0)


print(args.ego_hops)
args.resume_graphs =False
edge_indices,num_nodes_ls = [],[]
if args.resume_graphs:
    load_path = os.path.join("/data/caoyuxuan/gcc/sampled_ego2/", args.data_type+"_"+args.down_data+"_"+args.pre_data+".npy")
    subgraphs = np.load(load_path, allow_pickle=True)
    for i in range(len(subgraphs)):
        num_nodes = subgraphs[i].shape[0]
        edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.tensor(subgraphs[i]))
        edge_indices.append(edge_index)
        num_nodes_ls.append(num_nodes)
else:
    subgraphs, graph_sizes, edge_sizes=[],[],[]
    since = time.time()
    save_file = os.path.join(args.save_dir, args.data_type+"_"+args.down_data+"_"+args.pre_data)
    print(save_file)
    prob = None
    if args.data_type == "down_graph":
        graphs = load_down_graph_data(args.down_data)

    else:
        if args.data_type == "down_node":
            edge_index, num_nodes = load_down_node_data(args.down_data)
            candidates = [i for i in range(num_nodes)]
            threshold = len(candidates)
        elif args.data_type == "pre_single":
            edge_index,num_nodes = load_pretrain_data(args.pre_data,args.data_type)
            candidates = np.random.choice(num_nodes, size=min(num_nodes, 20000), replace=False, p=prob)
        else:
            print(args.pre_data)
            print(args.load_dir)
            data,size_graphs,prob = load_pretrain_data(args)
            num_graphs = len(data)
            size_graphs = [int(i) for i in size_graphs]
            num_nodes = sum(size_graphs)
            idx_list = [i for i in range(num_nodes)]
            edge_indexs = [torch.stack(data[i].edges()) for i in range(num_graphs)]
            graph_sample_cnt = [0 for i in range(num_graphs)]
            single_graphs = [[] for i in range(num_graphs)]
            # domains = [[] for i in range(args.domain_num)]
            prob = prob.numpy()
            candidates = np.random.choice(num_nodes, size=min(num_nodes, 15000), replace=False, p=prob)
            threshold = min(len(candidates),1000)
        idx,t= 0,0
        while idx < threshold and t < len(candidates):
            n_id =candidates[t]
            t += 1
            if idx%100 == 0:
                print("iter_{}".format(idx))
            if args.data_type == "pre_merge":
                for i in range(num_graphs):
                    if n_id < size_graphs[i]:
                        edge_index = edge_indexs[i]
                        break
                    else:
                        n_id -= size_graphs[i]
                graph_idx = i
            new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id), num_hops=args.ego_hops, edge_index=edge_index,relabel_nodes=True)
            if len(new_nodes) > args.node_threshold or len(new_edge_index) > args.edge_threshold:
                continue
            if args.data_type == "pre_merge":
                single_graphs[graph_idx].append(idx)
                graph_sample_cnt[graph_idx] += 1
            idx += 1
            edge_indices.append(new_edge_index)
            num_nodes_ls.append(len(new_nodes))
            adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
            subgraphs.append(adj)
            # single_graphs[i].append(adj)
            graph_sizes.append(len(new_nodes))
            edge_sizes.append(len(new_edge_index[0]))
        end = time.time()
        sample_time = end - since
        subgraphs=np.array(subgraphs)
        if args.data_type =="pre_merge":
            print(np.array(graph_sample_cnt) / min(num_nodes, 10))
        avg_nodes = np.mean(graph_sizes)
        avg_edges = np.mean(edge_sizes)
        print("sample_time{};#graphs{};#average nodes{};#average edges{}".format(sample_time,len(graph_sizes),avg_nodes, avg_edges))
        np.save(save_file,subgraphs)
        if args.data_type =="pre_merge":
            for k in range(len(single_graphs)):
                save_file_domain =  save_file + "domain_"+str(k)
                np.save(save_file_domain,single_graphs[k])
# subgraphs = np.load(save_file, allow_pickle=True)
# edge_indices = []
# num_nodes_ls = []
# for i in range(subgraphs.size):
#     num_nodes = subgraphs[i].shape[0]
#     edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.tensor(subgraphs[i]))
#     edge_indices.append(edge_index)
#     num_nodes_ls.append(num_nodes)

for m in range(len(methods)):
    if args.resume_graphs == False:
        if args.data_type =="pre_merge":
            estimate_graphon(single_graphs,methods[m],args,subgraphs,"domain")
            # np.save(os.path.join(args.f_result, "ego2_split_domain_graph_estimation_{}".format( methods[m])), single_graph_graphon)
            all_graphon,_ =  learner.estimate_graphon(subgraphs, method=methods[m], args=args)
            np.save(os.path.join(args.f_result, "ego2_split_all_graph_estimation_{}".format(methods[m])),
                    all_graphon)
        else:
            all_graphon,_ = learner.estimate_graphon(subgraphs, method=methods[m], args=args)
            np.save(os.path.join(args.f_result, "ego2_all_graph_estimation_{}".format(methods[m])),all_graphon)
    if args.data_type == "pre_merge":
        split_criterions = ["Kmeans","syn"]
        clusters_dic = split_subgraphs(num_nodes_ls,split_criterions,edge_indices, subgraphs)
        for j in range(len(split_criterions)):
            clusters = clusters_dic[split_criterions[j]]
            estimate_graphon(clusters,methods[m],args,subgraphs,split_criterions[j])
            # np.save(os.path.join(args.f_result, "ego2_split_{}_estimation_{}".format(split_criterions[j],methods[m])), pre_graphon)
            # print("{}_graphon_entropy:{}".format(split_criterions[j],graphon_entropy))
    # for m in range(len(methods)):
    #         since = time.time()
    #         _, estimation = learner.estimate_graphon(graphs, method=methods[m], args=args)
    #         simulator.visualize_graphon(estimation,
    #                                     title=methods[m],
    #                                     save_path=os.path.join(args.f_result,
    #                                                            '{}_ego_estimation_{}_{}.pdf'.format(args.pre_data, n, methods[m])))
    #         end=time.time()
    #         print("estimate time:{}".format(end-since))
    #         np.save(os.path.join(args.f_result,"estimation_{}".format(n)),estimation)
# with open(os.path.join(args.f_result, 'results_synthetic.pkl'), 'wb') as f:
#     pickle.dump(errors, f)
