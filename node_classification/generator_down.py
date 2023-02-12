from utils.data_util import create_node_classification_dataset, _create_dgl_graph
import torch_geometric
from utils.learner import estimate_graphon
import argparse
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
parser.add_argument('--down_data', type=str, default="",
                    help='downstream data path')
args = parser.parse_args()
method = 'LG'
file_path ="data/dataset/"
save_path ="data/graphons/"
load_path ="data/graphons/"
def load_down_data(name):
    data =create_node_classification_dataset(name).data
    num_nodes = len(data.y)
    edge_index = data.edge_index
    down_dgl = _create_dgl_graph(edge_index)
    return edge_index,num_nodes,down_dgl

edge_index, num_nodes = load_down_data(args.down_data)
candidates = [i for i in range(num_nodes)]
subgraphs = []
for t in range(len(candidates)):
    n_id = candidates[t]
    new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id),
                                                                                   num_hops=args.ego_hops,
                                                                                   edge_index=edge_index,
                                                                                   relabel_nodes=True)
    adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
    subgraphs.append(adj)
step_func, non_para_graphon = estimate_graphon(np.array(subgraphs), method=method, args=args)
np.save(save_path + "down/" + args.down_data + "/graphon.npy", non_para_graphon)
np.save(save_path + "down/" + args.down_data + "/func.npy", step_func)
