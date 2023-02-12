from utils.data_util import create_node_classification_dataset, _create_dgl_graph
import torch_geometric
from utils.learner import estimate_graphon
import numpy as np
def load_down_data(name):
    data =create_node_classification_dataset(name).data
    num_nodes = len(data.y)
    edge_index = data.edge_index
    down_dgl = _create_dgl_graph(edge_index)
    return edge_index,num_nodes,down_dgl
def estimate_generator_down(args):
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
    step_func, non_para_graphon = estimate_graphon(np.array(subgraphs), method=args.method, args=args)
    np.save(args.save_path + "down/" + args.down_data + "/graphon.npy", non_para_graphon)
    np.save(args.save_path + "down/" + args.down_data + "/func.npy", step_func)

