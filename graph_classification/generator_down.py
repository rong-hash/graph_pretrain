import argparse
from utils.load_utils import MoleculeDataset
import torch_geometric
from utils.learner import estimate_graphon
from tqdm import tqdm
import numpy as np
import os
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
parser.add_argument('--down_data', type=str, default="bbbp",
                    help='downstream data path')
args = parser.parse_args()
method = 'LG'
file_path ="data/dataset/"
save_path ="data//graphons/"
def estimate_down_graphon(file_path,dataname):
    dataset = MoleculeDataset(file_path + dataname, dataset=dataname)
    datas=dataset
    predata_splits = ""
    for i in range(args.split_num):
        predata_splits+=args.split_ids[i]
    graphons,stepfuncs = [],[]
    for j in tqdm(range(len(datas))):
        matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,num_nodes=datas[j].num_nodes).toarray()
        step_func, non_para_graphon = estimate_graphon([matrix_graph] , method=method, args=args)
        graphons.append(non_para_graphon)
        stepfuncs.append(step_func)
    np.save(save_path + "down/" + dataname + "/graphons.npy", graphons)
    np.save(save_path + "down/" + dataname + "/func.npy", stepfuncs)
estimate_down_graphon(file_path,args.down_data)

