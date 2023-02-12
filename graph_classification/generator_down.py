from utils.load_utils import MoleculeDataset
import torch_geometric
from utils.learner import estimate_graphon
from tqdm import tqdm
import numpy as np
def estimate_generator_down(args):
    file_path = args.file_path
    dataname = args.down_data
    save_path = args.save_path
    method = args.method
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

