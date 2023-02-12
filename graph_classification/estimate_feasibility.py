import torch.nn as nn
import torch
import argparse
import os
from other_gw import entropic_gw
import numpy as np
import cv2
import pandas as pd
import torch.nn.functional as F
import simulator
from tqdm import tqdm
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--epoch', type=int,default=100,help='number of trails to fit the final graphon')
parser.add_argument('--learning_rate', type=float,default=0.05,help='learning_rate')
parser.add_argument('--weight_decay', type=float,default=1e-5,help='weight_decay')
parser.add_argument('--beta1', type=float,default=0.9,help='learning_rate')
parser.add_argument('--beta2', type=float,default=0.99,help='weight_decay')
parser.add_argument('--gpu', type=int,default=0,help='gpu')
parser.add_argument('--func', type=int, default="0")
args = parser.parse_args()
assert args.gpu is not None and torch.cuda.is_available()
print("Use GPU: {} for training".format(args.gpu))
# pre_splits= ["012","023","013","014","123","134","124","234","124","024"]
pre_datasets=["0","1","2","3","4 "]
down_datasets=["bbbp",  "bace","muv","hiv","clintox"]
splits = ["trivial","topo","domain"]
pre_datas=[]
for i in range(len(pre_datasets)):
    for j in range(i+1,len(pre_datasets)):
        pre_data = pre_datasets[i]+"_"+pre_datasets[j]
        pre_datas.append(pre_data)
def mean_fit(pre_graphons_norm):
    return np.mean(pre_graphons_norm,axis = 0)
def alpha_fit(pre_graphons_norm,cluster_num, down_graphon,save_fig_file):
    alpha_graph = nn.Linear(1, cluster_num).to(torch.device(args.gpu))
    torch.nn.init.constant_(alpha_graph.weight, 1)
    optimizer_alpha = torch.optim.Adam(
        alpha_graph.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    loss_2,loss_3=[],[]

    min_loss = 100
    trigger_time = 0
    patience = 3
    for i in range(args.epoch):
        if i % 20 == 0:
            trigger_time = 0
        # print("epoch", i)
        final_graphon = 0
        normalized_alpha = F.softmax(alpha_graph.weight, dim=0).to(torch.device(args.gpu))
        for j in range(cluster_num):
            final_graphon += normalized_alpha[j].to(torch.device(args.gpu)) * torch.tensor(pre_graphons_norm[j]).to(
                torch.device(args.gpu))
        optimizer_alpha.zero_grad()
        loss2 = entropic_gw(final_graphon, torch.tensor(down_graphon).to(torch.device(args.gpu)), device=args.gpu)
        loss3 = simulator.gw_distance(final_graphon.cpu().detach().numpy(),down_graphon )
        loss_2.append(loss2.data.item())
        loss_3.append(loss3)
        loss2.backward()
        if min_loss > loss2.data.item():
            min_loss = min(min_loss, loss2.data.item())
            min_graphon = final_graphon
        if i > 30:
            if loss2 > min_loss:
                trigger_time += 1
                if trigger_time >= patience:
                    break
        optimizer_alpha.step()
    if min_graphon == None:
        gw_dis = simulator.gw_distance(final_graphon.cpu().detach().numpy(), down_graphon)
    else:
        gw_dis = simulator.gw_distance(min_graphon.cpu().detach().numpy(), down_graphon)
    print("final alpha:", normalized_alpha)
    return min_loss ,gw_dis


def load_graphons(split,pre_splits,down_graphon_path):
    pre_graphons = []
    if split == "topo":
        cluster_num = 5
    elif split == "domain":
        cluster_num = 2
    elif split == "trivial":
        cluster_num = 1

    for i in range(cluster_num):
        if args.func:
            if split == "trivial":
                load_path = "data/graphons/" + split + "/zinc_standard_agent" + pre_splits + "/func.npy"
            elif split == "domain":
                load_path = "data/graphons/" + split + "/zinc_standard_agent" + pre_splits[i] + "/func.npy"
            else:
                load_path = "data/graphons/" + split + "/zinc_standard_agent" + pre_splits + + "/func"+str(
                    i) + ".npy"
            graphon = np.load(load_path)
            cur_size = graphon.shape[0]
            max_size = max(cur_size, max_size)
            print(max_size)
        else:
            if split == "trivial":
                load_path = "data/graphons/" + split + "/zinc_standard_agent" + pre_splits + "/graphon.npy"
            elif split=="domain":
                load_path = "data/graphons/" + split + "/zinc_standard_agent"  + pre_splits[i]  + "/graphon.npy"
            else:
                load_path = "data/graphons/" + split + "/zinc_standard_agent" + pre_splits + "/graphon"+ str(
                    i) + ".npy"
            graphon = np.load(load_path)
            max_size = graphon.shape[0]
        pre_graphons.append(graphon)
    down_graphon = np.load(down_graphon_path)
    max_size = max(down_graphon.shape[0], max_size)
    pre_graphons_norm = []
    for i in range(cluster_num):
        pre_graphon = cv2.resize(pre_graphons[i], dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
        pre_graphons_norm.append(pre_graphon)
    down_graphon = cv2.resize(down_graphon, dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
    return pre_graphons_norm, cluster_num, down_graphon
save_path="save_diff"
for k in range(len(splits)):
    split = splits[k]
    mean_dis_dic,alpha_dis_dic,mean_dis_gw,alpha_dis_gw ={},{},{},{}
    if not os.path.exists(save_path + splits[k]):
        os.makedirs(save_path + splits[k])
    for j in tqdm(range(len(down_datasets))):
        mean_dis_dic[down_datasets[j]], alpha_dis_dic[down_datasets[j]], mean_dis_gw[down_datasets[j]], alpha_dis_gw[
            down_datasets[j]] = [], [], [], []
        down_data = down_datasets[j]
        for i in tqdm(range(len(pre_datas))):
            pre_split =pre_datas[i]
            down_graphon_path = "data/graphons/trivial/" + down_data +"/graphon.npy"
            pre_graphons, cluster_num, down_graphon = load_graphons(split,pre_split,down_graphon_path)
            mean_graphon = mean_fit(pre_graphons)
            mean_dis =simulator.gw_distance(mean_graphon , down_graphon)
            mean_dis1 = entropic_gw(torch.tensor(mean_graphon).to(torch.device(args.gpu)), torch.tensor(down_graphon).to(torch.device(args.gpu)),device=args.gpu)
            save_filg_path = "save_diff/" + pre_split + down_data + split
            alpha_dis1,alpha_dis2 =alpha_fit(pre_graphons,cluster_num,down_graphon,save_filg_path)
            mean_dis_gw[down_datasets[j]].append(mean_dis)
            mean_dis_dic[down_datasets[j]].append(mean_dis1.data.item())
            alpha_dis_dic[down_datasets[j]].append(alpha_dis1)
            alpha_dis_gw[down_datasets[j]].append(alpha_dis2)
    mean_df = pd.DataFrame(mean_dis_dic, index=pre_datas)
    alpha_df = pd.DataFrame(alpha_dis_dic, index=pre_datas)
    mean_df_gw = pd.DataFrame(mean_dis_gw, index=pre_datas)
    alpha_df_gw = pd.DataFrame(alpha_dis_gw, index=pre_datas)
    mean_df.to_csv(save_path + splits[k]+"/mean_diff.csv", sep=',')
    alpha_df.to_csv(save_path + splits[k]+"/alpha_diff.csv", sep=',')
    mean_df_gw.to_csv(save_path + splits[k]+"/mean_diff_gw.csv", sep=',')
    alpha_df_gw.to_csv(save_path + splits[k]+"/alpha_diff_gw.csv", sep=',')
    print(mean_df)
    print(alpha_df)
    print(mean_df_gw)
    print(alpha_df_gw)

