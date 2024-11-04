import os
import pandas as pd
import numpy as np
import argparse

import torch

from utils import *
from model import GNNs
from training import train_model
from earlystopping import stopping_args
from propagation import *
from load_data import *
from tqdm import tqdm
from scipy.sparse import triu

from training import normalize_attributes
from utils import matrix_to_torch

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
import skopt

def cal_edge(adj_matrix, lebiao):
    total_sum = adj_matrix.sum()
    diagonal_sum = adj_matrix.diagonal().sum()
    num_edge = (total_sum - diagonal_sum) / 2

    # 转置矩阵
    A_transpose = adj_matrix.transpose()

    # 相加并除以2
    A_undirected = (adj_matrix + A_transpose) / 2

    # 初始化同构边的数量
    homophily_edges = 0

    # 获取上三角矩阵，不包括对角线
    A_triu = triu(A_undirected, k=1)

    # 遍历每条边
    for i, j in zip(*A_triu.nonzero()):
        # 如果边连接的两个节点属于同一类别，同构边+1
        if lebiao[i] == lebiao[j]:
            homophily_edges += 1
    return (total_sum - diagonal_sum) / 2, homophily_edges / num_edge

search_space = list()
search_space.append(Real(1e-7, 5e-2, name='reg_lambda'))
search_space.append(Real(1e-7, 5e-2, name='reg_w'))
search_space.append(Real(1e-7, 5e-2, name='reg_b'))
search_space.append(Real(0.001, 0.05, name='lr'))
search_space.append(Real(0.35, 0.95, name='dropout'))
# search_space.append(Integer(0, 4, name='npow'))
# search_space.append(Integer(0, 4, name='npow_attn'))
# search_space.append(Integer(1, 15, name='niter'))
# search_space.append(Integer(1, 15, name='niter_attn'))
search_space.append(Integer(3, 20, name='T_VPoisson'))
search_space.append(Real(1e-2, 1e2, name='lambda_VPoisson'))
search_space.append(Real(1e-1, 1e1, name='alpha'))
search_space.append(Real(1e-1, 1e1, name='beta'))

@use_named_args(search_space)
def evaluate_model(**params):
    args.reg_lambda = params['reg_lambda']
    args.reg_w = params['reg_w']
    args.reg_b = params['reg_b']
    args.lr = params['lr']
    args.dropout = params['dropout']
    # args.npow = params['npow']
    # args.npow_attn = params['npow_attn']
    # args.niter = params['niter']
    # args.niter_attn = params['niter_attn']
    args.T_VPoisson = params['T_VPoisson']
    args.lambda_VPoisson = params['lambda_VPoisson']
    args.alpha = params['alpha']
    args.beta = params['beta']

    print(args)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.dataset == 'acm':
        graph, idx_np = load_new_data_acm(args.labelrate)
    elif args.dataset == 'wiki':
        # graph, idx_np = load_new_data_wiki(args.labelrate)
        # # args.lr = 0.03
        # # args.reg_lambda = 5e-4
        graph, idx_np = load_fixed_wikics()
    elif args.dataset == 'ms':
        graph, idx_np = load_new_data_ms(args.labelrate)
    elif args.dataset in ['chameleon', 'squirrel', 'cornell', 'texas', 'wisconsin', 'film']:
        graph, idx_np = load_new_data(args.dataset, args.train_labelrate, args.val_labelrate, args.test_labelrate,
                                      args.random_seed)
    elif args.dataset in ['computers', 'photo']:
        graph, idx_np = load_Amazon(args.dataset)
    elif args.dataset in ['ar', 'feret', 'humbi']:
        graph, idx_np = load_Lai_data(args.dataset, args.basemethod)
    elif args.dataset in ['deepface']:
        graph, idx_np = load_deepface_data('ar', base_path='./data/Lai/ar_proj_data_adj_vgg.mat')
    else:
        if args.dataset == 'cora':
            feature_dim = 1433
        elif args.dataset == 'citeseer':
            feature_dim = 3703
        elif args.dataset == 'pubmed':
            feature_dim = 500
        graph, idx_np = load_new_data_tkipf(args.dataset, feature_dim, args.labelrate)

    if args.dataset in ['chameleon', 'squirrel', 'cornell', 'texas', 'wisconsin', 'film', 'wiki', 'computers', 'photo', 'ar', 'feret', 'humbi', 'deepface']:
        fea_tensor = graph.attr_matrix
    else:
        fea_tensor = graph.attr_matrix.todense()
    fea_tensor = torch.from_numpy(fea_tensor).float().to(device)
    # graph.labels是ndarray类型数据，维度是n \times 1, 生成onehot编码存到torch中用Floattensor的形式
    init_labels = torch.from_numpy(graph.labels).long().squeeze()
    init_labels = F.one_hot(init_labels).float()

    # num_edges, homophily = cal_edge(graph.adj_matrix, graph.labels)
    # print('{}, {}'.format(num_edges, homophily))

    print_interval = 50
    test = True

    propagation = []
    results = []

    i_tot = 0
    # average_time: 每次实验跑average_time次取平均
    average_time = args.runs
    for _ in tqdm(range(average_time)):
        i_tot += 1
        propagation = PPRPowerIteration(graph.adj_matrix, init_labels, idx_np['train'], num_heads=args.num_heads,
                                        niter=args.niter, npow=args.npow, niter_L=1, T_VPoisson=args.T_VPoisson,
                                        lambda_VPoisson=args.lambda_VPoisson, alpha=args.alpha, beta=args.beta, w_k=args.w_k, b_k=args.b_k,
                                        num_nodes=fea_tensor.shape[0], num_hidden=64, device=device)

        model_args = {
            'hiddenunits': [64],
            'drop_prob': args.dropout,
            'propagation': propagation}

        logging_string = f"Iteration {i_tot} of {average_time}"

        _, result = train_model(idx_np, args.dataset, GNNs, graph, model_args, args.lr, args.reg_lambda, args.reg_w, args.reg_b, stopping_args,
                                test, device, None, print_interval)
        results.append({})
        results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
        results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
        results[-1]['runtime'] = result['runtime']
        results[-1]['runtime_perepoch'] = result['runtime_perepoch']
        tmp = propagation.linear1.weight.t().unsqueeze(1).squeeze()

    result_df = pd.DataFrame(results)
    result_df.head()

    stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
    valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
    runtime = calc_uncertainty(result_df['runtime'])
    runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])

    print(
        "Early stopping: Accuracy: {:.2f} ± {:.2f}%\n"
        "{}: ACC: {:.2f} ± {:.2f}%\n"
        "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms\n"
          .format(
            stopping_acc['mean'] * 100,
            stopping_acc['uncertainty'] * 100,
            'Test' if test else 'Validation',
            valtest_acc['mean'] * 100,
            valtest_acc['uncertainty'] * 100,
            runtime['mean'],
            runtime['uncertainty'],
            runtime_perepoch['mean'] * 1e3,
            runtime_perepoch['uncertainty'] * 1e3,
        ))

    return 1.0 - valtest_acc['mean']

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default='humbi')
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=60)
    parse.add_argument("-t", "--type", help="model for training, (PPNP=0, GNN-LF=1, GNN-HF=2)", type=int, default=0)
    parse.add_argument("--train_labelrate", help="labeled rate of training set", type=float, default=0.48)
    parse.add_argument("--val_labelrate", help="labeled data of validation set", type=float, default=0.32)
    parse.add_argument("--test_labelrate", help="labeled data of testing set", type=float, default=0.2)
    parse.add_argument("--seed", type=int, default=123, help="random seed")
    parse.add_argument("--random_seed", help="random seed", type=bool, default=False)
    parse.add_argument("-f", "--form", help="closed/iter form models (closed=0, iterative=1)", type=int, default=1)
    parse.add_argument('--cpu', action='store_true')
    parse.add_argument("--device", help="GPU device", type=str, default="1")
    parse.add_argument("--niter", help="times for iteration", type=int, default=1)
    parse.add_argument("--niter_attn", help="times for iteration", type=int, default=12)
    parse.add_argument("--num_heads", help="multi heads for attention", type=int, default=1)
    parse.add_argument("--reg_lambda", help="regularization", type=float, default=1.0741571327014965e-05)
    parse.add_argument("--reg_w", help="regularization", type=float, default=1e-07)
    parse.add_argument("--reg_b", help="regularization", type=float, default=0.012033910901829412)
    parse.add_argument("--lr", help="learning rate", type=float, default=0.05)
    parse.add_argument("--dropout", help="learning rate", type=float, default=0.37237810037971025)
    parse.add_argument("--runs", help="learning rate", type=int, default=1)
    parse.add_argument('--npow', type=int, default=4, help="for APGNN gap")
    parse.add_argument('--npow_attn', type=int, default=0, help="for APGNN gap with attention")

    parse.add_argument('--T_VPoisson', type=int, default=5, help="for V-Poisson iteration")
    parse.add_argument('--lambda_VPoisson', type=float, default=1.0, help="for V-Poisson regularization")
    parse.add_argument('--alpha', type=float, default=1.0, help="for W graph")
    parse.add_argument('--beta', type=float, default=1.0, help="for B graph")
    # parse.add_argument('--k', type=int, default=5, help="for k-nonzero graph")
    parse.add_argument('--w_k', type=int, default=5, help="for k-nonzero graph")
    parse.add_argument('--b_k', type=int, default=5, help="for k-nonzero graph")

    parse.add_argument("--basemethod", help="[pca, lda, lpp]", type=str, default=' ')

    args = parse.parse_args()

    result = skopt.gp_minimize(evaluate_model, search_space, verbose=True, n_calls=32)

    print('Best Accuracy: %.3f' % (1.0 - result.fun))
    print('Best Parameters: %s' % (result.x))