import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import expm
import torch.nn.functional as F

from utils import MixedDropout, sparse_matrix_to_torch, matrix_calculation

def full_attention_conv(qs, ks):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    # normalize input
    qs = qs / torch.norm(qs, p=2) # [N, H, M]
    ks = ks / torch.norm(ks, p=2) # [L, H, M]
    N = qs.shape[0]

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N

    # compute attention for visualization if needed
    attention = torch.einsum("nhm,lhm->nhl", qs, ks) / attention_normalizer # [N, L, H]

    return attention

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    # D_vec_invsqrt_corr = 1 / D_vec
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr
    # return D_invsqrt_corr @ A

def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())

class VisualizationData:
    def __init__(self, G_w, G_b, label):
        self.G_w = G_w
        self.G_b = G_b
        self.X_1 = None
        self.X_2 = None
        self.label = label

    def update_X_1(self, X_1):
        self.X_1 = X_1

    def update_X_2(self, X_2):
        self.X_2 = X_2

    def update_G_w(self, G_w):
        self.G_w = G_w

    def update_G_b(self, G_b):
        self.G_b = G_b

    def get_X_1(self):
        return self.X_1

    def get_X_2(self):
        return self.X_2

    def get_G_w(self):
        return self.G_w

    def get_G_b(self):
        return self.G_b

    def get_label(self):
        return self.label

class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions

class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, label: torch.FloatTensor, train_idx: np.array, num_heads:int, niter: int, npow: int,
                 niter_L: int, T_VPoisson:int = 10, lambda_VPoisson:float = 1.0, alpha:float=1.0, beta:float=1.0, w_k:int=3, b_k:int=3, num_nodes: int = None, num_hidden: int = None, drop_prob: float = None, device=None):
        '''
        Parameters
        ----------
        adj_matrix : 原始的图
        niter : 阶数
        npow : 跳
        drop_prob : dropout
        '''
        super().__init__()

        self.niter = niter
        self.niter_L = niter_L
        self.npow = npow
        self.device = device
        self.label = label
        self.train_idx = train_idx
        self.num_heads = num_heads
        self.T_VPoisson = T_VPoisson
        self.lambda_VPoisson = lambda_VPoisson
        self.alpha = alpha
        self.beta = beta
        # self.k = k
        self.w_k = w_k
        self.b_k = b_k
        if num_hidden is not None:
            self.num_hidden = num_hidden
        if num_nodes is not None:
            self.num_nodes = num_nodes

        self.linear1 = nn.Linear(self.niter+1, 1)
        self.linear2 = nn.Linear(self.niter_L, 1)
        self.softmax = torch.nn.Softmax(dim=0)
        # self.W, self.B = self.BW_generator(label, self.A, train_idx)
        # csr_matrix类型的adj_matrix转化成tensor的稀疏张量
        M = calc_A_hat(adj_matrix)
        self.A = M.todense()
        self.A = torch.from_numpy(self.A).float().to(device)
        self.W, self.B = self.V_Poisson(label, self.A, train_idx) # 得到W, B
        # 计算normalized的W和B，即(D_w+I)^{-1/2} @ (W+I) @ (D_w+I)^{-1/2}和(D_b+I)^{-1/2} @ (B+I) @ (D_b+I)^{-1/2}
        D_w = torch.diag(torch.sum(self.W, dim=1))
        D_b = torch.diag(torch.sum(self.B, dim=1))
        self.W = torch.inverse(D_w + torch.eye(D_w.shape[0]).to(device)) @ self.W @ torch.inverse(D_w + torch.eye(D_w.shape[0]).to(device))
        self.B = torch.inverse(D_b + torch.eye(D_b.shape[0]).to(device)) @ self.B @ torch.inverse(D_b + torch.eye(D_b.shape[0]).to(device))

        self.display_infor = VisualizationData(self.W, self.B, label)

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

        def reset_parameters(self):
            pass

    def BW_generator(self, Y, A, train_idx):
        '''
        Y: the label matrix
        A: the graph Laplacian
        train_n: the number of training samples
        '''
        # 整理Y，将train_idx的放到Y_l中，剩余的放到Y_u中，并记下对应的index
        L = torch.eye(A.shape[0]).to(A.device) - A
        Y_l = Y[train_idx].to(L.device)
        # 同样整理L调换其行和列，新的laplacian记为L_moved，使得其内部是L_ll, L_lu, L_ul, L_uu
        alpha = self.linear2.weight.t().squeeze(1)
        # 生成L_moved, 公式是 L_moved = alpha[0] * L + alpha[1] * L^2 + ..., 使用循环实现
        L_moved = 1/self.niter_L * L
        for i in range(self.niter_L - 1, 0, -1):
            L_moved =  1/self.niter_L * L + L_moved

        L_ll = L_moved[train_idx][:, train_idx]
        L_lu = L_moved[train_idx][:, [i for i in range(L_moved.shape[0]) if i not in train_idx]]
        L_ul = L_moved[[i for i in range(L_moved.shape[0]) if i not in train_idx]][:, train_idx]
        L_uu = L_moved[[i for i in range(L_moved.shape[0]) if i not in train_idx]][:, [i for i in range(L_moved.shape[0]) if i not in train_idx]]
        Y_u = 1/2 * torch.linalg.pinv(L_uu) @ (- L_lu.T - L_ul) @ Y_l
        # 用Y_u替换掉Y中的无标签样本, 但是要用Y_hat，要避免覆盖原来的Y
        Y_hat = Y.clone().to(L.device)
        Y_hat[[i for i in range(L_moved.shape[0]) if i not in train_idx]] = Y_u
        # Y_hat的每一行softmax一下
        Y_hat = self.softmax(Y_hat)
        # 计算W = Y_hat @ Y_hat^T, B = 全1矩阵-W
        W = torch.sigmoid(Y_hat @ Y_hat.T)
        # W中非零值设置为1，0则保持不变
        B = torch.ones_like(W) - W
        W = W * L
        B = B * L
        return W, B

    def V_Poisson(self, Y, A, train_idx):
        tmp_Y = Y.clone()
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        n, c = tmp_Y.shape[0], tmp_Y.shape[1]
        ones_vec = torch.ones(n, 1).to(self.device)
        Y_l = tmp_Y[train_idx].to(self.device)
        y_hat = 1 / Y_l.shape[0] * torch.sum(Y_l, dim=0)
        B = torch.zeros(n, c).to(self.device)
        B[train_idx] = Y_l - y_hat
        U = torch.zeros(n, c).to(self.device)
        for i in range(self.T_VPoisson):
            U_hat = D @ (U - ones_vec @ ones_vec.T @ D @ U / torch.sum(D))
            U = U + torch.linalg.inv(D) @ (B - L @ U + self.lambda_VPoisson * U_hat)
        U[train_idx] = Y_l

        W_mat = torch.sigmoid(U @ U.T)
        B_mat = torch.ones_like(W_mat) - W_mat
        W_mat = W_mat * self.A
        B_mat = B_mat * self.A
        return W_mat, B_mat

    def update_W_B(self, d_w, d_b, uptype='Dense'):
        if uptype == 'Dense':
            self.W = torch.exp(-d_w / self.alpha) / torch.sum(torch.exp(-d_w/self.alpha), dim=1).unsqueeze(1)
            self.B = torch.exp(d_b / self.beta) / torch.sum(torch.exp(d_b/self.beta), dim=1).unsqueeze(1)
        else:
            W = torch.zeros_like(self.W)
            B = torch.zeros_like(self.B)
            # 找到d_w每行中，最小的k+1个值的索引和值
            values_w, indices_w = torch.topk(d_w, self.w_k + 1, dim=1, largest=False)

            # 每行的这些索引中，对应的W设置为（每行第k+1小的值 - 每行这个索引的对应的元素）/ (k * 每行第k+1小的值 - 每行最小的k个值的和)
            min_k_plus_one_value_w = values_w[:, -1].unsqueeze(1)
            min_k_values_sum_w = values_w[:, :-1].sum(dim=1, keepdim=True)
            W.scatter_(1, indices_w[:, :-1], (min_k_plus_one_value_w - d_w.gather(1, indices_w[:, :-1])) / (
                        self.w_k * min_k_plus_one_value_w - min_k_values_sum_w))

            # 找到d_b每行中，最大的k+1个值的索引和值
            values_b, indices_b = torch.topk(d_b, self.b_k + 1, dim=1, largest=True)

            # 每行的这些索引中，对应的B设置为（每行这个索引的对应的元素 - 每行第k+1大的值）/ (k * 每行最大的k个值的和 - 每行第k+1大的值)
            max_k_plus_one_value_b = values_b[:, -1].unsqueeze(1)
            max_k_values_sum_b = values_b[:, :-1].sum(dim=1, keepdim=True)
            B.scatter_(1, indices_b[:, :-1], (d_b.gather(1, indices_b[:, :-1]) - max_k_plus_one_value_b) / (
                        self.b_k * max_k_values_sum_b - max_k_plus_one_value_b))

            self.W = W
            self.B = B

        return None

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor, origin_fea: torch.sparse.FloatTensor=None):
        preds = local_preds.float()
        preds_tmp = local_preds.float()
        W = self.W
        B = self.B
        self.display_infor.update_G_w(W)
        self.display_infor.update_G_b(B)
        I_B = torch.eye(B.shape[0]).to(B.device) - B

        tau = self.linear1.weight.t().squeeze()
        # tau = torch.exp(tau) / torch.sum(torch.exp(tau))

        H = I_B @ preds
        self.display_infor.update_X_1(H)
        tmp = tau[-1] * H
        for i in range(self.niter - 1, -1, -1):
            tmp = tau[i] * H + W @ tmp

        preds = tmp
        self.display_infor.update_X_2(preds)

        # d_w = torch.norm(preds.unsqueeze(0) - preds.unsqueeze(1), dim=2)
        # d_b = torch.norm(preds.unsqueeze(0) - preds_tmp.unsqueeze(1), dim=2)
        d_w = matrix_calculation(preds, preds)
        d_b = matrix_calculation(preds, preds_tmp)
        d_b = d_b / torch.max(d_b)

        self.update_W_B(d_w, d_b, uptype='Sparse')

        return preds[idx]


