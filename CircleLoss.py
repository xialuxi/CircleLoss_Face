# -*- coding: UTF-8 -*-
from typing import Tuple
import torch
from torch import nn, Tensor

class SparseCircleLoss(nn.Module):
    def __init__(self, m: float, batch_size: int, emdsize: int ,class_num: int, gamma: float) -> None:
        super(SparseCircleLoss, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        self.batch_size = batch_size
        self.class_num = class_num
        self.emdsize = emdsize

        self.weight = nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.weight)
        self.relu = nn.ReLU()
        self.use_cuda = False


    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        similarity_matrix = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        if self.use_cuda:
            one_hot = torch.zeros(similarity_matrix.size(), device='cuda')
        else:
            one_hot = torch.zeros(similarity_matrix.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        one_hot = one_hot.type(dtype=torch.bool)
        #sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sp = similarity_matrix[one_hot]

        mask = one_hot.logical_not()
        sn = similarity_matrix[mask]

        sp = sp.view(self.batch_size, -1)
        sn = sn.view(self.batch_size, -1)

        alpha_p = self.relu(self.O_p - sp)
        alpha_n = self.relu(sn - self.O_n)

        r_sp_m = alpha_p * (sp - self.Delta_p)
        r_sn_m = alpha_n * (sn - self.Delta_n)

        _Z = torch.cat((r_sn_m, r_sp_m), 1)
        _Z = _Z * self.gamma

        logZ = torch.logsumexp(_Z, dim=1, keepdims=True)

        loss =  -r_sp_m * self.gamma + logZ

        return loss.mean()
        
        
        
if __name__ == "__main__":

    feat = torch.rand(64, 128, requires_grad=True)
    label = torch.randint(high=9, size=(64,))

    SparseCircle = SparseCircleLoss(m=0.25, batch_size=64, emdsize=128, class_num=10, gamma=64)
