# -*- coding: UTF-8 -*-
from typing import Tuple
import torch
from torch import nn, Tensor

class SparseCircleLoss(nn.Module):
    def __init__(self, m: float, emdsize: int ,class_num: int, gamma: float) -> None:
        super(SparseCircleLoss, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        self.class_num = class_num
        self.emdsize = emdsize

        self.weight = nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.weight)
        self.use_cuda = False


    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        similarity_matrix = nn.functional.linear(nn.functional.normalize(input,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight,p=2, dim=1, eps=1e-12))
        
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

        sp = sp.view(input.size()[0], -1)
        sn = sn.view(input.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

        return loss.mean()
        
        
        
if __name__ == "__main__":

    features = torch.rand(64, 128, requires_grad=True)
    label = torch.randint(high=9, size=(64,))

    SparseCircle = SparseCircleLoss(m=0.25, emdsize=128, class_num=10, gamma=64)
    loss = SparseCircle(features , label)

    print(loss)
