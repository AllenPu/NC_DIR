import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from loss import DRLoss


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


class ETFHead(nn.Module):
    def __init__(self, num_classes : int, feat_in : int, device = 'cpu') -> None:
        super(ETFHead, self).__init__()
        orth_vec = generate_random_orthogonal_matrix(feat_in, num_classes)
        i_nc_nc = torch.eye(num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(
            num_classes, num_classes), (1 / num_classes))
        self.etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
        self.etf_vec = self.etf_vec.to(device)


    def pre_logits(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


    def forward(self, x , y):
        x = self.pre_logits(x)
        #print(f' deive is {x.device} and {y.device} and {self.etf_vec.device}')
        #
        pred = self.etf_vec[:, y.flatten().to(torch.long)].t()
        pred = pred.to(x.device)
        #
        #print(f' deive is {x.device} and {y.device} and {self.etf_vec.device}')
        #
        drloss = DRLoss()
        loss = drloss.forward(x,pred)
        #
        return loss
        