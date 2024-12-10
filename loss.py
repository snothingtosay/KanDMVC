import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self,batch_size,temperature,device):
        super(ContrastiveLoss,self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        N = self.batch_size
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N), device=self.device).fill_diagonal_(0)

        nominator = torch.exp(positives)
        denominator = (mask.bool()) * torch.exp(similarity_matrix)

        # 添加小常数以避免除以零
        denominator_sum = torch.sum(denominator, dim=1) + 1e-10
        loss_partial = -torch.log(nominator / denominator_sum)
        loss = torch.sum(loss_partial) / N
        loss = weight * loss if weight is not None else loss
        return loss
