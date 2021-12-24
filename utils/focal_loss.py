import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
# import torch
# import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, size_average=True, num_classes = 15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        #self.alpha = alpha
        self.alpha = torch.Tensor([alpha,1-alpha])
        #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        #if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.num_classes = num_classes

    def forward(self, input, target):
        target = torch.squeeze(F.one_hot(target.long(), num_classes = self.num_classes)).to(torch.int64)
        print(target)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        # input = input.to(torch.int64)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, num_classes,alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = F.one_hot(targets.long(), num_classes = self.num_classes).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()