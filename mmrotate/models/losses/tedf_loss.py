# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from ..builder import ROTATED_LOSSES

def l_cor(alpha, target_onehot):

    K = torch.tensor(target_onehot.shape[1])
    s = alpha.sum(dim=1)
    u = K / s

    tmp = ((alpha - 1) * target_onehot).sum(dim=1) + 1e-4
    cor = -u * torch.log(tmp)
    
    return cor

def kl_d(alpha, target_onehot):

    alpha_hat = target_onehot + (1 - target_onehot) * alpha
    alpha_hat_sum = alpha_hat.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])

    kl = torch.lgamma(alpha_hat_sum) - torch.lgamma(K) - torch.lgamma(alpha_hat).sum(dim=1, keepdim=True) + \
        ((alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(alpha_hat_sum))).sum(dim=1, keepdim=True)
    
    return kl.squeeze(1)

def edl_loss(alpha,
             target_onehot):
    
    s = alpha.sum(dim=1, keepdim=True)

    loss = (target_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1)

    return loss

def l1_regular(w):

    regular = w.mean(dim=(1,2,3))
    return regular
    
def tedf_loss(pred,
              w,
              target_onehot,
              weight=None,
              l1_regular_weight=1.0,
              lambdat=1.0,
              current_epoch=None,
              reduction='mean',
              avg_factor=None):
    
    lambdat = min(lambdat, lambdat * (current_epoch - 1) / 10)

    evidence = F.softplus(pred)
    alpha = evidence + 1

    # loss = edl_loss(alpha, target_onehot) + lambdat * kl_d(alpha, target_onehot) + l_cor(alpha, target_onehot) + l1_regular_weight * l1_regular(w)
    loss = edl_loss(alpha, target_onehot) + lambdat * kl_d(alpha, target_onehot) + l1_regular_weight * l1_regular(w)

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss


@ROTATED_LOSSES.register_module()
class TedfLoss(nn.Module):

    def __init__(self,
                 lambdat=1.0,
                 l1_regular_weight=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        
        super(TedfLoss, self).__init__()
        self.lambdat = lambdat
        self.l1_regular_weight = l1_regular_weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                w,
                target,
                weight=None,
                avg_factor=None,
                current_epoch=None,
                reduction_override=None):
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        target_onehot = F.one_hot(target, num_classes=pred.shape[-1]).float()

        loss_cls = self.loss_weight * tedf_loss(
            pred,
            w,
            target_onehot,
            weight,
            l1_regular_weight=self.l1_regular_weight,
            lambdat=self.lambdat,
            current_epoch=current_epoch,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
