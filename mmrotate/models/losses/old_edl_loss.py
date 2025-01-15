# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from ..builder import ROTATED_LOSSES

def exp_evidence(x):
    return torch.exp(torch.clamp(x, -10, 10))

def kl_d(alpha, target_onehot):

    alpha_hat = target_onehot + (1 - target_onehot) * alpha
    alpha_hat_sum = alpha_hat.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])

    kl = torch.lgamma(alpha_hat_sum) - torch.lgamma(K) - torch.lgamma(alpha_hat).sum(dim=1, keepdim=True) + \
        ((alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(alpha_hat_sum))).sum(dim=1, keepdim=True)
    
    return kl.squeeze(1)

def rx(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    K = torch.tensor(target_onehot.shape[1])
    u = K / s

    mask = target_onehot == 1
    p_gt = p[mask].unsqueeze(-1)

    l_rx = -p_gt * torch.log(1 - u + 1e-5) - (1 - p_gt) * torch.log(u + 1e-5)

    return l_rx.squeeze(-1)

def cor(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])
    u = K / s
    u_detach = u.detach()

    l_cor = (-target_onehot * u_detach * torch.log(alpha - 1 + 1e-5)).sum(dim=1)

    return l_cor.mean()

def log_loss(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    return (target_onehot * (torch.log(s) - torch.log(alpha))).sum(dim=1)

def digamma_loss(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    return (target_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1)

def mse_loss(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s

    loss1 = torch.square(target_onehot - p)
    loss2 = p * (1 - p) / (s + 1)

    loss = loss1 + loss2

    return loss.sum(dim=1)
    
def edl_loss(pred,
             target_onehot,
             weight=None,
             lambdat=1.0,
             current_epoch=None,
             reduction='mean',
             avg_factor=None):
    
    lambdat = min(lambdat, lambdat * (current_epoch - 1) / 10)

    evidence = F.softplus(pred)
    # evidence = exp_evidence(pred)
    alpha = evidence + 1

    loss = log_loss(alpha, target_onehot) + lambdat * kl_d(alpha, target_onehot) + cor(alpha, target_onehot)
    # loss = log_loss(alpha, target_onehot) + lambdat * rx(alpha, target_onehot)

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
class OLDEDLLoss(nn.Module):

    def __init__(self,
                 lambdat=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """Initialization.

        Args:
            use_sigmoid (bool, optional): Whether sigmoid operation is conducted
                in the prediction. Defaults to True.
            gamma (float, optional): Gamma parameter for focal loss.
                Defaults to 2.0.
            alpha (float, optional): Alpha parameter for focal loss.
                Defaults to 0.25.
            lambdat (float, optional): Lambda parameter for EDL loss.
                Defaults to 1.0.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.

        Notes:
            ``reduction`` is used to specify how the losses are averaged.
            Options are "none", "mean" and "sum".
            If ``reduction`` is "none", losses are not averaged and
            ``loss_weight`` is ignored.
            If ``reduction`` is "mean", the weighted losses are averaged
            over all observations.
            If ``reduction`` is "sum", the weighted losses are summed
            over all observations.
        """
        super(OLDEDLLoss, self).__init__()
        self.lambdat = lambdat
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                current_epoch=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            current_epoch (int, optional): Current epoch. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        target_onehot = F.one_hot(target, num_classes=pred.shape[-1]).float()

        loss_cls = self.loss_weight * edl_loss(
            pred,
            target_onehot,
            weight,
            lambdat=self.lambdat,
            current_epoch=current_epoch,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls