# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from ..builder import ROTATED_LOSSES

def xcor(p, u, target_onehot):

    l_cor = (-target_onehot * u * torch.log(p + 1e-5)).sum(dim=1)
    # l_cor = (-target_onehot * torch.log(p + 1e-5)).sum(dim=1)
    # l_cor = l_cor * 2.0

    return l_cor.mean()

def cor(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])
    u = K / s

    l_cor = (-target_onehot * u * torch.log(alpha - 1 + 1e-5)).sum(dim=1)

    return l_cor.mean()

def loss_euc(alpha, target_onehot, lambdat):

    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    K = torch.tensor(target_onehot.shape[1])
    u = K / s

    mask = target_onehot == 1
    p_gt = p[mask].unsqueeze(-1)

    preds = torch.argmax(p, dim=1)
    gt = torch.argmax(target_onehot, dim=1)
    is_correct = (preds == gt).float().unsqueeze(-1)

    num_all = torch.tensor(target_onehot.shape[0])
    num_correct = is_correct.sum()
    num_incorrect = num_all - num_correct

    # a = 0.25
    # gamma = 1.0
    # gap = torch.abs(u + p_gt - 1)

    l_euc_ce = -p_gt * torch.log(1 - u + 1e-5) - (1 - p_gt) * torch.log(u + 1e-5)
    l_euc_cor = (is_correct * l_euc_ce).sum() / num_correct * 0.1
    l_euc_inc = ((1 - is_correct) * l_euc_ce).sum() / num_incorrect * 0.5

    l_euc = l_euc_cor + l_euc_inc
    # l_euc = l_euc_ce.mean()
    l_euc = lambdat * l_euc

    return l_euc

def loss_orx(u, p, target_onehot, lambdat):

    mask = target_onehot == 1
    p_gt = p[mask].unsqueeze(-1)

    preds = torch.argmax(p, dim=1)
    gt = torch.argmax(target_onehot, dim=1)
    is_correct = (preds == gt).float().unsqueeze(-1)

    num_all = torch.tensor(target_onehot.shape[0])
    num_correct = is_correct.sum()
    num_incorrect = num_all - num_correct

    l_euc_cor = -p_gt * torch.log(1 - u + 1e-5)
    l_euc_inc = -(1 - p_gt) * torch.log(u + 1e-5)
    l_euc_cor = (is_correct * l_euc_cor).sum() / num_correct
    l_euc_inc = ((1 - is_correct) * l_euc_inc).sum() / num_incorrect

    l_euc = l_euc_cor + l_euc_inc
    l_euc = lambdat * l_euc

    return l_euc
    
def edl_loss_rx_cor(pred,
                    e_sum,
                    target_onehot,
                    weight=None,
                    lambdat=1.0,
                    current_epoch=None,
                    reduction='mean',
                    avg_factor=None):
    
    lambdat = min(lambdat, lambdat * (current_epoch - 1) / 10)
    # lambdat = min(lambdat, lambdat * current_epoch / 10)

    p = torch.softmax(pred, dim=1)
    e_sum = F.softplus(e_sum)
    K = torch.tensor(target_onehot.shape[1])
    s = e_sum + K
    e = e_sum * p
    alpha = e + 1

    # loss_ce = (target_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1)
    loss_ce = (-target_onehot * (torch.log(alpha / s + 1e-5))).sum(dim=1).mean()

    u = K / s
    # u_detach = u.detach()

    l_euc = loss_euc(alpha, target_onehot, lambdat)
    # l_cor = cor(alpha, target_onehot)
    l_xcor = xcor(p, u, target_onehot)

    loss = loss_ce + l_euc + l_xcor

    return loss


@ROTATED_LOSSES.register_module()
class EDLLossRXCor(nn.Module):

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
        super(EDLLossRXCor, self).__init__()
        self.lambdat = lambdat
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                e_sum,
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

        loss_cls = self.loss_weight * edl_loss_rx_cor(
            pred,
            e_sum,
            target_onehot,
            weight,
            lambdat=self.lambdat,
            current_epoch=current_epoch,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls