# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from ..builder import ROTATED_LOSSES

def entropy_reg(p, e_sum):

    entropy = (-p * torch.log(p)).sum(dim=1, keepdim=True)
    reg = entropy * torch.log(e_sum + 1)

    return reg.mean()

def xcor(p, target_onehot):

    # l_cor = (-target_onehot * u * torch.log(p + 1e-5)).sum(dim=1)
    l_cor = -torch.log((target_onehot * p).sum(dim=1, keepdim=True))
    # l_cor = l_cor * 2.0

    return l_cor.mean()

def cor(alpha, target_onehot):

    s = alpha.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])
    u = K / s

    l_cor = (-target_onehot * u * torch.log(alpha - 1 + 1e-5)).sum(dim=1)

    return l_cor.mean()

def kl_d(alpha, target_onehot, lambdat):

    alpha_hat = target_onehot + (1 - target_onehot) * alpha
    alpha_hat_sum = alpha_hat.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])

    kl = torch.lgamma(alpha_hat_sum) - torch.lgamma(K) - torch.lgamma(alpha_hat).sum(dim=1, keepdim=True) + \
        ((alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(alpha_hat_sum))).sum(dim=1, keepdim=True)
    kl = lambdat * kl.mean()

    return kl

def kl_w(alpha, target_onehot):

    alpha_hat = target_onehot + (1 - target_onehot) * alpha
    alpha_hat_sum = alpha_hat.sum(dim=1, keepdim=True)
    K = torch.tensor(target_onehot.shape[1])

    kl = torch.lgamma(alpha_hat_sum) - torch.lgamma(K) - torch.lgamma(alpha_hat).sum(dim=1, keepdim=True) + \
        ((alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(alpha_hat_sum))).sum(dim=1, keepdim=True)

    return kl

def loss_cl(e_sum, p, target_onehot):

    p_not_gt = (p * (1 - target_onehot)).sum(dim=1, keepdim=True)
    l_cl = torch.log(e_sum + 1) * torch.square(p_not_gt)

    return l_cl
    
def edl_loss_cl(pred,
                e_sum,
                target_onehot,
                weight=None,
                lambdat=1.0,
                current_epoch=None,
                reduction='mean',
                avg_factor=None):
    
    lambdat = min(lambdat, lambdat * (current_epoch - 1) / 10)
    # lambdat = min(lambdat, lambdat * current_epoch / 10)
    # if current_epoch == 1:
    #     lambdat = 0.001

    p = torch.softmax(pred, dim=1) # p represents p^e
    # p = pred
    e_sum = torch.exp(e_sum)
    K = torch.tensor(target_onehot.shape[1])
    s = e_sum + K
    e = e_sum * p
    alpha = e + 1

    # pred_label = p.argmax(dim=1, keepdim=True)
    # gt_label = target_onehot.argmax(dim=1, keepdim=True)

    # correct_mask = (pred_label == gt_label).int()
    # incorrect_mask = 1 - correct_mask
    # correct_num = correct_mask.sum()
    # incorrect_num = incorrect_mask.sum()

    # loss_ce = (target_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1)
    loss_ce = (-target_onehot * (torch.log(alpha / s + 1e-5))).sum(dim=1).mean() * 0.5

    # u = K / s
    # u_detach = u.detach()
    # loss = loss_ce + kl_d(alpha, target_onehot, lambdat) + u_detach * xcor(p, target_onehot)

    # alpha_p_only = e_sum * p.detach() + 1
    # loss = loss_ce + kl_d(alpha_p_only, target_onehot, lambdat) + xcor(p, target_onehot)

    # loss = loss_ce + kl_d(alpha, target_onehot, lambdat) + xcor(p, target_onehot)
    # loss = loss_ce + loss_cl(e_sum, p, target_onehot, lambdat)
    # alpha_detach = alpha.detach()
    kl_weight = 1 - (1 / (1 + kl_w(alpha, target_onehot)))

    new_cl = (kl_weight * torch.log(e_sum + 1))
    # new_cl_cor = 0 if correct_num == 0 else ((new_cl * correct_mask).sum() / correct_num)
    # new_cl_inc = 0 if incorrect_num == 0 else ((new_cl * incorrect_mask).sum() / incorrect_num)

    loss = loss_ce + lambdat * new_cl.mean() + xcor(p, target_onehot)

    return loss


@ROTATED_LOSSES.register_module()
class EDLLossCL(nn.Module):

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
        super(EDLLossCL, self).__init__()
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

        loss_cls = self.loss_weight * edl_loss_cl(
            pred,
            e_sum,
            target_onehot,
            weight,
            lambdat=self.lambdat,
            current_epoch=current_epoch,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls