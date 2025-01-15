# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from ..builder import ROTATED_LOSSES

def loss_euc(u, p, target_onehot, lambdat):

    mask = target_onehot == 1
    p_gt = p[mask].unsqueeze(-1)

    preds = torch.argmax(p, dim=1)
    gt = torch.argmax(target_onehot, dim=1)
    is_correct = (preds == gt).float().unsqueeze(-1)

    l_euc1 = is_correct * torch.log(p_gt * (1 - u) + 1e-5)
    l_euc2 = (1 - is_correct) * torch.log((1 - p_gt) * u + 1e-5)

    l_euc = -lambdat * l_euc1 - (1 - lambdat) * l_euc2

    return l_euc.squeeze(-1)
    
def edl_loss_v4(pred,
                e_sum,
                target_onehot,
                weight=None,
                lambdat=1.0,
                current_epoch=None,
                reduction='mean',
                avg_factor=None):
    
    lambdat = min(lambdat, lambdat * current_epoch / 20)

    p = torch.softmax(pred, dim=1)
    loss_ce = (-target_onehot * torch.log(p + 1e-5)).sum(dim=1)

    p_detach = p.detach()
    u = torch.sigmoid(e_sum)

    l_euc = loss_euc(u, p_detach, target_onehot, lambdat)

    loss = loss_ce + l_euc

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
class EDLLossV4(nn.Module):

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
        super(EDLLossV4, self).__init__()
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

        loss_cls = self.loss_weight * edl_loss_v4(
            pred,
            e_sum,
            target_onehot,
            weight,
            lambdat=self.lambdat,
            current_epoch=current_epoch,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls