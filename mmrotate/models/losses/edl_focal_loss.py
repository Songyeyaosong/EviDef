# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from ..builder import ROTATED_LOSSES


def edl_loss(evidence, target_onehot, reduction='none'):
    """ EDL loss.

    Args:
        evidence (torch.Tensor): The evidence of each class.
        target_onehot (torch.Tensor): The learning label of the prediction.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'none'.
            
    Returns:
        torch.Tensor: The calculated loss
    """

    ab = evidence + 1

    loss1 = (torch.digamma(ab[:,:,0] + ab[:,:,1]) - torch.digamma(ab[:,:,0])) * target_onehot
    loss2 = (torch.digamma(ab[:,:,0] + ab[:,:,1]) - torch.digamma(ab[:,:,1])) * (1 -target_onehot)

    loss = loss1 + loss2

    return loss

def u_loss(evidence, target_onehot, reduction='none'):
    """ U loss.

    Args:
        evidence (torch.Tensor): The evidence of each class.
        target_onehot (torch.Tensor): The learning label of the prediction.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'none'.

    Returns:
        torch.Tensor: The calculated loss
    """

    ab = evidence + 1

    a21_mask = target_onehot == 1
    b21_mask = (1 - target_onehot) == 1

    a_hat = ab[:,:,0].clone()
    b_hat = ab[:,:,1].clone()
    a_hat[a21_mask] = 1
    b_hat[b21_mask] = 1

    loss1 = torch.lgamma(a_hat + b_hat) - torch.lgamma(torch.tensor(2)) - torch.lgamma(a_hat) - torch.lgamma(b_hat)
    loss2 = (a_hat - 1) * (torch.digamma(a_hat) - torch.digamma(a_hat + b_hat)) \
        + (b_hat - 1) * (torch.digamma(b_hat) - torch.digamma(a_hat+ b_hat))
    
    loss = loss1 + loss2

    return loss


# This method is only for debugging
def edl_focal_loss(pred,
                   target_onehot,
                   weight=None,
                   gamma=2.0,
                   alpha=0.25,
                   lambdat=1.0,
                   tau=1.0,
                   nonlinear_trans='none',
                   current_epoch=None,
                   reduction='mean',
                   avg_factor=None):
    """EDL focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (B, N, 2).
        target_onehot (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        lambdat (float, optional): A parameter to control the largest loss
            value for hard example mining. Defaults to 1.0.
        current_epoch (int, optional): The current epoch, which is used for
            hard example mining. Defaults to None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    if current_epoch == None:
        raise NotImplementedError
    lambdat = min(lambdat, lambdat * current_epoch / 11.0)

    # evidence = F.softplus(pred)
    evidence = torch.exp(pred)
    ab = evidence + 1
    s = torch.sum(ab, dim=2)
    prob = ab[:,:,0] / s

    target_onehot = target_onehot.type_as(pred)
    pt = (1 - prob) * target_onehot + prob * (1 - target_onehot)
    focal_weight = (alpha * target_onehot + (1 - alpha) *
                    (1 - target_onehot)) * pt.pow(gamma)
    
    neg_prob = ab[:,:,1] / s
    u_pt = neg_prob * target_onehot + prob * (1 - target_onehot)
    loss_p = edl_loss(evidence, target_onehot, reduction='none')
    if lambdat == 0.0:
        u_weight = 0.0
        loss_u = torch.tensor(0.0).type_as(loss_p)
    else:
        u_weight = (alpha * target_onehot + (1 - alpha) *
                    (1 - target_onehot)) * u_pt.pow(gamma)
        loss_u = u_loss(evidence, target_onehot, reduction='none')

    # u_weight = 1

    if nonlinear_trans == 'none':
        loss_p = loss_p
        loss_u = loss_u
    elif nonlinear_trans == 'identity':
        loss_p = 1 - 1 / (tau + loss_p)
        loss_u = 1 - 1 / (tau + loss_u)
    elif nonlinear_trans == 'sqrt':
        loss_p = 1 - 1 / (tau + torch.sqrt(loss_p))
        loss_u = 1 - 1 / (tau + torch.sqrt(loss_u))
    elif nonlinear_trans == 'log':
        loss_p = 1 - 1 / (tau + torch.log(loss_p + 1))
        loss_u = 1 - 1 / (tau + torch.log(loss_u + 1))
    else:
        raise NotImplementedError
    
    loss = loss_p * focal_weight + loss_u * u_weight * lambdat

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

    # print(weight_reduce_loss(loss_p, weight, reduction, avg_factor))
    # print(weight_reduce_loss(loss_u, weight, reduction, avg_factor))

    return loss


@ROTATED_LOSSES.register_module()
class EDLFocalLoss(nn.Module):

    def __init__(self,
                 num_classes,
                 is_obj_score=False,
                 use_sigmoid=True,
                 need_bg=False,
                 gamma=2.0,
                 alpha=0.25,
                 lambdat=1.0,
                 tau=1.0,
                 nonlinear_trans='none',
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
        super(EDLFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        assert nonlinear_trans == 'none' \
            or nonlinear_trans == 'identity' \
            or nonlinear_trans == 'sqrt' \
            or nonlinear_trans == 'log', \
            'Only support none, identity, sqrt and log nonlinear transformation now.'
        self.num_classes = num_classes + 1 if need_bg else num_classes
        self.is_obj_score = is_obj_score
        self.use_sigmoid = use_sigmoid
        self.need_bg = need_bg
        self.gamma = gamma
        self.alpha = alpha
        self.lambdat = lambdat
        self.tau = tau
        self.nonlinear_trans = nonlinear_trans
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
        if self.use_sigmoid:

            if self.is_obj_score:
                pred = pred.reshape(-1, 1, 2)
                target_onehot = F.one_hot(target, num_classes=self.num_classes+1)
                target_onehot = target_onehot[:,:self.num_classes]
                target_onehot = target_onehot.sum(dim=1, keepdim=True)
                # target_onehot = torch.abs(target_onehot - 1)
                # target_onehot = F.one_hot(target_onehot, num_classes=2)
            else:
                pred = pred.reshape(-1,self.num_classes,2)
                if self.need_bg:
                    target_onehot = F.one_hot(target, num_classes=self.num_classes)  # if need_bg, num_classes = label_dim
                else:
                    target_onehot = F.one_hot(target, num_classes=self.num_classes + 1)  # if not need_bg, num_classes + 1 = label_dim
                target_onehot = target_onehot[:,:self.num_classes]

            loss_cls = self.loss_weight * edl_focal_loss(
                pred,
                target_onehot,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                lambdat=self.lambdat,
                tau=self.tau,
                nonlinear_trans=self.nonlinear_trans,
                current_epoch=current_epoch,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls