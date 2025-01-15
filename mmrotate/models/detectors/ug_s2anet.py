# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

@ROTATED_DETECTORS.register_module()
class UGS2ANet(RotatedBaseDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 fam_head=None,
                 align_cfgs=None,
                 edl_head=None,
                 ug_ma_head=None,
                 ug_mi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(UGS2ANet, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            fam_head.update(train_cfg=train_cfg['fam_cfg'])
        fam_head.update(test_cfg=test_cfg)
        self.fam_head = build_head(fam_head)

        self.align_conv_type = align_cfgs['type']
        self.align_conv_size = align_cfgs['kernel_size']
        self.feat_channels = align_cfgs['channels']
        self.featmap_strides = align_cfgs['featmap_strides']

        if self.align_conv_type == 'AlignConv':
            self.align_conv = AlignConvModule(self.feat_channels,
                                              self.featmap_strides,
                                              self.align_conv_size)

        if train_cfg is not None:
            edl_head.update(train_cfg=train_cfg['odm_cfg'])
            ug_ma_head.update(train_cfg=train_cfg['odm_cfg'])
            ug_mi_head.update(train_cfg=train_cfg['odm_cfg'])
        edl_head.update(test_cfg=test_cfg)
        ug_ma_head.update(test_cfg=test_cfg)
        ug_mi_head.update(test_cfg=test_cfg)
        self.edl_head = build_head(edl_head)
        self.ug_ma_head = build_head(ug_ma_head)
        self.ug_mi_head = build_head(ug_mi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def set_epoch(self, epoch): 
        self.current_epoch = epoch + 1  # 因为传进来的epoch是以0开始的, 所以要加1

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function of S2ANet."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.fam_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.fam_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value

        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)

        edl_outs = self.edl_head(align_feat)
        loss_inputs = edl_outs + (gt_bboxes, gt_labels, img_metas)
        loss_edl = self.edl_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois, current_epoch=self.current_epoch)
        for name, value in loss_edl.items():
            losses[f'edl.{name}'] = value

        beta_cls_score, _ = edl_outs
        u = self.beta_cls_score_to_u(beta_cls_score=beta_cls_score)
        ma_weight = self.u_to_ma_weight(u)
        mi_weight = self.u_to_mi_weight(u)

        ug_ma_outs = self.ug_ma_head(align_feat)
        loss_inputs = ug_ma_outs + (gt_bboxes, gt_labels, img_metas)
        loss_ug_ma = self.ug_ma_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois, ug_weight=ma_weight)
        for name, value in loss_ug_ma.items():
            losses[f'ma.{name}'] = value

        ug_mi_outs = self.ug_mi_head(align_feat)
        loss_inputs = ug_mi_outs + (gt_bboxes, gt_labels, img_metas)
        loss_ug_mi = self.ug_mi_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois, ug_weight=mi_weight)
        for name, value in loss_ug_mi.items():
            losses[f'mi.{name}'] = value

        return losses
    
    def beta_cls_score_to_u(self, beta_cls_score):

        outs = []

        for score in beta_cls_score:

            e = F.softplus(score).permute(0,2,3,1)
            num_classes = int(e.shape[-1] / 2)

            shape = e.shape[:3] + (num_classes, 2)
            e = e.reshape(shape)
            e = 2 / (e[:,:,:,:,0] + e[:,:,:,:,1] + 2)

            outs.append(e.permute(0,3,1,2))

        return outs
    
    def u_to_ma_weight(self, mlu):

        outs = []

        for u in mlu:

            ma_weight = 1 - u
            outs.append(ma_weight)

        return outs
    
    def u_to_mi_weight(self, mlu):

        outs = []

        for u in mlu:

            mi_weight = 1 + u
            outs.append(mi_weight)

        return outs

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)

        # show_feat = x[0]
        # show_feature_map(show_feat)

        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)

        edl_outs = self.edl_head(align_feat)
        edl_cls_score, edl_bbox_pred = edl_outs

        # edl_cls_score = self.beta_cls_score_to_logit(edl_cls_score)
        # edl_outs = (edl_cls_score, edl_bbox_pred)
        # bbox_inputs = edl_outs + (img_meta, self.test_cfg, rescale)
        # bbox_list = self.edl_head.get_bboxes(*bbox_inputs, rois=rois)

        # bbox_results = [
        #     rbbox2result(det_bboxes, det_labels, self.edl_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]

        mlu = self.beta_cls_score_to_u(edl_cls_score)
        ma_outs = self.ug_ma_head(align_feat)
        mi_outs = self.ug_mi_head(align_feat)
        outs = self.merge_mami_bbox(mlu, ma_outs, mi_outs)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.ug_mi_head.get_bboxes(*bbox_inputs, rois=rois)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.ug_mi_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
    
    def merge_mami_bbox(self, mlu, ma_outs, mi_outs):

        ma_cls_scores, ma_bbox_preds = ma_outs
        mi_cls_scores, mi_bbox_preds = mi_outs

        cls_scores = []
        bbox_preds = []

        for u, ma_cls_score, mi_cls_score in zip(mlu, ma_cls_scores, mi_cls_scores):

            ma_cls_score, mi_cls_score = torch.sigmoid(ma_cls_score), torch.sigmoid(mi_cls_score)
            cls_score = (1 - u) * ma_cls_score + u * mi_cls_score
            cls_score = torch.logit(cls_score)
            # cls_score = torch.cat([cls_score, cls_score], dim=1)
            cls_scores.append(cls_score)

        # for ma_bbox_pred, mi_bbox_pred in zip(ma_bbox_preds, mi_bbox_preds):

        #     bbox_pred = torch.cat([ma_bbox_pred, mi_bbox_pred], dim=1)
        #     bbox_preds.append(bbox_pred)

        # return cls_scores, bbox_preds
        return cls_scores, mi_bbox_preds
    
    def beta_cls_score_to_logit(self, beta_cls_score):

        outs = []

        for score in beta_cls_score:

            ab = (F.softplus(score) + 1).permute(0,2,3,1)
            num_classes = int(ab.shape[-1] / 2)

            shape = ab.shape[:3] + (num_classes, 2)
            ab = ab.reshape(shape)
            ab = torch.logit(ab[:,:,:,:,0] / (ab[:,:,:,:,0] + ab[:,:,:,:,1]))

            outs.append(ab.permute(0,3,1,2))

        return outs
    
    def beta_cls_score_to_u(self, beta_cls_score):

        outs = []

        for score in beta_cls_score:

            e = F.softplus(score).permute(0,2,3,1)
            num_classes = int(e.shape[-1] / 2)

            shape = e.shape[:3] + (num_classes, 2)
            e = e.reshape(shape)
            e = 2 / (e[:,:,:,:,0] + e[:,:,:,:,1] + 2)

            outs.append(e.permute(0,3,1,2))

        return outs

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
 
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.plot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        plt.show()
        # break