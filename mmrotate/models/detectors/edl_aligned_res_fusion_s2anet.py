# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule
from mmdet.core import images_to_levels

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


@ROTATED_DETECTORS.register_module()
class EdlAlignedResFusionS2ANet(RotatedBaseDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 fam_head=None,
                 align_cfgs=None,
                 fusion_head=None,
                 odm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EdlAlignedResFusionS2ANet, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            fam_head.update(train_cfg=train_cfg['fam_cfg'])  # update: 在cfg中加入一个dict
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
            fusion_head.update(train_cfg=train_cfg['fusion_cfg'])
        fusion_head.update(test_cfg=test_cfg)
        self.fusion_head = build_head(fusion_head)

        if train_cfg is not None:
            odm_head.update(train_cfg=train_cfg['odm_cfg'])
        odm_head.update(test_cfg=test_cfg)
        self.odm_head = build_head(odm_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def set_epoch(self, epoch): 
        self.current_epoch = epoch + 1  # 因为传进来的epoch是以0开始的, 所以要加1

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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

        cls_scores, bbox_preds  = self.fam_head(x)  # cls_score, bbox_pred
        loss_inputs = (cls_scores, bbox_preds) + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.fam_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value

        rois = self.fam_head.refine_bboxes(cls_scores, bbox_preds)  # rois里面的角度是弧度: pi转换成的数值
        # rois: list(indexed by images) of list(indexed by levels)
        # levels_rois = [torch.stack(roi, dim=0) for roi in zip(*rois)]  # 将rois从按batch排列转换为按level排列
        align_feat = self.align_conv(x, rois)

        cls_scores, bbox_preds = self.fusion_head(align_feat)
        fusion_feat = self.fusion_head.fusion_forward(align_feat, cls_scores)
        loss_inputs = (cls_scores, bbox_preds) + (gt_bboxes, gt_labels, img_metas)
        loss_fusion = self.fusion_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois, current_epoch=self.current_epoch)
        for name, value in loss_fusion.items():
            losses[f'fusion.{name}'] = value

        fusion_rois = self.fusion_head.refine_bboxes(cls_scores, bbox_preds, rois, img_metas)
        outs = self.odm_head(fusion_feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_refine = self.odm_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=fusion_rois)
        for name, value in loss_refine.items():
            losses[f'odm.{name}'] = value

        return losses

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
        # show_feature_map(x[0])
        cls_scores, bbox_preds = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(cls_scores, bbox_preds)
        # rois: list(indexed by images) of list(indexed by levels)
        # levels_rois = [torch.stack(roi, dim=0) for roi in zip(*rois)]  # 将rois从按batch排列转换为按level排列
        align_feat = self.align_conv(x, rois)
        # show_feature_map(align_feat[0])
        cls_score, bbox_pred = self.fusion_head(align_feat)
        b1 = self.beta_cls_score_to_b1(cls_score)
        b2 = self.beta_cls_score_to_b2(cls_score)
        u = self.beta_cls_score_to_u(cls_score)
        show_feature_map(b1[0])
        show_feature_map(b2[0])
        show_feature_map(u[0])
        show_feature_map(b1[0] + u[0])
        show_feature_map(b1[1])
        show_feature_map(b2[1])
        show_feature_map(u[1])
        show_feature_map(b1[1] + u[1])
        fusion_feat = self.fusion_head.fusion_forward(align_feat, cls_score)
        # show_feature_map(fusion_feat[0])
        fusion_rois = self.fusion_head.refine_bboxes(cls_score, bbox_pred, rois, img_meta)
        cls_score, bbox_pred = self.odm_head(fusion_feat)
        # cls_score = self.beta_cls_score_to_logit(beta_cls_score=cls_score)
        outs = (cls_score, bbox_pred)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=fusion_rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
    
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
    
    def beta_cls_score_to_b1(self, beta_cls_score):

        outs = []

        for score in beta_cls_score:

            e = F.softplus(score).permute(0,2,3,1)
            num_classes = int(e.shape[-1] / 2)

            shape = e.shape[:3] + (num_classes, 2)
            e = e.reshape(shape)
            e = e[:,:,:,:,0] / (e[:,:,:,:,0] + e[:,:,:,:,1] + 2)

            outs.append(e.permute(0,3,1,2))

        return outs
    
    def beta_cls_score_to_b2(self, beta_cls_score):

        outs = []

        for score in beta_cls_score:

            e = F.softplus(score).permute(0,2,3,1)
            num_classes = int(e.shape[-1] / 2)

            shape = e.shape[:3] + (num_classes, 2)
            e = e.reshape(shape)
            e = e[:,:,:,:,1] / (e[:,:,:,:,0] + e[:,:,:,:,1] + 2)

            outs.append(e.permute(0,3,1,2))

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

import matplotlib.pyplot as plt
import numpy as np

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