# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule
from mmdet.core import images_to_levels

import torch
import torch.nn.functional as F


@ROTATED_DETECTORS.register_module()
class FusionRetinanet(RotatedBaseDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 fusion_head=None,
                 final_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FusionRetinanet, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
            
        if train_cfg is not None:
            fusion_head.update(train_cfg=train_cfg['fusion_cfg'])
        fusion_head.update(test_cfg=test_cfg)
        self.fusion_head = build_head(fusion_head)

        if train_cfg is not None:
            final_head.update(train_cfg=train_cfg['final_cfg'])
        final_head.update(test_cfg=test_cfg)
        self.final_head = build_head(final_head)

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

        cls_scores, bbox_preds = self.fusion_head(x)
        fusion_feat = self.fusion_head.fusion_forward(x, cls_scores)
        loss_inputs = (cls_scores, bbox_preds) + (gt_bboxes, gt_labels, img_metas)
        loss_fusion = self.fusion_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, current_epoch=self.current_epoch)
        for name, value in loss_fusion.items():
            losses[f'fusion.{name}'] = value

        fusion_rois = self.fusion_head.refine_bboxes(cls_scores, bbox_preds, img_metas)
        outs = self.final_head(fusion_feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_refine = self.final_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=fusion_rois)
        for name, value in loss_refine.items():
            losses[f'final.{name}'] = value

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
        cls_score, bbox_pred = self.fusion_head(x)
        fusion_feat = self.fusion_head.fusion_forward(x, cls_score)
        fusion_rois = self.fusion_head.refine_bboxes(cls_score, bbox_pred, img_meta)
        cls_score, bbox_pred = self.final_head(fusion_feat)
        # cls_score = self.beta_cls_score_to_logit(beta_cls_score=cls_score)
        outs = (cls_score, bbox_pred)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.final_head.get_bboxes(*bbox_inputs, rois=fusion_rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.final_head.num_classes)
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

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
