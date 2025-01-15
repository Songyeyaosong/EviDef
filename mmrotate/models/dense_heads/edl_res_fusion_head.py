# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS
from .edl_fusion_anchor_head import EdlFusionAnchorHead

import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF


@ROTATED_HEADS.register_module()
class EdlResFusionHead(EdlFusionAnchorHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 is_obj_score=False,
                 max_rate=2.0,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.is_obj_score = is_obj_score
        super(EdlResFusionHead, self).__init__(
            num_classes,
            in_channels,
            is_obj_score=is_obj_score,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.zoom_rate = nn.Sequential(
            nn.Linear(self.in_channels * 2, self.in_channels)
            # nn.GELU(),
            # nn.AdaptiveAvgPool1d(1)
        )
        self.max_rate = max_rate

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        
        self.edl_retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * 2 if self.is_obj_score else self.num_anchors * self.cls_out_channels * 2,
            3,
            padding=1)
        self.edl_retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.edl_retina_cls(cls_feat)
        bbox_pred = self.edl_retina_reg(reg_feat)
        return cls_score, bbox_pred
    
    # def forward(self, feats):

    #     # TODO: add context fusion

    #     reversed_feats = list(reversed(feats))
    #     reversed_cls_scores = []
    #     reversed_bbox_preds = []

    #     for i, _ in enumerate(reversed_feats):

    #         if i == 0:
    #             continue

    #         feat_top = reversed_feats[i - 1]
    #         feat_down = reversed_feats[i]

    #         cls_score_top, bbox_pred_top = self.forward_single(feat_top)
    #         cls_score_down, bbox_pred_down = self.forward_single(feat_down)

    #         fusion_feat = self.evidence_fusion_module(feat_top, feat_down, cls_score_top, cls_score_down)

    #         reversed_cls_scores.append(cls_score_top)
    #         reversed_bbox_preds.append(bbox_pred_top)

    #         down_size = feat_down.shape[-2:]
    #         reversed_feats[i] = fusion_feat + TF.resize(feat_top, down_size)

    #     cls_score_down, bbox_pred_down = self.forward_single(reversed_feats[-1])
    #     reversed_cls_scores.append(cls_score_down)
    #     reversed_bbox_preds.append(bbox_pred_down)

    #     cls_scores = list(reversed(reversed_cls_scores))
    #     bbox_preds = list(reversed(reversed_bbox_preds))
    #     fusion_feats = list(reversed(reversed_feats))

    #     return cls_scores, bbox_preds, fusion_feats

    def fusion_forward(self, feats, cls_scores):

        # TODO: add context fusion

        reversed_feats = list(reversed(feats))
        reversed_fusion_feats = []
        reversed_cls_scores = list(reversed(cls_scores))

        for i, _ in enumerate(reversed_feats):

            if i == 0:
                reversed_fusion_feats.append(reversed_feats[i])
            else:
                feat_top = reversed_feats[i - 1]
                feat_down = reversed_feats[i]

                cls_score_top = reversed_cls_scores[i - 1]
                cls_score_down = reversed_cls_scores[i]

                fusion_feat = self.evidence_fusion_module(feat_top, feat_down, cls_score_top, cls_score_down)

                # reversed_fusion_feats.append(fusion_feat)

                # TODO: add residual fusion
                fusion_feat_top = reversed_fusion_feats[-1]
                down_size = feat_down.shape[-2:]
                reversed_fusion_feats.append(fusion_feat + TF.resize(fusion_feat_top, down_size))

        fusion_feats = list(reversed(reversed_fusion_feats))

        return fusion_feats

    def evidence_fusion_module(self, feat_top, feat_down, cls_score_top, cls_score_down):

        down_size = feat_down.shape[-2:]

        evidence_top = F.softplus(cls_score_top)
        evidence_down = F.softplus(cls_score_down)

        top_e1, top_e2 = evidence_top[:, 0, :, :], evidence_top[:, 1, :, :]
        down_e1, down_e2 = evidence_down[:, 0, :, :], evidence_down[:, 1, :, :]

        top_a1, top_a2 = top_e1 + 1, top_e2 + 1
        down_a1, down_a2 = down_e1 + 1, down_e2 + 1
        top_s, down_s = top_a1 + top_a2, down_a1 + down_a2

        top_b1, top_b2, top_u = top_e1 / top_s, top_e2 / top_s, 2 / top_s
        down_b1, down_b2, down_u = down_e1 / down_s, down_e2 / down_s, 2 / down_s

        c = TF.resize(top_b1, down_size) * down_b2 + TF.resize(top_b2, down_size) * down_b1

        fusion_feat = (TF.resize(top_b1[:,None,:,:] * feat_top, down_size) * down_b1[:,None,:,:] * feat_down + \
                       TF.resize(top_b1[:,None,:,:] * feat_top, down_size) * down_u[:,None,:,:] * feat_down + \
                        down_b1[:,None,:,:] * feat_down * TF.resize(top_u[:,None,:,:] * feat_top, down_size)) / \
                            (1 - c)[:,None,:,:]
        
        gap = self.gap(fusion_feat)
        gmp = self.gmp(fusion_feat)
        gap_gmp = torch.cat([gap, gmp], dim=1).reshape(gap.shape[0], -1)
        zoom_rate = self.zoom_rate(gap_gmp).sigmoid() * self.max_rate
        fusion_feat = fusion_feat * zoom_rate[:,:,None,None]
        
        return fusion_feat
    
    # def max_evidence(self, evidence):

    #     evidence = evidence.permute(0, 2, 3, 1)
    #     batch, h, w = evidence.shape[:3]
    #     evidence = evidence.reshape(batch, h, w, self.num_classes, 2)
    #     e1 = evidence[..., 0]
    #     e2 = evidence[..., 1]
    #     s = e1 + e2 + 2
    #     b1 = e1 / s

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            bboxes_as_anchors (list[list[Tensor]]) bboxes of levels of images.
                before further regression just like anchors.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image
                - valid_flag_list (list[Tensor]): Valid flags of each image
        """
        anchor_list = [[
            bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             rois=None,
             current_epoch=None):
        """Loss function of EdlFusionHead."""
        assert rois is not None
        self.bboxes_as_anchors = rois
        return super(EdlResFusionHead, self).loss(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            current_epoch=current_epoch)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self, cls_scores, bbox_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i,
                                                     best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, rois, img_metas):
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        assert rois is not None
        num_imgs = cls_scores[0].size(0)
        self.bboxes_as_anchors = rois
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # mlvl_anchors = self.anchor_generator.grid_priors(
        #     featmap_sizes, device=device)
        mlvl_anchors, _ = self.get_anchors(featmap_sizes, img_metas, device=device)
        lv_anchors = [torch.stack(anchor, dim=0) for anchor in zip(*mlvl_anchors)]  # 将anchors从按batch排列转换为按level排列

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            anchors = lv_anchors[lvl]
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)

            for img_id in range(num_imgs):
                anchors_i = anchors[img_id]
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors_i, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list
