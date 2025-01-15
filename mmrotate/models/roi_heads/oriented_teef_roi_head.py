# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS
from .rotate_teef_roi_head import RotatedTeefRoIHead


@ROTATED_HEADS.register_module()
class OrientedTeefRoIHead(RotatedTeefRoIHead):
    """Oriented RCNN roi head including one bbox head."""

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      ted3_x,
                      img_metas,
                      proposal_list,
                    #   ted3_proposal_list,
                      gt_bboxes,
                      gt_labels,
                      current_epoch,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            # ted3_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # ted3_assign_result = self.bbox_assigner.assign(
                #     ted3_proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                #     gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # ted3_sampling_result = self.bbox_sampler.sample(
                #     ted3_assign_result,
                #     ted3_proposal_list[i],
                #     gt_bboxes[i],
                #     gt_labels[i],
                #     feats=[lvl_feat[i][None] for lvl_feat in ted3_x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                    # ted3_sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                    #     (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
                    # ted3_sampling_result.pos_gt_bboxes = \
                    #     gt_bboxes[i][ted3_sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)
                # ted3_sampling_results.append(ted3_sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, ted3_x, sampling_results,
                                                    # ted3_sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    current_epoch, img_metas)
            losses.update(bbox_results['loss_bbox'])
            # losses.update(bbox_results['ted3_loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, ted3_x, sampling_results,
                            # ted3_sampling_results,
                            gt_bboxes, gt_labels, current_epoch,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        # ted3_rois = rbbox2roi([res.bboxes for res in ted3_sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # ted3_bbox_targets = self.bbox_head.get_targets(ted3_sampling_results, gt_bboxes,
        #                                                gt_labels, self.train_cfg)
        
        bbox_results = self._bbox_forward(x, rois)
        ted3_bbox_results = self._ted3_bbox_forward(ted3_x, rois)

        p_e = F.softmax(bbox_results['cls_score'], dim=1)
        # e_sum = torch.exp(bbox_results['e_sum'])
        e_sum = F.softplus(bbox_results['e_sum'])
        ted3_p_e = F.softmax(ted3_bbox_results['ted3_cls_score'], dim=1)
        # ted3_e_sum = torch.exp(ted3_bbox_results['ted3_e_sum'])
        ted3_e_sum = F.softplus(ted3_bbox_results['ted3_e_sum'])
        e = e_sum * p_e
        ted3_e = ted3_e_sum * ted3_p_e

        p, u = self.refine_edl_score(e, ted3_e)
        K = torch.tensor(p.shape[1])
        a_sum = K / u
        a = a_sum * p
        e = a - 1
        e_sum = a_sum - K
        p_e = e / e_sum

        loss_bbox = self.bbox_head.loss(p_e,
                                        e_sum,
                                        ted3_bbox_results['ted3_bbox_pred'], rois,
                                        *bbox_targets, current_epoch)

        # loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
        #                                 bbox_results['e_sum'],
        #                                 bbox_results['bbox_pred'], rois,
        #                                 *bbox_targets, current_epoch)
        
        # ted3_loss_bbox = self.bbox_head.ted3_loss(ted3_bbox_results['ted3_cls_score'],
        #                                           ted3_bbox_results['ted3_e_sum'],
        #                                           ted3_bbox_results['ted3_bbox_pred'], rois,
        #                                           *bbox_targets, current_epoch)

        bbox_results.update(loss_bbox=loss_bbox)
        # bbox_results.update(ted3_loss_bbox=ted3_loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           ted3_x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        ted3_bbox_results = self._ted3_bbox_forward(ted3_x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        p_e = F.softmax(bbox_results['cls_score'], dim=1)
        # e_sum = torch.exp(bbox_results['e_sum'])
        e_sum = F.softplus(bbox_results['e_sum'])
        ted3_p_e = F.softmax(ted3_bbox_results['ted3_cls_score'], dim=1)
        # ted3_e_sum = torch.exp(ted3_bbox_results['ted3_e_sum'])
        ted3_e_sum = F.softplus(ted3_bbox_results['ted3_e_sum'])
        e = e_sum * p_e
        ted3_e = ted3_e_sum * ted3_p_e

        p, u = self.refine_edl_score(e, ted3_e)
        K = torch.tensor(p.shape[1])
        a_sum = K / u
        a = a_sum * p
        e = a - 1
        e_sum = a_sum - K
        cls_score = e / e_sum

        bbox_pred = ted3_bbox_results['ted3_bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        u = u.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                u[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
    
    def refine_edl_score(self, evidence1, evidence2):

        K = torch.tensor(evidence1.shape[1])
        # K = torch.tensor(edl_score1.shape[1])
        # evidence1 = F.softplus(edl_score1)
        # evidence2 = F.softplus(edl_score2)
        alpha1 = evidence1 + 1
        alpha2 = evidence2 + 1
        s1 = torch.sum(alpha1, dim=1, keepdim=True)
        s2 = torch.sum(alpha2, dim=1, keepdim=True)
        b1 = evidence1 / s1
        b2 = evidence2 / s2
        u1 = K / s1
        u2 = K / s2

        tmp_b1 = b1.unsqueeze(2).repeat(1, 1, K)
        tmp_b1 = tmp_b1.view(b1.size(0), -1)

        tmp_b2 = b2.unsqueeze(1).repeat(1, K, 1)
        tmp_b2 = tmp_b2.view(b2.size(0), -1)
        for i in range(K):
            tmp_b2[:, i * K + i] = 0
        tmp_b2 = tmp_b2.T

        C = torch.matmul(tmp_b1, tmp_b2)
        C = torch.diag(C).unsqueeze(1)

        refined_b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - C)
        refine_u = (u1 * u2) / (1 - C)
        refined_p = refined_b + refine_u / K

        return refined_p, refine_u
