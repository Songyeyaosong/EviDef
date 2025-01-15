# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .tem_two_stage import RotatedTemTwoStageDetector


@ROTATED_DETECTORS.register_module()
class TemRoITransformer(RotatedTemTwoStageDetector):
    """Implementation of `Learning RoI Transformer for Oriented Object
    Detection in Aerial Images.`__

    __ https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf#:~:text=The%20core%20idea%20of%20RoI%20Transformer%20is%20to,embed-%20ded%20into%20detectors%20for%20oriented%20object%20detection # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 tem_channels,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(TemRoITransformer, self).__init__(
            backbone=backbone,
            tem_channels=tem_channels,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
