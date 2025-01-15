# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .gliding_vertex import GlidingVertex
from .oriented_rcnn import OrientedRCNN
from .r3det import R3Det
from .redet import ReDet
from .roi_transformer import RoITransformer
from .rotate_faster_rcnn import RotatedFasterRCNN
from .rotated_fcos import RotatedFCOS
from .rotated_reppoints import RotatedRepPoints
from .rotated_retinanet import RotatedRetinaNet
from .s2anet import S2ANet
from .single_stage import RotatedSingleStageDetector
from .two_stage import RotatedTwoStageDetector
from .edl_fusion_s2anet import EDLFUSIONS2ANet
from .edl_aligned_fusion_s2anet import EdlAlignedFusionS2ANet
from .edl_aligned_res_fusion_s2anet import EdlAlignedResFusionS2ANet
from .edl_res_fusion_s2anet import EdlResFusionS2ANet
from .fusion_retinanet import FusionRetinanet
from .edl_s2anet import EDLS2ANet
from .edl_tem_s2anet import EDLTEMS2ANet
from .tem_s2anet import TEMS2ANet
from .ug_s2anet import UGS2ANet
from .tedf_roi_transformer import TedfRoITransformer
from .tedf_two_stage import RotatedTedfTwoStageDetector
from .oriented_tedf_rcnn import OrientedTedfRCNN
from .edl_oriented_rcnn import EdlOrientedRCNN
from .edl_two_stage import RotatedEdlTwoStageDetector
from .tem_oriented_rcnn import TemOrientedRCNN
from .tem_two_stage import RotatedTemTwoStageDetector
from .oldedl_oriented_rcnn import OldEdlOrientedRCNN
from .oldedl_two_stage import RotatedOldEdlTwoStageDetector
from .tem_edl_oriented_rcnn import TemEdlOrientedRCNN
from .tem_edl_two_stage import RotatedTemEdlTwoStageDetector
from .teef_oriented_rcnn import TeefOrientedRCNN
from .teef_two_stage import RotatedTeefTwoStageDetector
from .tem_roi_transformer import TemRoITransformer
from .teef_roi_transformer import TeefRoITransformer

__all__ = [
    'RotatedRetinaNet', 'RotatedFasterRCNN', 'OrientedRCNN', 'RoITransformer',
    'GlidingVertex', 'ReDet', 'R3Det', 'S2ANet', 'RotatedRepPoints',
    'RotatedBaseDetector', 'RotatedTwoStageDetector',
    'RotatedSingleStageDetector', 'RotatedFCOS', 'EDLFUSIONS2ANet',
    'EdlAlignedFusionS2ANet', 'EdlAlignedResFusionS2ANet',
    'EdlResFusionS2ANet', 'FusionRetinanet', 'EDLS2ANet',
    'EDLTEMS2ANet', 'TEMS2ANet', 'UGS2ANet', 
    'TedfRoITransformer', 'RotatedTedfTwoStageDetector',
    'OrientedTedfRCNN', 'EdlOrientedRCNN', 'RotatedEdlTwoStageDetector',
    'TemOrientedRCNN', 'RotatedTemTwoStageDetector', 'OldEdlOrientedRCNN',
    'RotatedOldEdlTwoStageDetector', 'TemEdlOrientedRCNN', 'RotatedTemEdlTwoStageDetector',
    'TeefOrientedRCNN', 'RotatedTeefTwoStageDetector', 'TemRoITransformer', 'TeefRoITransformer'
]
