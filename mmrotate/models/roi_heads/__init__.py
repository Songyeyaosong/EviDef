# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_heads import (RotatedBBoxHead, RotatedConvFCBBoxHead,
                         RotatedShared2FCBBoxHead)
from .gv_ratio_roi_head import GVRatioRoIHead
from .oriented_standard_roi_head import OrientedStandardRoIHead
from .roi_extractors import RotatedSingleRoIExtractor
from .roi_trans_roi_head import RoITransRoIHead
from .rotate_standard_roi_head import RotatedStandardRoIHead
from .tedf_roi_trans_roi_head import TedfRoITransRoIHead
from .oriented_tedf_roi_head import OrientedTedfRoIHead
from .rotate_tedf_roi_head import RotatedTedfRoIHead
from .oriented_edl_roi_head import OrientedEdlRoIHead
from .rotate_edl_roi_head import RotatedEdlRoIHead
from .oriented_oldedl_roi_head import OrientedOldEdlRoIHead
from .rotate_oldedl_roi_head import RotatedOldEdlRoIHead
from .oriented_teef_roi_head import OrientedTeefRoIHead
from .rotate_teef_roi_head import RotatedTeefRoIHead
from .teef_roi_trans_roi_head import TeefRoITransRoIHead

__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'RotatedStandardRoIHead', 'RotatedSingleRoIExtractor',
    'OrientedStandardRoIHead', 'RoITransRoIHead', 'GVRatioRoIHead',
    'TedfRoITransRoIHead', 'RotatedTedfRoIHead', 'OrientedTedfRoIHead',
    'OrientedEdlRoIHead', 'RotatedEdlRoIHead', 'OrientedOldEdlRoIHead',
    'RotatedOldEdlRoIHead', 'OrientedTeefRoIHead', 'RotatedTeefRoIHead',
    'TeefRoITransRoIHead'
]
