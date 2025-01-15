# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead)
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead
from .rotated_tedf_bbox_head import RotatedTedfBBoxHead
from .tedf_convfc_rbbox_head import (RotatedTedfConvFCBBoxHead,
                                     RotatedTedfShared2FCBBoxHead,
                                     RotatedTedfKFIoUShared2FCBBoxHead)
from .rotated_edl_bbox_head import RotatedEdlBBoxHead
from .edl_convfc_rbbox_head import (RotatedEdlConvFCBBoxHead,
                                    RotatedEdlShared2FCBBoxHead,
                                    RotatedEdlKFIoUShared2FCBBoxHead)
from .rotated_oldedl_bbox_head import RotatedOldEdlBBoxHead
from .oldedl_convfc_rbbox_head import (RotatedOldEdlConvFCBBoxHead,
                                       RotatedOldEdlShared2FCBBoxHead,
                                       RotatedOldEdlKFIoUShared2FCBBoxHead)
from .rotated_teef_bbox_head import RotatedTeefBBoxHead
from .teef_convfc_rbbox_head import (RotatedTeefConvFCBBoxHead,
                                       RotatedTeefShared2FCBBoxHead,
                                       RotatedTeefKFIoUShared2FCBBoxHead)

__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead', 'RotatedTedfBBoxHead',
    'RotatedTedfConvFCBBoxHead', 'RotatedTedfShared2FCBBoxHead', 'RotatedTedfKFIoUShared2FCBBoxHead',
    'RotatedEdlBBoxHead', 'RotatedEdlConvFCBBoxHead', 'RotatedEdlShared2FCBBoxHead', 'RotatedEdlKFIoUShared2FCBBoxHead',
    'RotatedOldEdlBBoxHead', 'RotatedOldEdlConvFCBBoxHead', 'RotatedOldEdlShared2FCBBoxHead', 'RotatedOldEdlKFIoUShared2FCBBoxHead',
    'RotatedTeefBBoxHead', 'RotatedTeefConvFCBBoxHead', 'RotatedTeefShared2FCBBoxHead', 'RotatedTeefKFIoUShared2FCBBoxHead'
]
