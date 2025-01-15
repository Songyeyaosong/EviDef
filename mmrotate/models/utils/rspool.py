# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn


class RSPool(nn.Module):

    def __init__(self, in_channels, orientation=8, num_layers=1):
        """
        RSPool is a module that aligns the features of each ROI according to the orientation of the ROI.

        Args:
            in_channels (int): Number of channels in the input feature map.
            orientation (int): Number of orientations.

        Returns:
            torch.Tensor: The aligned features.
        """
        super(RSPool, self).__init__()
        self.in_channels = in_channels
        self.orientation = orientation
        self.out_channels = int(in_channels / orientation)

    def forward(self, feats, rois):
        """
        Args:
            feats (torch.Tensor): Features of each level of the FPN feature map.
            rois (torch.Tensor): The rois of each image.
            
        Returns:
            torch.Tensor: The aligned features.
        """

        outs = []

        for feat, roi in zip(feats, rois):

            h, w = feat.shape[1], feat.shape[2]
            a = roi[:, 4]

            a = a + torch.pi / 4
            a = (torch.floor(a / (torch.pi / 8)).type(torch.long) * int(32)).reshape(-1, 1)
            bias = torch.arange(0, self.out_channels, device = a.device, dtype = torch.long)

            index = torch.arange(0, a.shape[0], dtype=torch.long, device=a.device).reshape(-1,1).repeat(1,self.out_channels).reshape(-1)
            j = (a + bias).reshape(-1)

            feat = feat.reshape(-1, a.shape[0]).permute(1,0)

            aligned_feat = feat[index, j]

            outs.append(aligned_feat.reshape(feat.shape[0], -1).permute(1,0).reshape(-1,h,w))

        outs = torch.stack(outs, dim=0)

        return outs