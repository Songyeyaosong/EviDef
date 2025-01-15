# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn


class RSConv(nn.Module):
    """
    RSConv is a module that aligns the features of each ROI according to the orientation of the ROI.
    Args:
        in_channels (int): Number of channels in the input feature map.
        orientation (int): Number of orientations.
        num_layers (int): Number of conv layers.

    Returns:
        torch.Tensor: The aligned features.
    """

    def __init__(self, in_channels, orientation=8, num_layers=2):
        super(RSConv, self).__init__()
        self.in_channels = in_channels
        self.orientation = orientation
        self.out_channels = int(in_channels / orientation)
        self.rs_convs = self.init_layers(num_layers)

    def init_layers(self, num_layers):

        layers = []

        for _ in range(num_layers):
            layers.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(self.out_channels))

        return nn.Sequential(*layers)

    def forward(self, feats, rois):
        """
        Args:
            feats (torch.Tensor): Features of each level of the FPN feature map.
            rois (torch.Tensor): The rois of each image.

        Returns:
            torch.Tensor: The aligned features.
        """

        batch_size = feats.shape[0]
        h, w = feats.shape[2], feats.shape[3]
        aligned_feats = []
        indexs = []

        for feat, roi in zip(feats, rois):

            a = roi[:, 4]

            a = a + torch.pi / 4
            a = (torch.floor(a / (torch.pi / 8)).type(torch.long) * int(32)).reshape(-1, 1)
            bias = torch.arange(0, self.out_channels, device = a.device, dtype = torch.long)

            row_i = torch.arange(0, a.shape[0], dtype=torch.long, device=a.device).reshape(-1,1).repeat(1,self.out_channels).reshape(-1)
            col_i = (a + bias).reshape(-1)

            feat = feat.reshape(-1, a.shape[0]).permute(1,0)

            aligned_feat = feat[row_i, col_i]
            aligned_feats.append(aligned_feat.reshape(feat.shape[0], -1).permute(1,0).reshape(-1,h,w))
            indexs.append([row_i, col_i])

        aligned_feats = torch.stack(aligned_feats, dim=0)
        aligned_feats = self.rs_convs(aligned_feats)

        feats = feats.reshape(batch_size, self.in_channels, -1).permute(0,2,1)
        aligned_feats = aligned_feats.reshape(batch_size, self.out_channels, -1).permute(0,2,1)

        feat_col_i = torch.arange(0, self.out_channels, dtype=torch.long, device=aligned_feats.device).repeat(aligned_feats.shape[1])

        for i in range(batch_size):
            feat = feats[i]
            aligned_feat = aligned_feats[i]
            row_i = indexs[i][0]
            col_i = indexs[i][1]
            feat[row_i, col_i] = aligned_feat[row_i, feat_col_i]

        feats = feats.permute(0,2,1).reshape(batch_size, self.in_channels, h, w)
        
        return feats