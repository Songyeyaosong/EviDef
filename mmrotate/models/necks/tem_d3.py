import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS

@ROTATED_NECKS.register_module()
class TEMD3(BaseModule):

    def __init__(self, tem_channels):
        super(TEMD3, self).__init__()

        self.stage_convs = nn.ModuleList()

        for num_channels in tem_channels:

            convs = nn.ModuleList()

            in_channels = num_channels
            out_channels = int(num_channels / 4)

            # large version
            # out_channels = num_channels

            in_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            te_convs1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            te_convs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1,1), padding=(0,0)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1,1), padding=(0,0)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1), dilation=1)
            )
            te_convs3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(2,2), dilation=2)
            )
            te_convs4 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(3,3), dilation=3)
            )
            out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

            # large version
            # out_conv = nn.Conv2d(int(out_channels * 4), in_channels, kernel_size=1)

            convs.append(in_conv)
            convs.append(te_convs1)
            convs.append(te_convs2)
            convs.append(te_convs3)
            convs.append(te_convs4)
            convs.append(out_conv)

            self.stage_convs.append(convs)

    @auto_fp16()
    def forward(self, x):

        out = []
        
        for i, feat in enumerate(x):

            feat_in = self.stage_convs[i][0](feat)
            feat1 = self.stage_convs[i][1](feat)
            feat2 = self.stage_convs[i][2](feat)
            feat3 = self.stage_convs[i][3](feat)
            feat4 = self.stage_convs[i][4](feat)

            feat_cat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
            feat_out = self.stage_convs[i][5](feat_cat) + feat_in
            out.append(feat_out)

        return out