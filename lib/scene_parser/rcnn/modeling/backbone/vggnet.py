from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

class VGGNet(nn.Module):
    def __init__(self, cfg):
        super(VGGNet, self).__init__()
        builder = cfg.MODEL.BACKBONE.CONV_BODY.replace('-', '').lower()
        assert 'vgg' in builder
        vgg_model = getattr(models, builder)(pretrained=True)

        self.features = vgg_model.features[:-1] # remove last maxpool

        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        conv_index = 0
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = False
                conv_index += 1
                if conv_index == freeze_at:
                    break

    def forward(self, x):
        outputs = [self.features(x)]
        return outputs


class VGGNetHead(nn.Module):
    def __init__(self, cfg):
        super(VGGNetHead, self).__init__()
        builder = cfg.MODEL.BACKBONE.CONV_BODY.replace('-', '').lower()
        assert 'vgg' in builder
        vgg_model = getattr(models, builder)(pretrained=True)

        self.roi_fmap = vgg_model.classifier[:-1]
        self.out_channels = 4096

    def forward(self, x):
        x = self.roi_fmap(x)
        return x
