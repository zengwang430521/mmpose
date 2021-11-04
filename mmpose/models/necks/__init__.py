# Copyright (c) OpenMMLab. All rights reserved.
from .gap_neck import GlobalAveragePooling
from .inter_neck import UpSampleNeck, TokenInterNeck1, TokenInterNeck2
from .fpn import FPN, TokenFPN
from .resize_cat import ResizeCat, ResizeCat2
from .atten_neck import AttenNeck
from .atten_neck_sparse import AttenNeckS

__all__ = ['GlobalAveragePooling', 'UpSampleNeck', 'TokenInterNeck1', 'TokenInterNeck2',
           'FPN', 'TokenFPN', 'ResizeCat', 'ResizeCat2', 'AttenNeck', 'AttenNeckS']

