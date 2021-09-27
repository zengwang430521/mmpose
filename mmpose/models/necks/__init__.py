# Copyright (c) OpenMMLab. All rights reserved.
from .gap_neck import GlobalAveragePooling
from .inter_neck import UpSampleNeck, TokenInterNeck1, TokenInterNeck2

__all__ = ['GlobalAveragePooling', 'UpSampleNeck', 'TokenInterNeck1', 'TokenInterNeck2']
