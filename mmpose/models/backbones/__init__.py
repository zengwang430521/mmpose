# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .cpm import CPM
from .hourglass import HourglassNet
from .hrnet import HRNet
from .litehrnet import LiteHRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mspn import MSPN
from .regnet import RegNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .rsn import RSN
from .scnet import SCNet
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .tcn import TCN
from .vgg import VGG
from .vipnas_resnet import ViPNAS_ResNet
from .pvt_v2 import pvt_v2_b2
from .pvt_v2_20_2 import mypvt20_2_small
from .pvt_v2_3f12_1 import mypvt3f12_1_small
from .pvt_v2_3f12_2 import mypvt3f12_2_small
from .pvt_v2_3g import mypvt3g_small
from .pvt_v2_3g1 import mypvt3g1_small
from .pvt_v2_5f import mypvt5f_small
from .pvt_v2_3h1 import mypvt3h1_small
from .pvt_v2_3h2 import mypvt3h2_small
from .pvt_v2_3h2_fast_norm import mypvt3h2_fast_norm_small
from .pvt_v2_3h2a import mypvt3h2a_small
from .pvt_v2_3h11 import mypvt3h11_small
from .hrt import HRT
from .hrpvt import HRPVT
from .myhrpvt import MyHRPVT


__all__ = [
    'AlexNet', 'HourglassNet', 'HRNet', 'MobileNetV2', 'MobileNetV3', 'RegNet',
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SCNet', 'SEResNet', 'SEResNeXt',
    'ShuffleNetV1', 'ShuffleNetV2', 'CPM', 'RSN', 'MSPN', 'ResNeSt', 'VGG',
    'TCN', 'ViPNAS_ResNet', 'LiteHRNet', 'pvt_v2_b2', 'mypvt20_2_small', 'mypvt3f12_1_small',
    'mypvt3g_small', 'mypvt3g1_small', 'mypvt5f_small', 'mypvt3h2_small', 'mypvt3h1_small',
    'mypvt3h11_small', 'HRPVT', 'HRT', 'mypvt3h2a_small', 'MyHRPVT', 'mypvt3h2_fast_norm_small'
]
