import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init

from mmpose.core.post_processing import flip_back
from mmpose.models.backbones.resnet import Bottleneck
from mmpose.models.necks import GlobalAveragePooling
from ..registry import HEADS
from .heatmap_1d_head import Heatmap1DHead
from .heatmap_3d_head import Heatmap3DHead
from .multilabel_classification_head import MultilabelClassificationHead
from .top_down_simple_head import TopDownSimpleHead


@HEADS.register_module()
class Interhand3DHead(nn.Module):
    """Interhand 3D head of paper ref: Gyeongsik Moon. "InterHand2.6M: A
    Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single
    RGB Image".

    Args:
        keypoint_head_cfg (dict): Configs of Heatmap3DHead for hand
        keypoints estimation.
        root_head_cfg (dict): Configs of Heatmap1DHead for relative
        hand root depth estimation.
        hand_type_head_cfg (dict): Configs of MultilabelClassificationHead
        for hand type classification.
    """

    def __init__(self,
                 keypoint_head_cfg,
                 root_head_cfg,
                 hand_type_head_cfg,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        # build heads
        self.right_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.left_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.root_head = Heatmap1DHead(**root_head_cfg)
        self.hand_type_head = MultilabelClassificationHead(
            **hand_type_head_cfg)
        self.neck = GlobalAveragePooling()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        self.left_hand_head.init_weights()
        self.right_hand_head.init_weights()
        self.root_head.init_weights()
        self.hand_type_head.init_weights()

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
            multiple heads.
        """
        losses = dict()
        losses['hand_loss'] = self.right_hand_head.get_loss(
            output[0], target[0], target_weight[0])['heatmap_loss']
        losses['rel_root_loss'] = self.root_head.get_loss(
            output[1], target[1], target_weight[1])['value_loss']
        losses['hand_type_loss'] = self.hand_type_head.get_loss(
            output[2], target[2], target_weight[2])['classification_loss']
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
            multiple heads.
        """
        acc = {}
        acc['acc_hand_type'] = self.hand_type_head.get_accuracy(
            output[2], target[2], target_weight[2])['acc_classification']
        return acc

    def forward(self, x):
        """Forward function."""
        outputs = []
        outputs.append(
            torch.cat([self.right_hand_head(x),
                       self.left_hand_head(x)], dim=1))
        x = self.neck(x)
        outputs.append(self.root_head(x))
        outputs.append(self.hand_type_head(x))
        return outputs

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output (list[np.ndarray]): list of output hand keypoint
            heatmaps, relative root depth and hand type.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """

        output = self.forward(x)

        if flip_pairs is not None:
            heatmap_3d = output[0]
            N, K, D, H, W = heatmap_3d.shape
            # reshape 3D heatmap to 2D heatmap
            heatmap_3d = heatmap_3d.reshape(N, K * D, H, W)
            # 2D heatmap flip
            heatmap_3d_flipped_back = flip_back(
                heatmap_3d.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.right_hand_head.target_type)
            # reshape back to 3D heatmap
            heatmap_3d_flipped_back = heatmap_3d_flipped_back.reshape(
                N, K, D, H, W)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                heatmap_3d_flipped_back[...,
                                        1:] = heatmap_3d_flipped_back[..., :-1]
            output[0] = heatmap_3d_flipped_back

            # flip relative hand root depth
            output[1] = -output[1].detach().cpu().numpy()

            # flip hand type
            hand_type = output[2].detach().cpu().numpy()
            hand_type_flipped_back = hand_type.copy()
            hand_type_flipped_back[:, 0] = hand_type[:, 1]
            hand_type_flipped_back[:, 1] = hand_type[:, 0]
            output[2] = hand_type_flipped_back
        else:
            output = [out.detach().cpu().numpy() for out in output]

        return output

    def decode(self, img_metas, output, **kwargs):
        """Decode hand keypoint, relative root depth and hand type.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint
                 3D heatmap
                - "root_depth_bound": depth bound of relative root depth
                 1D heatmap


            output (list[np.ndarray]): model predicted 3D heatmaps, relative
            root depth and hand type.
        """

        batch_size = len(img_metas)
        heatmap3d_depth_bound = np.ones(batch_size, dtype=np.float32)
        root_depth_bound = np.ones(batch_size, dtype=np.float32)
        for i in range(batch_size):
            heatmap3d_depth_bound[i] = img_metas[i]['heatmap3d_depth_bound']
            root_depth_bound[i] = img_metas[i]['root_depth_bound']

        # decode 3D heatmaps of hand keypoints
        result = self.right_hand_head.decode(img_metas, output[0], **kwargs)
        keypoints_3d = result['preds']
        # transform keypoint depth to camera space
        keypoints_3d[:, :, 2] = \
            (keypoints_3d[:, :, 2] / self.right_hand_head.depth_size - 0.5) \
            * heatmap3d_depth_bound[:, np.newaxis]
        keypoints_3d = keypoints_3d[:, :, :3]
        result['preds'] = keypoints_3d

        # decode relative hand root depth
        result_root = self.root_head.decode(img_metas, output[1], **kwargs)
        # transform relative root depth to camera space
        result['rel_root_depth'] = \
            (result_root['values'] / self.root_head.heatmap_size - 0.5) \
            * root_depth_bound

        # decode hand type
        result_hand_type = self.hand_type_head.decode(img_metas, output[2],
                                                      **kwargs)
        result['hand_type'] = result_hand_type['labels']
        return result


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_channels, out_channels, adjmat, bias=True):
        super().__init__()
        # self.adjmat = adjmat
        self.register_buffer('adjmat', adjmat)
        self.proj_layer = torch.nn.Conv1d(
            in_channels, out_channels, 1, bias=bias)

    # X in shape
    def forward(self, x):
        x = self.proj_layer(x)
        x = torch.matmul(x, self.adjmat)
        return x


class GraphBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, adjmat, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = GraphConvolution(planes, planes, adjmat, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, GraphConvolution):
                normal_init(m.proj_layer, std=0.001, bias=0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GCNRefineNet(nn.Module):

    def __init__(self, in_channels, planes, adjmat, num_blocks):
        super().__init__()
        gcn = []
        gcn.append(nn.Conv1d(in_channels, planes * 4, kernel_size=1))
        for _ in range(num_blocks):
            gcn.append(GraphBottleneck(planes * 4, planes, adjmat))
        gcn.append(nn.Conv1d(planes * 4, in_channels, kernel_size=1))
        self.gcn = nn.Sequential(*gcn)

        self.conv = Bottleneck(
            in_channels=in_channels * 2 + 1,
            out_channels=in_channels,
            downsample=nn.Conv2d(
                in_channels * 2 + 1, in_channels, kernel_size=1))

    def init_weights(self):
        for m in self.gcn.modules():
            if isinstance(m, nn.Conv1d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, GraphBottleneck):
                m.init_weights()

        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def extract_feature(self, feature_map, heatmap):
        B, C, H, W = feature_map.shape
        K = heatmap.shape[1]
        x = torch.matmul(
            feature_map.reshape(B, C, H * W),
            heatmap.reshape(B, K, H * W).permute(0, 2, 1))
        return x

    def distribute_feature(self, x, heatmap):
        B, K, H, W = heatmap.shape
        _, C, _ = x.shape
        heatmap = heatmap / heatmap.sum(dim=1, keepdim=True)
        y = torch.matmul(x, heatmap.reshape(B, K, H * W))
        return y.reshape(B, C, H, W)

    def forward(self, feature_map, heatmap):
        B, C, H, W = feature_map.shape
        K = heatmap.shape[1]
        heatmap = F.interpolate(heatmap, [H, W], mode='bilinear')
        heatmap = F.softmax(
            heatmap.reshape(B, K, -1), dim=-1).reshape((B, K, H, W))

        x = self.extract_feature(feature_map, heatmap)
        x = self.gcn(x)
        x = self.distribute_feature(x, heatmap)
        x = torch.cat([feature_map, x,
                       heatmap.sum(dim=1, keepdim=True)],
                      dim=1)
        x = self.conv(x)
        x = x + feature_map
        return x


@HEADS.register_module()
class GCNInterhand3DHead(Interhand3DHead):
    """Interhand 3D head of paper ref: Gyeongsik Moon. "InterHand2.6M: A
    Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single
    RGB Image".

    Args:
        keypoint_head_cfg (dict): Configs of Heatmap3DHead for hand
        keypoints estimation.
        root_head_cfg (dict): Configs of Heatmap1DHead for relative
        hand root depth estimation.
        hand_type_head_cfg (dict): Configs of MultilabelClassificationHead
        for hand type classification.
    """

    def __init__(self,
                 keypoint_head_cfg,
                 root_head_cfg,
                 hand_type_head_cfg,
                 refine_net_cfg,
                 keypoint2d_head_cfg,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(keypoint_head_cfg, root_head_cfg, hand_type_head_cfg,
                         train_cfg, test_cfg)

        adjmat = np.load(refine_net_cfg.pop('adjmat_file'))
        adjmat = torch.FloatTensor(adjmat)
        # normalize for each column
        adjmat = adjmat / adjmat.sum(dim=0, keepdim=True)
        refine_net_cfg['adjmat'] = adjmat
        self.refine_net = GCNRefineNet(**refine_net_cfg)
        self.hand2d_head = TopDownSimpleHead(**keypoint2d_head_cfg)

    def init_weights(self):
        self.left_hand_head.init_weights()
        self.right_hand_head.init_weights()
        self.root_head.init_weights()
        self.hand_type_head.init_weights()
        self.refine_net.init_weights()
        self.hand2d_head.init_weights()

    def forward(self, x):
        """Forward function."""
        heatmap = self.hand2d_head(x)
        x = self.refine_net(x, heatmap)
        output = super().forward(x)
        output.append(heatmap)
        return output

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
            multiple heads.
        """
        losses = super().get_loss(output, target, target_weight)
        losses['hand2d_loss'] = self.hand2d_head.get_loss(
            output[3], target[3], target_weight[3])['mse_loss']
        return losses
