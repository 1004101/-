from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.RepViT_small import mobile_repvgg_small

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_repvit_small(aux, num_classes=21, pretrain_backbone=False, deploy=False):
    backbone = mobile_repvgg_small(deploy=deploy)

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        # weights_dict = torch.load("F:/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights/best_model.pth", map_location='cpu')
        weights_dict = torch.load(
            "/root/autodl-tmp/RepViT_small/weights/best_model.pth",
            map_location='cpu')
        # for k in list(weights_dict.keys()):
        #     if "layer_4" in k:
        #         del weights_dict[k]
        #     if "layer_5" in k:
        #         del weights_dict[k]
        backbone.load_state_dict(weights_dict, strict=False)

    out_inplanes = 160
    aux_inplanes = 128

    return_layers = {'layer_5': 'out'}
    if aux:
        return_layers['layer_4'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model