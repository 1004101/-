from src import fcn_resnet50
import numpy as np
import torch

if __name__ == "__main__":
    # model = fcn_resnet50(aux=True)
    # # for n in model.modules():
    # #     print(n)
    # for name, module in model.named_children():
    #     print(name, module)
    weights_dict = torch.load("C:/Users/JunhaoZhan/Desktop/论文/图像分割源码/fcn_ResNext50/fcn_resnet50_coco.pth", map_location='cpu')
    for k in list(weights_dict.keys()):
        print(k)