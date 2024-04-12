from src import fcn_resnet50
import numpy as np
import torch
import torchvision
from thop import profile

if __name__ == "__main__":
    model = fcn_resnet50(aux=True)
    # Total_params = 0
    # Trainable_params = 0
    # NonTrainable_params = 0
    # for param in model.parameters():
    #     #print(param.size())
    #     mulValue = np.prod(param.size())
    #     Total_params += mulValue
    #     if param.requires_grad:
    #         Trainable_params += mulValue
    #     else:
    #         NonTrainable_params += mulValue
    # print(f'Total params:{Total_params}')
    # print(f'Trainable params: {Trainable_params}')
    # print(f'Non-trainable params: {NonTrainable_params}')
    # for name, module in model.named_children():
    #     print(name)
    #     print(module)
    # Total
    # params: 35322218
    # Trainable
    # params: 35322218
    # Non - trainable
    # params: 0
    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, (dummy_input,))
    print(flops, params)

