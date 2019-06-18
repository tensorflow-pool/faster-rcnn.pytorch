import torch

from model.roi_layers import nms

a = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4]]).cuda()
b = torch.Tensor([0.8, 0.9]).cuda()
print(nms(a, b, 0.1))
