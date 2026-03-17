import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left

        return activation

    return get_activation(l, r)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.uniform_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001, affine=False)

    def forward(self, x):
        return self.w_0 * self.bn(x)


class lrelu(nn.Module):
    def __init__(self):
        super(lrelu, self).__init__()

    def forward(self, x):
        return torch.max(x * 0.2, x)


def build_net(norm=AdaptiveNorm, layer=5, width=32):
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
    ]
    for l in range(1, layer):
        layers += [nn.Conv2d(width, width, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False),
                   norm(width),
                   lrelu(),
                   ]
    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
        nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init)
    return net