import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8 mg
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 如果最小值未设置，就使用除数作为最小值
    if min_value is None:
        minValue = divisor
    # 调整v值使其加上一半的除数后能够被除数整除，同时不小于最小值
    new_v = max(minValue, int(v + divisor / 2) // divisor * divisor)
    # 确保调整后的值不比原始值小10%以上
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# 定义一个硬Sigmoid函数，用于激活
def hard_sigmoid(x, inplace: bool = False):
    # 如果就地操作，则更新x的值
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


# 定义Squeeze-and-Excitation模块
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.SiLU, gate_fn=hard_sigmoid, divisor=4):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn  # 定义激活函数
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)  # 计算减少的通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化层
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)  # 降维卷积
        self.act1 = act_layer(inplace=True)  # 激活函数
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)  # 升维卷积

    def forward(self, x):
        x_se = self.avg_pool(x)  # 对输入进行平均池化
        x_se = self.conv_reduce(x_se)  # 降维
        x_se = self.act1(x_se)  # 激活
        x_se = self.conv_expand(x_se)  # 升维
        x = x * self.gate_fn(x_se)  # 通过激活函数调节特征重要性，并与原始输入相乘
        return x


# 定义一个包含卷积、批归一化和激活层的组合模块
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.SiLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)  # 卷积层
        self.bn1 = nn.BatchNorm2d(out_chs)  # 批归一化层
        self.act1 = act_layer(inplace=True)  # 激活层

    def forward(self, x):
        x = self.conv(x)  # 卷积操作
        x = self.bn1(x)  # 批归一化
        x = self.act1(x)  # 激活函数
        return x


class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup  # 初始化通道数
        new_channels = oup  # 新的通道数
        self.deploy = deploy  # 是否部署模式

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),  # 基础卷积层
            nn.BatchNorm2d(init_channels),  # 批归一化
            nn.SiLU(inplace=True) if relu else nn.Sequential(),  # 激活函数
        )
        fusion_conv = []  # 融合卷积列表
        fusion_bn = []  # 融合批彅一化列表
        if not deploy and reparam_bn:  # 如果不是部署模式并且重参数化批归一化
            fusion_conv.append(nn.Identity())  # 添加恒等层
            fusion_bn.append(nn.BatchNorm2d(init_channels))  # 添加批归一化
        if not deploy and reparam_identity:  # 如果不是部署模式并且重参数化恒等层
            fusion_conv.append(nn.Identity())  # 添加恒等层
            fusion_bn.append(nn.Identity())  # 添加恒等层

        self.fusion_conv = nn.Sequential(*fusion_conv)  # 定义融合卷积模块
        self.fusion_bn = nn.Sequential(*fusion_bn)  # 定义融合批归一化模块

        # 廉价操作
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=deploy),  # 深度卷积
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),  # 批归一化
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),  # 激活函数
        )
        if deploy:  # 如果是部署模式
            self.cheap_operation = self.cheap_operation[0]  # 只保留卷积操作
        if relu:  # 如果使用ReLU激活
            self.relu = nn.SiLU(inplace=False)  # 定义激活函数
        else:
            self.relu = nn.Sequential()  # 不使用激活函数

    def forward(self, x):
        x1 = self.primary_conv(x)  # mg
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class C2f_repghost(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  #
        self.m = nn.ModuleList(RepGhostBottleneck(self.c, self.c, self.c, dw_kernel_size=((3), (3))) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))