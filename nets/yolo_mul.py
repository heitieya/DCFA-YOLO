import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors
from nets.repghost import C2f_repghost



class SPPF_CBAM(nn.Module):
    # SPP structure, max pooling with kernel sizes of 5, 9, and 13.
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cbam1 = CBAM(c_, c_)
        self.cbam2 = CBAM(c_, c_)
        self.cbam3 = CBAM(c_, c_)
        self.cbam4 = CBAM(c_, c_)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cbam1(x)
        y1 = self.m(x)
        y1 =  self.cbam2(y1)
        y2 = self.m(y1)
        y2 =  self.cbam3(y2)
        y3 = self.m(y2)
        y3 = self.cbam4(y3)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# Suitable for three
class Concat_BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(Concat_BiFPN, self).__init__()
        self.d = dimension
        # Initialize weight parameters, now there are three elements, corresponding to the weights of three input feature maps
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # Normalize weights
        w = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        # Assume x is a list or tuple containing three feature maps
        # Apply weights and select the first two feature maps for weighted sum (since there are only three weights, only the first two are weighted)
        weighted_x = [w[0] * x[0], w[1] * x[1], w[2] * x[2]]
        # Concatenate along the specified dimension
        return torch.cat(weighted_x, self.d)


# 1. CBAM attention mechanism
# By introducing the attention mechanism, we can adaptively adjust the weights of each modal feature, enhancing the contribution of specific modalities in different scenarios.
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Use 1x1 convolution instead of fully connected layer
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

class Conv_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # copy input
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,
                      groups=branch_features),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # Automatically pad the input feature layer according to the Same principle
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    # SiLU activation function
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # Standard convolution + normalization + activation function
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck structure, residual structure
    # c1 is the input channel number, c2 is the output channel number
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    # CSPNet structure, large residual structure
    # c1 is the input channel number, c2 is the output channel number
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # Perform one convolution, then split into two parts, each part has c channels
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # Each residual structure is retained and then stacked together, dense residual
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    # SPP structure, max pooling with kernel sizes of 5, 9, and 13.
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()

        self.stem = Conv_maxpool(3, base_channels)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),

            ShuffleNetV2(base_channels * 2, base_channels * 2, stride=1),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            ShuffleNetV2(base_channels * 4, base_channels * 4, stride=1),
        )

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            ShuffleNetV2(base_channels * 8, base_channels * 8, stride=1),
        )


        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            ShuffleNetV2(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), stride=1),
    
            SPPF_CBAM(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )


        if pretrained:
            url = {
                "n": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)

        x = self.dark2(x)

        x = self.dark3(x)
        feat1 = x

        x = self.dark4(x)
        feat2 = x

        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3



class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # -----------------------------------------------#
        #   创建两个模态的主干网络（RGB 和 NIR 模态）
        # -----------------------------------------------#
        self.backbone_rgb = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.backbone_nir = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.bi_fpn = Concat_BiFPN(dimension=1)

        self.cbam_rgb_feat1 = CBAM(base_channels * 4)   # CBAM for feat1_rgb
        self.cbam_nir_feat1 = CBAM(base_channels * 4)   # CBAM for feat1_nir

        self.cbam_rgb_feat2 = CBAM(base_channels * 8)   # CBAM for feat2_rgb
        self.cbam_nir_feat2 = CBAM(base_channels * 8)   # CBAM for feat2_nir

        self.cbam_rgb_feat3 = CBAM(int(base_channels * 16 * deep_mul), ratio=8, kernel_size=7)  # CBAM for feat3_rgb
        self.cbam_nir_feat3 = CBAM(int(base_channels * 16 * deep_mul), ratio=8, kernel_size=7)  # CBAM for feat3_nir



        # ------------------------ Strengthen feature extraction network ------------------------ #
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")


        self.conv3_for_upsample1 = C2f_repghost(int(base_channels * 16 * deep_mul) + base_channels * 8 + 128, base_channels * 8,
                                       base_depth, shortcut=False)
  
        self.conv3_for_upsample2 = C2f_repghost(base_channels * 8 + base_channels * 4 + 64, base_channels * 4, base_depth,
                                       shortcut=False)


        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)

        self.conv3_for_downsample1 = C2f_repghost(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)


        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)

        self.conv3_for_downsample2 = C2f_repghost(int(base_channels * 16 * deep_mul) + base_channels * 8 + 256,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)

        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        self.stride = torch.tensor([256 / x.shape[-2] for x in self.backbone_rgb.forward(torch.zeros(1, 3, 256, 256))])
        self.reg_max = 16  # DFL channels
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)

        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, rgb, nir):
    
        feat1_rgb, feat2_rgb, feat3_rgb = self.backbone_rgb.forward(rgb)
        feat1_nir, feat2_nir, feat3_nir = self.backbone_nir.forward(nir)
    

        feat1_rgb = self.cbam_rgb_feat1(feat1_rgb)


        feat1_nir = self.cbam_nir_feat1(feat1_nir)
        feat2_rgb = self.cbam_rgb_feat2(feat2_rgb)


        feat2_nir = self.cbam_nir_feat2(feat2_nir)

        feat3_rgb = self.cbam_rgb_feat3(feat3_rgb)


        feat3_nir = self.cbam_nir_feat3(feat3_nir)





        feat3 = feat3_rgb + feat3_nir


        # ------------------------ Strengthen feature extraction network ------------------------ #

        P5_upsample = F.interpolate(feat3, size=(40, 40), mode='bilinear', align_corners=True)
 
        P4 =  self.bi_fpn([P5_upsample, feat2_rgb, feat2_nir])

        P4 = self.conv3_for_upsample1(P4) 


        P4_upsample = F.interpolate(P4, size=(80, 80), mode='bilinear', align_corners=True)

        P3 =  self.bi_fpn([P4_upsample, feat1_rgb, feat1_nir])
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 =  self.bi_fpn([P4_downsample, feat3_rgb, feat3_nir])
        P5 = self.conv3_for_downsample2(P5)


        # ------------------------ Strengthen feature extraction network ------------------------ #

        shape = P3.shape

        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


