#--------------------------------------------#
#   This part of the code is used to view the network structure
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.yolo_mul import YoloBody

if __name__ == "__main__":
    input_shape1     = [640, 640]
    input_shape2     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 1
    phi             = 's'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(input_shape1, num_classes, phi, False).to(device)
    for i in m.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, 640, 640).to(device)
    flops, params   = profile(m.to(device), (dummy_input, dummy_input), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2 is because profile does not count convolution as two operations
    #   Some papers consider convolution as both multiplication and addition operations. In this case, multiply by 2
    #   Some papers only consider the number of multiplication operations, ignoring addition. In this case, do not multiply by 2
    #   This code chooses to multiply by 2, referencing YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
