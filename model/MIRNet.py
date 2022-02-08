import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(Channel_Attention, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 =nn.Conv2d(in_channels, in_channels//reduction, 1, 1, 0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels//reduction, in_channels, 1, 1, 0)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        gap = self.gap(x)
        x_out = self.conv1(gap)
        x_out = self.relu1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.sigmoid2(x_out)
        x_out = x_out * x        
        return x_out


class Dual_Attention_Unit(nn.Module):
    def __init__(self, in_channels, kernel_size=7, reduction=8):
        super(Dual_Attention_Unit, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(in_channels, in_channels, 3, padding=1))
        self.sa = Spatial_Attention(kernel_size)
        self.ca = Channel_Attention(in_channels, reduction)

        self.conv = nn.Conv2d(in_channels*2, in_channels, 1)

    def forward(self, x):
        x1 = self.stem(x)
        x_sa = self.sa(x1)
        x_ca = self.ca(x1)        
        x_cat = torch.cat([x_sa, x_ca], dim=1)
        x_cat = self.conv(x_cat)
        x_out = x_cat + x
        return x_out


class Selective_Kernel_Feature_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(Selective_Kernel_Feature_Fusion, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels//reduction, 1)
        self.prelu = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channels//reduction, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels//reduction, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels//reduction, out_channels, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):  
        # Fuse      
        x_sum = x1 + x2 + x3
        x_gap = self.gap(x_sum)
        x_gap = self.conv(x_gap)
        x_gap = self.prelu(x_gap)
                
        v1 = self.conv1(x_gap)
        v2 = self.conv2(x_gap)
        v3 = self.conv3(x_gap)

        attention_v = torch.stack((v1, v2, v3), dim=1)
        attention_v = self.softmax(attention_v)

        x = torch.stack((x1, x2, x3), dim=1)

        # Select
        x_out = torch.sum(x * attention_v, dim=1)

        return x_out


class Down_Sampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Down_Sampling, self).__init__()

        self.path1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                                nn.PReLU(),
                                nn.MaxPool2d(scale_factor),
                                nn.Conv2d(in_channels, out_channels, 1))
        self.path2 = nn.Sequential(nn.MaxPool2d(scale_factor),
                                nn.Conv2d(in_channels, out_channels, 1))
    def forward(self, x):        
        x1 = self.path1(x)
        x2 = self.path2(x)
        return x1 + x2


class Up_Sampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(Up_Sampling, self).__init__()

        self.path1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                                nn.PReLU(),
                                nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
                                nn.Conv2d(in_channels, out_channels, 1))
        
        self.path2 = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
                                nn.Conv2d(in_channels, out_channels, 1))
    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        return x1 + x2


class Multiscale_Residual_Block(nn.Module):
    def __init__(self, num_features):
        super(Multiscale_Residual_Block, self).__init__()      

        self.down_l1_l2 = Down_Sampling(in_channels=num_features, out_channels=num_features*2, scale_factor=2)                
        self.down_l2_l3 = Down_Sampling(in_channels=num_features*2, out_channels=num_features*4, scale_factor=2)
        self.down_l1_l3 = Down_Sampling(in_channels=num_features, out_channels=num_features*4, scale_factor=4)


        self.dau1_1 = Dual_Attention_Unit(num_features)
        self.dau1_2 = Dual_Attention_Unit(num_features*2)
        self.dau1_3 = Dual_Attention_Unit(num_features*4)

        # l2_l1 is scale2 -> scale1 upsampling
        self.up_l2_l1 = Up_Sampling(in_channels=num_features*2, out_channels=num_features, scale_factor=2)        
        self.up_l3_l2 = Up_Sampling(in_channels=num_features*4, out_channels=num_features*2, scale_factor=2)
        self.up_l3_l1 = Up_Sampling(in_channels=num_features*4, out_channels=num_features, scale_factor=4)


        self.skff1_1 = Selective_Kernel_Feature_Fusion(num_features, out_channels=num_features)
        self.skff1_2 = Selective_Kernel_Feature_Fusion(num_features*2, out_channels=num_features*2)
        self.skff1_3 = Selective_Kernel_Feature_Fusion(num_features*4, out_channels=num_features*4)

        self.dau2_1 = Dual_Attention_Unit(num_features)
        self.dau2_2 = Dual_Attention_Unit(num_features*2)
        self.dau2_3 = Dual_Attention_Unit(num_features*4)

        self.skff2_1 = Selective_Kernel_Feature_Fusion(num_features, out_channels=num_features)
        
        self.conv = nn.Conv2d(num_features, num_features, 3, padding=1)
       
    def forward(self, x):
        x_down_2x = self.down_l1_l2(x)
        x_down_4x = self.down_l1_l3(x)        

        x_dau1_1 = self.dau1_1(x) # (32, 512, 512)
        x_dau1_2 = self.dau1_2(x_down_2x) # (64, 256, 256)        
        x_dau1_3 = self.dau1_3(x_down_4x) # (128, 128, 128)

        
        x_skff1 = self.skff1_1(x_dau1_1, 
                             self.up_l2_l1(x_dau1_2), 
                             self.up_l3_l1(x_dau1_3))        

        x_skff2 = self.skff1_2(self.down_l1_l2(x_dau1_1), 
                             x_dau1_2,
                             self.up_l3_l2(x_dau1_3))    
        
        x_skff3 = self.skff1_3(self.down_l1_l3(x_dau1_1), 
                                self.down_l2_l3(x_dau1_2),
                                x_dau1_3)


        x_dau2_1 = self.dau2_1(x_skff1)
        x_dau2_2 = self.dau2_2(x_skff2)
        x_dau2_3 = self.dau2_3(x_skff3)

        x_skff2_1 = self.skff2_1(x_dau2_1,
                                self.up_l2_l1(x_dau2_2),
                                self.up_l3_l1(x_dau2_3))
        
        x_out = self.conv(x_skff2_1)
        x_out = x_out + x
        return x_out


class Recursive_Residual_Group(nn.Module):
    def __init__(self, num_features):
        super(Recursive_Residual_Group, self).__init__()

        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.mrb1 = Multiscale_Residual_Block(num_features)
        self.mrb2 = Multiscale_Residual_Block(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
       
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.mrb1(x1)
        x2 = self.mrb2(x1)
        x2 = self.conv2(x2)        
        x_out = x + x2        
        return x_out


class MIRNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features):
        super(MIRNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.rrg1 = Recursive_Residual_Group(num_features)
        self.rrg2 = Recursive_Residual_Group(num_features)
        self.rrg3 = Recursive_Residual_Group(num_features)
        self.conv2 = nn.Conv2d(num_features, out_channels, 3, padding=1)

        self.init_weights()

    def forward(self, x):
        x0 = self.conv1(x)        
        x1 = self.rrg1(x0)
        x2 = self.rrg2(x1)
        x3 = self.rrg3(x2)
        xd = self.conv2(x3)                
        x_out = xd + x
        return x_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
