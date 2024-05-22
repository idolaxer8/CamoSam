import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import math
from Resnet import res2net50_v1b_26w_4s
import Data_loading


class Bcnn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0, bias=False):
        super(Bcnn, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)
    

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.f2_cnn = Bcnn(in_channels=256, out_channels=64, kernel_size=1, bias=True) #1x1Conv
        self.f5_cnn = Bcnn(in_channels=2048, out_channels=256, kernel_size=1, bias=True) #1x1Conv
        # 2 - 3*3 Bcnn -> 1*1Conv
        self.concat_block = nn.Sequential(Bcnn(in_channels=256+64, out_channels=256, padding=1), 
                                          Bcnn(in_channels=256, out_channels=256, padding=1),
                                          nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)) 

    def forward(self, f2, f5):
        f2_size = f2.size()[2:]
        f2 = self.f2_cnn(f2)
        f5 = self.f5_cnn(f5)
        f5 = F.interpolate(f5, f2_size, mode='bilinear', align_corners=False) # Upsampling - f5
        x = torch.cat((f2, f5), dim=1) # concatenation
        out = self.concat_block(x)
        
        return out
    

class EFM(nn.Module):
    def __init__(self, channels):
        super(EFM, self).__init__()
        t = int(abs((math.log(channels, 2) + 1) / 2))
        ker_size = t if t % 2 else t + 1
        padd = (ker_size-1) // 2
        self.cnn3x3 = Bcnn(in_channels=channels, out_channels=channels, padding=1)
        self.block = nn.Sequential(nn.AdaptiveAvgPool1d(1), 
                                    nn.Conv1d(1, 1, kernel_size=ker_size, padding=padd, bias=False),
                                    nn.Sigmoid())
        self.cnn1x1 = Bcnn(in_channels=256, out_channels=64, kernel_size=1, bias=True) #1x1Conv

    def forward(self, f_i, f_e):
        if f_i.size() != f_e.size():
            att = F.interpolate(f_e, f_i.size()[2:], mode='bilinear', align_corners=False)
        x = att * f_i + f_i
        x = self.cnn3x3(x)
        out = x * self.block(x)

        return self.cnn1x1(out)
    

class CAM(nn.Module):
    def __init__(self, hchannels, out_channels):
        super(EFM, self).__init__()
        self.cnn1x1 = Bcnn(hchannels+out_channels, out_channels, kernel_size=1, bias=True) #1x1Conv
        self.cnn3x3_d1 = Bcnn(out_channels//4, out_channels//4, 3, dilation=1, padding=1)
        self.cnn3x3_d2 = Bcnn(out_channels//4, out_channels//4, 3, dilation=2, padding=2)
        self.cnn3x3_d3 = Bcnn(out_channels//4, out_channels//4, 3, dilation=3, padding=3)
        self.cnn3x3_d4 = Bcnn(out_channels//4, out_channels//4, 3, dilation=4, padding=4)
        self.cnn1x1_2 = Bcnn(out_channels, out_channels, kernel_size=1, bias=True) #1x1Conv
        self.final_cnn3x3 = Bcnn(out_channels, out_channels, 3)

    def forward(self, lo_f, hi_f): # lo_f - f_i, hi_f - f_i+1
        if lo_f.size()[2:] != hi_f.size()[2:]: 
            hi_f = F.interpolate(hi_f, lo_f.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat((lo_f, hi_f), dim=1)
        x = self.cnn1x1(x)
        xc = torch.chunk(x, 4, dim=1)
        x1 = self.cnn3x3_d1(xc[0]+xc[1])
        x2 = self.cnn3x3_d2(x1+xc[1]+xc[2])
        x3 = self.cnn3x3_d3(x2+xc[2]+xc[3])
        x4 = self.cnn3x3_d4(x3+xc[3])
        x = self.cnn1x1_2(torch.cat(x1,x2,x3,x4), dim=1)
        out = self.final_cnn3x3(x)

        return out


class OurEncoder(nn.Module):
    def __init__(self):
        super(OurEncoder, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)  # Assuming you have defined this ResNet model
        self.EAM = EAM()  # Assuming you have defined this Edge Attention Module
        self.full_features = [256, 512, 1024, 2048]

    def forward(self, x):
        # Forward pass through the ResNet backbone
        f2, f3, f4, f5 = self.resnet(x)

        # Forward pass through the Edge Attention Module
        edge_attention_map = self.EAM(f2, f5)

        # Returning features as a custom object
        class FeatureObject:
            def __init__(self, f2, f3, f4, f5):
                self.f2 = f2
                self.f3 = f3
                self.f4 = f4
                self.f5 = f5

            def __getitem__(self, item):
                if item == 0:
                    return self.f2
                elif item == 1:
                    return self.f3
                elif item == 2:
                    return self.f4
                elif item == 3:
                    return self.f5
                else:
                    raise IndexError("Index out of range")

        features = FeatureObject(f2, f3, f4, f5)

        return features, edge_attention_map


class OurBigEncoder(nn.Module):
    def __init__(self):
        super(OurBigEncoder, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)  # Assuming you have defined this ResNet model
        self.EAM = EAM()  # Assuming you have defined this Edge Attention Module
        self.EFM = EFM(256)  # Adding the Edge Feature Module
        self.CAM = CAM(256, 64)  # Adding the Channel Attention Module
        self.full_features = [256, 512, 1024, 2048]

    def forward(self, x):
        # Forward pass through the ResNet backbone
        f2, f3, f4, f5 = self.resnet(x)

        # Forward pass through the Edge Attention Module
        edge_attention_map = self.EAM(f2, f5)

        # Forward pass through the Edge Feature Module
        ef_map = self.EFM(edge_attention_map, f2)

        # Forward pass through the Channel Attention Module
        cam_out = self.CAM(ef_map, f3)

        # Returning features as a custom object
        class FeatureObject:
            def __init__(self, f2, f3, f4, f5, cam_out):
                self.f2 = f2
                self.f3 = f3
                self.f4 = f4
                self.f5 = f5
                self.cam_out = cam_out

            def __getitem__(self, item):
                if item == 0:
                    return self.f2
                elif item == 1:
                    return self.f3
                elif item == 2:
                    return self.f4
                elif item == 3:
                    return self.f5
                else:
                    raise IndexError("Index out of range")

        features = FeatureObject(f2, f3, f4, f5, cam_out)

        return features, edge_attention_map


class Bgnet(nn.Module):
    def __init__(self):
        super(EFM, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True) #TODO: consider using HarDnet instead

        self.EAM = EAM() 

        self.EFM1 = EFM(256)
        self.EFM2 = EFM(512)
        self.EFM3 = EFM(1024)
        self.EFM4 = EFM(2048)

        self.CAM1 = CAM(128, 64)
        self.CAM2 = CAM(256, 128)
        self.CAM3 = CAM(256, 256)

        self.pred1 = nn.Conv2d(64, 1, 1)
        self.pred2 = nn.Conv2d(128, 1, 1)
        self.pred3 = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        f2, f3, f4, f5 = self.resnet(x)

        fe = self.EAM(f2, f5)
        edge_att = torch.sigmoid(fe)

        x4, x3, x2, x1 = self.EFM4(fe, f5), self.EFM3(fe, f4), self.EFM2(fe, f3), self.EFM1(fe, f2)

        y1 = self.CAM3(x4, x3)
        y2 = self.CAM2(y1, x2)
        y3 = self.CAM1(y2, x1)

        o3 = self.pred3(y1)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.pred2(y2)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.pred1(y3)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe









        


