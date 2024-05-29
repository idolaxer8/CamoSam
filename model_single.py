from base import *
from Bgnet import OurEncoder, OurBigEncoder

class Decoder(nn.Module):
    def __init__(self, full_features, out):
        super(Decoder, self).__init__()
        # self.up1 = UpBlockSkip(full_features[4] + full_features[3], full_features[3],
        #                        func='relu', drop=0).cuda()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.up3 = UpBlockSkip(full_features[1] + full_features[0], full_features[0],
                               func='relu', drop=0).cuda()
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final = CNNBlock(full_features[0], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        z = self.up3(z, x[0])
        # z = self.up4(z, x[0])
        z = self.Upsample(z)
        out = F.tanh(self.final(z))
        return out



class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        return out




class OurBigModelEmb(nn.Module):
    def __init__(self, args):
        super(OurBigModelEmb, self).__init__()
        self.backbone = OurBigEncoder()
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=256)  # Remove self.backbone from here
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        # Forward pass through the backbone
        features, _ = self.backbone(img)  # Now features is a tuple containing features at different levels

        # Passing features to the decoder
        dense_embeddings = self.decoder(features)  # Pass features directly
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings



class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.norm1 = LayerNorm2d(4)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv2d(16, 256, kernel_size=1)

    def forward(self, mask):
        z = self.conv1(mask)
        z = self.norm1(z)
        z = self.gelu(z)
        z = self.conv2(z)
        z = self.norm2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        return z




if __name__ == "__main__":
    import argparse
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-nP', '--nP', default=10, help='image size', required=False)
    args = vars(parser.parse_args())
