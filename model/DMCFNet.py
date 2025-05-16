import torch
import torch.nn as nn
from thop import profile

class BasicConv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out) * x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        weight = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(weight)) * x
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
class CASF(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels):
        super(CASF, self).__init__()
        self.low_branch = None
        self.low_branch = (
            nn.Sequential(
                nn.Conv2d(low_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ) if low_channels else None
        )
        self.high_branch = (
            nn.Sequential(
                nn.Conv2d(high_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ) if high_channels else None
        )
        self.low_cross_scale_conv = BasicConv_Block(mid_channels * 2, mid_channels, kernel_size=1, stride=1, padding=0)
        self.high_cross_scale_conv = BasicConv_Block(mid_channels * 2, mid_channels, kernel_size=1, stride=1, padding=0)
        self.final_cross_scale_conv = BasicConv_Block(mid_channels * 3, mid_channels, kernel_size=1, stride=1, padding=0)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(mid_channels * 2)
    def forward(self, low_x, x, high_x):
        low_features = self.low_branch(low_x) if self.low_branch else None
        high_features = self.high_branch(high_x) if self.high_branch else None
        if low_x is None:
            combined_features = torch.cat((x, high_features), dim=1)
            combined_features = self.ca(combined_features)
            combined_features = self.high_cross_scale_conv(combined_features)
        elif high_x is None:
            combined_features = torch.cat((low_features, x), dim=1)
            combined_features = self.sa(combined_features)
            combined_features = self.low_cross_scale_conv(combined_features)
        else:
            combined_features = torch.cat((low_features, x, high_features), dim=1)
            combined_features = self.final_cross_scale_conv(combined_features)
        return combined_features
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = BasicConv_Block(in_channels, out_channels)
        self.conv2 = BasicConv_Block(out_channels, out_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
class MSFA(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, size=0, is_max_size = False):
        super().__init__()
        self.conv1x1 = BasicConv_Block(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        if size in [512, 256]:
            dilations1, dilations2 = (1, 2, 4), (1, 3, 5)

        elif size in [128, 64]:
            dilations1, dilations2 = (1, 2, 3), (1, 3, 4)
        else:
            dilations1, dilations2 = (1, 2, 3), (1, 3, 5)

        self.attention = DWConvBranch(mid_channels)
        self.CBAM = CBAM(mid_channels)
        self.convout = BasicConv_Block(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.branch1 = self._create_dilated_branch(mid_channels, dilations1)
        self.branch2 = self._create_dilated_branch(mid_channels, dilations2)

    def _create_dilated_branch(self, mid_channels, dilations):
        layers = []
        for dilation in dilations:
            layers.append(BasicConv_Block(mid_channels, mid_channels, kernel_size=3, stride=1, dilation=dilation,
                                          padding=dilation, groups=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        xin = self.conv1x1(x)
        x_branch1 = self.branch1(xin)
        x_branch2 = self.branch2(xin)

        x_att = self.attention(xin)
        sumx = x_branch1 + x_branch2 + xin + x_att
        sumx = self.CBAM(sumx)
        x_out = self.convout(sumx)
        return x_out
class DWConvBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.conv1x1 = BasicConv_Block(channels, channels, 1, 1, 1, 0)
        self.group_conv = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=k, stride=1, padding=k // 2, groups=channels)
            for k in [3, 5, 7]
        ])
        self.sigmoid = nn.Sigmoid()
        self.weights = nn.Parameter(torch.ones(len(self.group_conv)))
        self.direct_conv = BasicConv_Block(channels, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        pooled_x = self.pool(x)
        depthwise_results = [conv(pooled_x) for conv in self.group_conv]
        normalized_weights = torch.softmax(self.weights, dim=0)
        # print("Weights:", self.weights.detach().cpu().numpy())
        combined_features = sum(w * feat for w, feat in zip(normalized_weights, depthwise_results))
        combined_features = self.conv1x1(combined_features)
        atten = self.sigmoid(combined_features)
        x1 = x * atten
        x2 = self.direct_conv(x)
        return x1 + x2
class FRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = BasicConv_Block(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = BasicConv_Block(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = BasicConv_Block(in_channels, out_channels // 4, kernel_size=5, stride=1, padding=2)
        self.conv7x7 = BasicConv_Block(in_channels, out_channels // 4, kernel_size=7, stride=1, padding=3)
        self.channel_fusion = BasicConv_Block(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.conv7x7(x)

        multi_scale_features = torch.cat([x1, x2, x3, x4], dim=1)

        refined_features = self.channel_fusion(multi_scale_features)

        return refined_features
class DMCFNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DMCFNet, self).__init__()

        self.enc1 = MSFA(in_channels, 64, 24, 512, is_max_size=True)
        self.enc2 = MSFA(64, 128, 24, 256)
        self.enc3 = MSFA(128, 256, 24, 128)
        self.enc4 = MSFA(256, 512, 24, 64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.FeatureFusion1 = CASF(None, 64, 128)
        self.FeatureFusion2 = CASF(64, 128, 256)
        self.FeatureFusion3 = CASF(128, 256, 512)
        self.FeatureFusion4 = CASF(256, 512, None)

        self.bottom = FRB(512, 1024)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU())
        self.dec1 = MSFA(1024, 512, 24, 64)
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
        self.dec2 = MSFA(512, 256, 24, 128)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU())
        self.dec3 = MSFA(256, 128, 24, 256)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())
        self.dec4 = MSFA(128, 64, 24, 512)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)  # 64 512 512
        e2 = self.enc2(self.pool(e1))  # 128 256 256
        e3 = self.enc3(self.pool(e2))  # 256 128 128
        e4 = self.enc4(self.pool(e3))  # 512 64 64

        e1 = self.FeatureFusion1(None, e1, e2)
        e2 = self.FeatureFusion2(e1, e2, e3)
        e3 = self.FeatureFusion3(e2, e3, e4)
        e4 = self.FeatureFusion4(e3, e4, None)
        b = self.bottom(self.pool(e4))  # 512 32 32

        d1 = self.up1(b)
        d1 = torch.cat((d1, e4), dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.dec4(d4)

        output = self.out_conv(d4)
        return output

if __name__ == "__main__":
    model = DMCFNet(in_channels=2, out_channels=1)
    x = torch.randn(1, 2, 512, 512)
    out = model(x)
    flops, params = profile(model, (x,))
    flops_in_gflops = flops / 1e9
    params_in_m = params / 1e6
    print("Size:", out.shape)
    print(model)
    print(f"FLOPs: {flops_in_gflops:.2f} GFLOPS", f"Params: {params_in_m:.2f} MParams")