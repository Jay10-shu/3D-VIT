import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet3D, self).__init__()
        self.down1 = DoubleConv3D(in_channels, 64)
        self.down2 = DoubleConv3D(64, 128)
        self.down3 = DoubleConv3D(128, 256)
        self.down4 = DoubleConv3D(256, 512)
        self.pool = nn.MaxPool3d(2)
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose3d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.pool(x3)
        x5 = self.down3(x4)
        x6 = self.pool(x5)
        x7 = self.down4(x6)
        x8 = self.up1(x7)
        x9 = self.up2(x8)
        x10 = self.up3(x9)
        x11 = self.up4(x10)
        return x11



model = UNet3D(in_channels=1, num_classes=2).to("cuda:1")  # 输入通道数为1，输出类别数为2
tensor = torch.rand((1,1,64,64,64)).to("cuda:1")
print(model(tensor).shape)
