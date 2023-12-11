import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        # Can use for InstancenNorm2d
        use_bias = False

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, True)

        self.conv2 = nn.Conv2d(
            ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        self.norm2 = norm_layer(ndf * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(
            ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        self.norm3 = norm_layer(ndf * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, True)

        self.conv4 = nn.Conv2d(
            ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=use_bias
        )
        self.norm4 = norm_layer(ndf * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, True)

        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, input):
        x = self.lrelu1(self.conv1(input))
        x = self.lrelu2(self.norm2(self.conv2(x)))
        x = self.lrelu3(self.norm3(self.conv3(x)))
        x = self.lrelu4(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        return x
