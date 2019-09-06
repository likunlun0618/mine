import os
import torch
import torch.nn as nn


class Residual(nn.Module):

    def __init__(self, inp, out):
        super(Residual, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.ReLU(),
            nn.Conv2d(inp, out//2, 1, 1, 0),
            nn.BatchNorm2d(out//2),
            nn.ReLU(),
            nn.Conv2d(out//2, out//2, 3, 1, 1),
            nn.BatchNorm2d(out//2),
            nn.ReLU(),
            nn.Conv2d(out//2, out, 1, 1, 0)
        )
        if inp == out:
            self.skip_layer = None
        else:
            self.skip_layer = nn.Conv2d(inp, out, 1, 1, 0)

    def forward(self, x):
        if self.skip_layer is None:
            return self.conv_block(x) + x
        else:
            return self.conv_block(x) + self.skip_layer(x)


class Hourglass(nn.Module):

    def __init__(self, f, n):
        super(Hourglass, self).__init__()
        self.upper_branch = Residual(f, f)
        self.lower_branch = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            Residual(f, f),
            Residual(f, f) if n == 1 else Hourglass(f, n-1),
            Residual(f, f),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.upper_branch(x) + self.lower_branch(x)


class StackedHourglass(nn.Module):

    def __init__(self, out_channels, f, n, stacks):
        super(StackedHourglass, self).__init__()
        self.stacks = stacks
        self.pre_module = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Residual(64, f//2),
            nn.MaxPool2d(2, stride=2),
            Residual(f//2, f//2),
            Residual(f//2, f)
        )
        self.hg = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        self.middle_branch = nn.ModuleList()
        self.output_branch = nn.ModuleList()
        for i in range(stacks):
            self.hg.append(nn.Sequential(
                Hourglass(f, n),
                Residual(f, f),
                nn.Conv2d(f, f, 1, 1, 0),
                nn.BatchNorm2d(f),
                nn.ReLU()
            ))
            self.out_layer.append(nn.Conv2d(f, out_channels, 1, 1, 0))
            if i < stacks-1:
                self.middle_branch.append(nn.Conv2d(f, f, 1, 1, 0))
                self.output_branch.append(nn.Conv2d(out_channels, f, 1, 1, 0))

    def argmax(self, hm):
        w = hm.size(3)
        scores, index = torch.max(hm.view(hm.size(0), hm.size(1), -1), dim=2)
        y = index // w
        x = index - y*w

        # *4是因为heatmap的尺寸是原图的1/4
        coordinates = torch.cat([x.unsqueeze(2), y.unsqueeze(2)], 2) * 4

        return coordinates, scores

    def forward(self, x, targets=None):
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')

        if x.size(2) != 256 or x.size(3) != 256:
            raise ValueError('Inputs should be 256 x 256')

        y = []
        x = self.pre_module(x)
        for i in range(self.stacks):
            middle = self.hg[i](x)
            output = self.out_layer[i](middle)
            y.append(output)
            if i < self.stacks-1:
                x = x + self.middle_branch[i](middle) + self.output_branch[i](output)

        # 测试模式时，返回最后一个stack的heatmap
        # 训练模式时，返回loss
        if not self.training:
            return (y[-1], *self.argmax(y[-1]))
        else:
            loss = 0.
            for i in range(len(y)):
                loss = loss + (y[i] - targets).pow(2).mean()
            return loss


def get_root():
    '''
    获取当前package的目录，用于寻找.cache下的模型
    '''
    root, _ = os.path.split(__file__)
    return root


def hg4(pretrained=False):
    model = StackedHourglass(16, 256, 4, 4)
    if pretrained:
        model.load_state_dict(torch.load(get_root() + '/.cache/checkpoints/hg4.pth'))
    return model


def hg8(pretrained=False):
    model = StackedHourglass(16, 256, 4, 8)
    if pretrained:
        model.load_state_dict(torch.load(get_root() + '/.cache/checkpoints/hg8.pth'))
    return model
