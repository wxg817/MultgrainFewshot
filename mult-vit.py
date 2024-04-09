import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()

        # 确保三个分支的输出通道数一致
        self.conv1a = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, bias=False)
        self.bn1a = nn.BatchNorm2d(planes)
        self.conv1b = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out1 = F.leaky_relu(self.bn1a(self.conv1a(x)), negative_slope=0.1)
        out2 = F.leaky_relu(self.bn1b(self.conv1b(x)), negative_slope=0.1)
        out3 = F.leaky_relu(self.bn2(self.conv2(out2)), negative_slope=0.1)
        out = out1 + out2 + out3  # 将三个分支的特征相加
        out += self.shortcut(x)
        return out

class Res2Net12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(Res2Net12, self).__init__()        
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))        
        self.layers = nn.Sequential(*layers)
        self.linear = linear(10 * feature_maps, num_classes)
        self.rotations = rotations
        self.linear_rot = linear(10 * feature_maps, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup=None, lam=-1):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope=0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features
