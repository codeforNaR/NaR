import torch.nn as nn

from .layers import inverted_residual_sequence, conv2d_bn_relu6

__all__ = ['mobilenet']

import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV2_bad(nn.Module):
    def __init__(self, args):
        super(MobileNetV2_bad, self).__init__()

        s1, s2 = 2, 2
        if args.downsampling == 16:
            s1, s2 = 2, 1
        elif args.downsampling == 8:
            s1, s2 = 1, 1

        # Network is created here, then will be unpacked into nn.sequential
        self.network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': s1},
                                 {'t': 1, 'c': 16, 'n': 1, 's': 1},
                                 {'t': 6, 'c': 24, 'n': 2, 's': s2},
                                 {'t': 6, 'c': 32, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 64, 'n': 4, 's': 2},
                                 {'t': 6, 'c': 96, 'n': 3, 's': 1},
                                 {'t': 6, 'c': 160, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 320, 'n': 1, 's': 1},
                                 {'t': None, 'c': 1280, 'n': 1, 's': 1}]
        self.num_classes = args.num_classes

        ###############################################################################################################

        # Feature Extraction part
        # Layer 0
        self.network = [
            conv2d_bn_relu6(args.num_channels,
                            int(self.network_settings[0]['c'] * args.width_multiplier),
                            args.kernel_size,
                            self.network_settings[0]['s'], args.dropout_prob)]

        # Layers from 1 to 7
        for i in range(1, 8):
            self.network.extend(
                inverted_residual_sequence(
                    int(self.network_settings[i - 1]['c'] * args.width_multiplier),
                    int(self.network_settings[i]['c'] * args.width_multiplier),
                    self.network_settings[i]['n'], self.network_settings[i]['t'],
                    args.kernel_size, self.network_settings[i]['s']))

        # Last layer before flattening
        self.network.append(
            conv2d_bn_relu6(int(self.network_settings[7]['c'] * args.width_multiplier),
                            int(self.network_settings[8]['c'] * args.width_multiplier), 1,
                            self.network_settings[8]['s'],
                            args.dropout_prob))

        ###############################################################################################################

        # Classification part
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(nn.AvgPool2d(
            (args.img_height // args.downsampling, args.img_width // args.downsampling)))
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(
            nn.Conv2d(int(self.network_settings[8]['c'] * args.width_multiplier), self.num_classes,
                      1, bias=True))

        self.network = nn.Sequential(*self.network)

        self.initialize()

    def forward(self, x):
        # Debugging mode
        # for op in self.network:
        #     x = op(x)
        #     print(x.shape)
        x = self.network(x)
        x = x.view(-1, self.num_classes)
        return x

    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

def mobilenet(args):
    """
    construct mobilenet-v2
    """
    return MobileNetV2(n_class=args.num_classes, input_size=args.img_height)