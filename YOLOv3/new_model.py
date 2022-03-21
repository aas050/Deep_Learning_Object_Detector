# We used the following tutorail from Ultralytics in order to implement our simplified version of YoloV3 model
# so that it could be trained in Datahub. The link to the github is https://github.com/ultralytics/yolov3.
import numpy as np
import torch
import torch.nn as nn
from layers import *
import yaml, math

# Initialize the weights.
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.SiLU]:
            m.inplace = True

# Parse the model hyperparameters.
def parse_model(config):
    anchors = config['anchors']
    nc = config['nc']
    ch = config['ch']
    na = len(anchors[0])//2
    no = na * (nc + 5)
    ch = [ch]

    layers, save, c2 = [], [], ch[-1]
    for i, (from_layer, num, layer_name, args) in enumerate(config['backbone'] + config['head']):
        layer = eval(layer_name) if isinstance(layer_name, str) else layer_name
        for j, arg in enumerate(args):
            try:
                args[j] = eval(arg) if isinstance(arg, str) else arg
            except NameError:
                pass
        if layer in [Conv, Bottleneck]:
            in_ch, out_ch = ch[from_layer], args[0]
            if out_ch != no:
                out_ch = math.ceil(out_ch / 8) * 8
            args = [in_ch, out_ch, *args[1:]]

        elif layer is Concat:
            out_ch = sum(ch[x] for x in from_layer)

        elif layer is Detect:
            args.append([ch[x] for x in from_layer])

        m_ = nn.Sequential(*(layer(*args) for _ in range(num))) if num > 1 else layer(*args)
        layers.append(m_)
        m_.i, m_.from_layer= i, from_layer
        if i == 0:
            ch = []
        ch.append(out_ch)

    return nn.Sequential(*layers)

class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2).to(torch.device("cuda:0"))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class Model(nn.Module):
    def __init__(self, cfg, hyp):
        super().__init__()
        self.hyp = hyp

        with open(cfg) as f:
                self.config = yaml.safe_load(f)

        self.model = parse_model(self.config)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.config['ch'], s, s))]).to(torch.device("cuda:0"))  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)

    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            # print(m.i, x.shape, m.from_layer)
            # print(m)
            if m.from_layer != -1:  # if not from previous layer
                x = y[m.from_layer] if isinstance(m.from_layer, int) else [x if j == -1 else y[j] for j in m.from_layer]  # from earlier layers
            x = m(x)  # run
            y.append(x)
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)