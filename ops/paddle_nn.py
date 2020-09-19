#coding:utf-8
from paddle import fluid
from paddle.fluid.dygraph import *
from paddle.fluid.initializer import Normal
from paddle.fluid.framework import _varbase_creator
import paddle.fluid.layers as L
import numpy as np
import math
from .basic_nn import *
from .spectral_norm import MySpectralNorm, SpectralNormConv, SpectralNormLinear
from .base import var

from .instance_norm import MyInstanceNorm2d as InstanceNorm2d

class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = padding
    def forward(self, x):
        padding = self.padding
        if isinstance(padding, int):
            padding = [padding for _ in range(4)]
        return L.pad2d(x, paddings=padding, mode='reflect')

class ReLU(fluid.dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()
    def forward(self, x):
        return L.relu(x)

class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return L.leaky_relu(x, alpha=self.alpha)

class Tanh(fluid.dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()
    def forward(self, x):
        return L.tanh(x)

class Shape:
    def __init__(self, shape):
        self.shape = shape

class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        assert mode in ['nearest']
        self.mode = mode
        self.scale = scale_factor
    def forward(self, x):
        return L.resize_nearest(x, scale=self.scale)
