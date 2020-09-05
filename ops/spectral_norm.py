from paddle import fluid
import paddle.fluid.layers as L
import numpy as np
import math
from paddle.fluid.initializer import Normal, Uniform
from paddle.fluid.framework import _varbase_creator

class MySpectralNorm(fluid.dygraph.Layer):
    def __init__(self,
                 weight_shape,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(MySpectralNorm, self).__init__()
        self.power_iters = power_iters
        self.eps = eps
        self.weight_u = self.create_parameter(
            attr=None,
            shape=(weight_shape[0],),
            dtype='float32',
            is_bias=False,
            default_initializer=Normal(0, 1))
        self.weight_v = self.create_parameter(
            attr=None,
            shape=(int(np.prod(weight_shape[1:])),),
            dtype='float32',
            is_bias=False,
            default_initializer=Normal(0, 1))
    def forward(self, weight):
        weight_mat = L.reshape(weight, (weight.shape[0], -1))
        with fluid.dygraph.no_grad():
            for i in range(self.power_iters):
                self.weight_v.set_value(L.l2_normalize(
                    L.matmul(weight_mat, self.weight_u, transpose_x=True, transpose_y=False),
                    axis=0, epsilon=self.eps,
                    )
                )

                self.weight_u.set_value(L.l2_normalize(
                    L.matmul(weight_mat, self.weight_v, transpose_x=False, transpose_y=False),
                    axis=0, epsilon=self.eps,
                    )
                )
        sigma = L.matmul(self.weight_u, L.matmul(weight_mat, self.weight_v)) 
        norm_weight = L.elementwise_div(weight, sigma)
        return norm_weight
                 

class SpectralNormConv(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, filter_size, stride, padding, bias_attr=False):
        super(SpectralNormConv, self).__init__()
        bias = bias_attr
        kernel_size = filter_size

        def _get_default_param_initializer():
            filter_elem_num = kernel_size * kernel_size * in_channels
            negative_slope = math.sqrt(5)  
            gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
            bound = gain * math.sqrt(3.0 / filter_elem_num)
            return Uniform(-bound, bound, 0)

        def _get_default_bias_initializer():
            fan_in = kernel_size * kernel_size * in_channels
            bound = 1.0 / math.sqrt(fan_in) 
            return Uniform(-bound, bound, 0)

        self.weight_orig = self.create_parameter(
            attr=None,
            shape=(out_channels, in_channels, kernel_size, kernel_size),
            dtype='float32',
            is_bias=False,
            default_initializer=_get_default_param_initializer())
        self.spectral_norm = MySpectralNorm(
                     weight_shape=(out_channels, in_channels, kernel_size, kernel_size),
                     dim=0,
                     power_iters=1,
                     eps=1e-12)
        if bias:
            self.bias = self.create_parameter(
                attr=True,
                shape=[out_channels],
                dtype='float32',
                is_bias=True,
                default_initializer=_get_default_bias_initializer(),
                )
        else:
            self.bias = None
        stride = L.utils.convert_to_list(stride, 2, 'stride')
        padding = L.utils.convert_to_list(padding, 2, 'padding')
        dilation = L.utils.convert_to_list(1, 2, 'dilation')
        self.attr = ('strides', stride, 'paddings', padding,
                'dilations', dilation, 'groups', 1, 'use_cudnn', True)
    def forward(self, x):
        self.weight = self.spectral_norm(self.weight_orig)
        y = fluid.core.ops.conv2d(x, self.weight, *self.attr)
        if self.bias is None:
            return y
        return y + L.unsqueeze(self.bias, (0,2,3))


class SpectralNormLinear(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, bias_attr=False):
        super(SpectralNormLinear, self).__init__()
        bias = bias_attr

        def _get_default_param_initializer():
            filter_elem_num = in_channels
            negative_slope = math.sqrt(5)  
            gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
            bound = gain * math.sqrt(3.0 / filter_elem_num)
            return Uniform(-bound, bound, 0)

        def _get_default_bias_initializer():
            fan_in = in_channels
            bound = 1.0 / math.sqrt(fan_in) 
            return Uniform(-bound, bound, 0)

        self.weight_orig = self.create_parameter(
            attr=None,
            shape=(out_channels, in_channels),
            dtype='float32',
            is_bias=False,
            default_initializer=_get_default_param_initializer(),
            )
        self.spectral_norm = MySpectralNorm(
                     weight_shape=(out_channels, in_channels),
                     dim=0,
                     power_iters=1,
                     eps=1e-12)
        if bias:
            self.bias = self.create_parameter(
                attr=True,
                shape=[out_channels],
                dtype='float32',
                is_bias=True,
                default_initializer=_get_default_bias_initializer(),
                )
        else:
            self.bias = None
    def forward(self, x):
        self.weight = self.spectral_norm(self.weight_orig)
        pre_bias = _varbase_creator(dtype='float32')
        y = fluid.core.ops.matmul(x, self.weight, pre_bias, 'transpose_X', False, 'transpose_Y', True, 'alpha', 1)
        if self.bias is None:
            return y
        return y + L.unsqueeze(self.bias, 0)
