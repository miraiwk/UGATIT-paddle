from paddle.fluid.dygraph import nn
from paddle.fluid.dygraph.nn import utils, core, in_dygraph_mode, _varbase_creator, dygraph_utils
from paddle.fluid.initializer import Uniform, Constant
from paddle.fluid.dygraph.layers import Layer
import math

class MyConv2D(nn.Conv2D):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        assert param_attr is not False, "param_attr should not be False here."
        Layer.__init__(self)
        self._num_channels = num_channels
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._padding = utils.convert_to_list(padding, 2, 'padding')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._act = act
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._use_mkldnn = core.globals()["FLAGS_use_mkldnn"]
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype

        if (self._num_channels == self._groups and
                num_filters % self._num_channels == 0 and
                not self._use_cudnn and not self._use_mkldnn):
            self._l_type = 'depthwise_conv2d'
        else:
            self._l_type = 'conv2d'

        self._num_channels = num_channels
        if self._groups is None:
            num_filter_channels = self._num_channels
        else:
            if self._num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = self._num_channels // self._groups
        filter_size = utils.convert_to_list(self._filter_size, 2, 'filter_size')
        filter_shape = [self._num_filters, num_filter_channels] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[
                1] * self._num_channels
            negative_slope = math.sqrt(5)  
            gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
            bound = gain * math.sqrt(3.0 / filter_elem_num)
            return Uniform(-bound, bound, 0)

        def _get_default_bias_initializer():
            fan_in = filter_size[0] * filter_size[
                1] * self._num_channels
            bound = 1.0 / math.sqrt(fan_in) 
            return Uniform(-bound, bound, 0)

        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True,
            default_initializer=_get_default_bias_initializer())


class MyLinear(Layer):
    def __init__(self,
             input_dim,
             output_dim,
             param_attr=None,
             bias_attr=None,
             act=None,
             dtype="float32"):
        Layer.__init__(self)
        self._act = act
        self._dtype = dtype

        def _get_default_param_initializer():
            filter_elem_num = input_dim
            negative_slope = math.sqrt(5)  
            gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
            bound = gain * math.sqrt(3.0 / filter_elem_num)
            return Uniform(-bound, bound, 0)

        def _get_default_bias_initializer():
            fan_in = input_dim
            bound = 1.0 / math.sqrt(fan_in) 
            return Uniform(-bound, bound, 0)

        self.weight = self.create_parameter(
            shape=[output_dim, input_dim],
            attr=param_attr,
            dtype=dtype,
            is_bias=False,
            default_initializer=_get_default_param_initializer())
        self.bias = self.create_parameter(
            shape=[output_dim], attr=bias_attr, dtype=dtype, is_bias=True,
            default_initializer=_get_default_bias_initializer())

        self._use_mkldnn = core.globals()["FLAGS_use_mkldnn"]

    def forward(self, input):
        if in_dygraph_mode():
            pre_bias = _varbase_creator(dtype=input.dtype)
            core.ops.matmul(input, self.weight, pre_bias, 'transpose_X', False,
                            'transpose_Y', True, "alpha", 1, "use_mkldnn",
                            self._use_mkldnn)
            pre_act = dygraph_utils._append_bias_in_dygraph(
                pre_bias,
                self.bias,
                axis=len(input.shape) - 1)

            return dygraph_utils._append_activation_in_dygraph(
                pre_act, self._act, use_mkldnn=self._use_mkldnn)
        raise NotImplementedError()
