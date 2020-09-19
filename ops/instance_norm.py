from paddle import fluid
import paddle.fluid.layers as L
from paddle.fluid.dygraph import BatchNorm
from .base import var


class MyInstanceNorm2d(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(MyInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
    def forward(self, x):
        if self.training:
            N, C, H, W = x.shape
            x_r = L.reshape(x, (1, N*C, H, W))
            # for numeric stability
            y_r = L.batch_norm(x_r, epsilon=self.eps, do_model_average_for_mean_and_var=False, use_global_stats=False)
            return L.reshape(y_r, (N, C, H, W))
        else:
            _mean = L.reduce_mean(x, dim=[2, 3], keep_dim=True)
            _var = var(x, dim=[2, 3], unbiased=False, keepdim=True)
            y = (x - _mean) / L.sqrt(_var + self.eps)
            return y
