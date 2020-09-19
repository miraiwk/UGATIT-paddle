from paddle import fluid
import paddle.fluid.layers as L
from paddle.fluid.dygraph import BatchNorm


class MyInstanceNorm2d(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(MyInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.bn = BatchNorm(num_features, epsilon=self.eps, do_model_average_for_mean_and_var=False, use_global_stats=False)
    def forward(self, x):
        N, C, H, W = x.shape
        x_r = L.reshape(x, (1, N*C, H, W))
        y_r = self.bn(x_r)
        return L.reshape(y_r, (N, C, H, W))
