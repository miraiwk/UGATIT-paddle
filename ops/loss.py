import paddle.fluid as fluid
import paddle.fluid.layers as L

class L1Loss(fluid.dygraph.Layer):
    def __init__(self):
        super(L1Loss, self).__init__()
    def forward(self, x, y):
        return L.mean(L.abs(x - y))

class MSELoss(fluid.dygraph.Layer):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, x, y):
        return L.mse_loss(x, y)

class BCEWithLogitsLoss(fluid.dygraph.Layer):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
    def forward(self, x, y):
        return L.mean(L.sigmoid_cross_entropy_with_logits(x, y))
