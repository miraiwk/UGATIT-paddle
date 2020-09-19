import paddle.fluid.layers as L
import numpy as np

def var(x, dim, unbiased=True, keepdim=False):
    # unbiased variance
    shape = x.shape
    if isinstance(dim, int):
        e = shape[dim]
    else:
        e = int(np.prod([shape[d] for d in dim]))
    if unbiased:
        e -= 1
    return L.reduce_sum(L.square(x - L.reduce_mean(x, dim=dim, keep_dim=True)), dim=dim, keep_dim=keepdim) / e
