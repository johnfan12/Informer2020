import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        # MindSpore不需要no_grad上下文
        ones = ops.ones(mask_shape, ms.bool_)
        self._mask = ops.triu(ones, diagonal=1)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        ones = ops.ones((L, scores.shape[-1]), ms.bool_)
        _mask = ops.triu(ones, diagonal=1)
        _mask_ex = ops.expand_dims(_mask, axis=0)
        _mask_ex = ops.expand_dims(_mask_ex, axis=0)
        _mask_ex = ops.tile(_mask_ex, (B, H, 1, 1))
        
        # 使用gather操作来模拟高级索引
        B_range = ops.arange(B).view(B, 1, 1)
        H_range = ops.arange(H).view(1, H, 1)
        
        indicator = ops.gather_nd(_mask_ex, ops.stack([B_range, H_range, index], axis=-1))
        self._mask = indicator.view(scores.shape)
    
    @property
    def mask(self):
        return self._mask