from posix import lseek
from typing import Optional, Any, Union

from numpy import ndarray
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

from typing import TYPE_CHECKING, Optional, TypeAlias, List, Tuple, Union, Sequence, cast
if TYPE_CHECKING:
    # 只在 Pyright 做静态分析时执行，不影响运行时。
    # 它告诉 Pyright："array_api 就是 backend_ndarray"，
    # 这样所有类型推断和跳转都基于 backend_ndarray 模块
    from .. import backend_ndarray as array_api
    NDArray: TypeAlias = array_api.NDArray

def _logsumexp(Z, axes: Optional[tuple] = None, keepdims=False):
    if axes is None:
        view_shape = tuple(1 for _ in range(len(Z.shape)))
    else:
        if isinstance(axes, int):
            axes = (axes,)
        ndim = len(Z.shape)
        axes = tuple(dim if dim>=0 else dim + ndim for dim in axes)
        view_shape = tuple(1 if i in axes else dim for i,dim in enumerate(Z.shape))
    Z_max = Z.max(axes)
    lse = Z_max + array_api.log(array_api.sum(array_api.exp(Z-Z_max.reshape(view_shape).broadcast_to(Z.shape)), axis=axes))
    if keepdims:
        return lse.reshape(view_shape)
    return lse   


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return _logsumexp(Z, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if TYPE_CHECKING:
            Z = cast(Tensor, Z)
        lse = expand_reduced(node, self.axes, Z.shape) 
        out_grad = expand_reduced(out_grad, self.axes, Z.shape)
        return out_grad * exp(Z-lse)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axes = (-1,)
        lse = _logsumexp(Z, axes, keepdims=True)
        return Z -lse.broadcast_to(Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if TYPE_CHECKING:
            Z = cast(Tensor, Z)
        softmax = exp(node)
        out_grad_sum = expand_reduced(out_grad.sum((-1,)), (-1,), Z.shape)
        return out_grad - out_grad_sum * softmax
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)