"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, TypeAlias, Union

from ..autograd import NDArray  # type: ignore
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


from typing import TYPE_CHECKING, Optional, List, Tuple, Union, Sequence, cast
if TYPE_CHECKING:
    # 只在 Pyright 做静态分析时执行，不影响运行时。
    # 它告诉 Pyright："array_api 就是 backend_ndarray"，
    # 这样所有类型推断和跳转都基于 backend_ndarray 模块
    from .. import backend_ndarray as array_api
    NDArray: TypeAlias = array_api.NDArray


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        if TYPE_CHECKING:
            a,b = cast(Tensor, a), cast(Tensor, b)
        return out_grad * b * a ** (b-1), out_grad * (a ** b) * log(a) 
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.pow(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if TYPE_CHECKING:
            out_grad = cast(Tensor, out_grad)
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)    
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs
        if TYPE_CHECKING:
            a = cast(Tensor, a)
            b = cast(Tensor, b)
            out_grad = cast(Tensor, out_grad)
        return  out_grad/b, -out_grad*a/b**2
        
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            axes = (-1, -2)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = [i for i in range(len(a.shape))]
        x, y = self.axes
        axes[x],axes[y] = axes[y], axes[x]
        if TYPE_CHECKING:
            a = cast(array_api.NDArray, a)
        return a.permute(tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if TYPE_CHECKING:
            a = cast(Tensor, a)
            out_grad = cast(Tensor, out_grad)
        return out_grad.reshape(a.shape)
        ### END YOUR SOLUTION



def reshape(a, shape):
    return Reshape(shape)(a)



class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if TYPE_CHECKING:
            a = cast(Tensor, a)
            out_grad = cast(Tensor, out_grad)
        in_shape, out_shape = a.shape, out_grad.shape
        
        # 三种情况
        # a.  (5,) -> (2, 3, 5)  右对齐广播
        # b. (1,1,5) -> (1,3,5)  普通广播
        # c. (1, 5) -> (2, 3, 5) mixed
        view_shape = [1 for _ in range(len(out_shape)-len(in_shape))] + list(in_shape)
        axes = []
        for i, x in enumerate(view_shape):
            if out_shape[i]>x:
                axes.append(i)       

        return summation(out_grad, axes=tuple(axes)).reshape(in_shape)
        
        ### END YOUR SOLUTION

def broadcast_to(a: Value, shape: Union[tuple, Sequence[int]]) -> Tensor:
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):  # type: ignore
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes, keepdims=self.keepdims)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if TYPE_CHECKING:
            a = cast(Tensor, a)
        in_shape = a.shape
        if not in_shape:
            # a是标量
            return out_grad

        # 只有当 keepdims=False 时，我们才需要恢复那些被减掉的维度 (变为1)
        if not self.keepdims:
            if self.axes is None:
                # 全局 sum,相当于传了所有维度
                axes = tuple(range(len(in_shape)))
            else:
                axes = [ax if ax >= 0 else len(in_shape) + ax for ax in self.axes]

            n_dim = len(in_shape)
            view_shape = [in_shape[i] if i not in axes else 1  for i in range(n_dim)]

            out_grad = reshape(out_grad, shape=view_shape)
        return broadcast_to(out_grad, shape=in_shape)
        ### END YOUR SOLUTION

def summation(a: Value, axes: Optional[Union[int, tuple, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        # return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if TYPE_CHECKING:
            out_grad = cast(Tensor, out_grad)
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if TYPE_CHECKING:
            out_grad = cast(Tensor, out_grad)
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        """
        ReLU 的梯度是 indicator 函数 (1 if a > 0 else 0)，
        框架没有将 > 比较运算作为可微的 TensorOp，
        所以直接在 NDArray 层面计算 mask，再包装为 Tensor。
        该 mask 不参与计算图，但 ReLU 二阶导几乎处处为零，无需追踪。
        """
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        mask = Tensor(a.cached_data > 0, device=a.device)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * (1-tanh(a) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = len(args)
        out_shape = list(args[0].shape)
        out_shape.insert(self.axis, n)
        res = array_api.empty(out_shape, device=args[0].device)
        for i in range(n):
            # slice(None)相当于 :,表示选择所有元素
            tar_slice = [slice(None)]*len(out_shape)
            tar_slice[self.axis]=i
            # numpy 等张量库中，整数索引会压缩该维度；此框架中整数索引变为 size=1 的维度，但 setitem 只校验元素总数，效果等价。
            res[tuple(tar_slice)] = args[i]
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A): 
        ### BEGIN YOUR SOLUTION
        out_shape = list(A.shape)
        out_shape = out_shape[:self.axis]+out_shape[self.axis+1:]
        res =[]
        slice_list = [slice(None)]*len(A.shape)
        for i in range(A.shape[self.axis]):
            slice_list[self.axis] = i  # type: ignore
            res.append(A[tuple(slice_list)].compact().reshape(out_shape))
        return tuple(res)
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


