"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_normal(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
            self.bias = Parameter(init.kaiming_normal(out_features, 1, device=device, dtype=dtype).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = out + ops.broadcast_to(self.bias, out.shape)
        return out 
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(X)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        """
        sum(logsumexp - z_y)/batch_size
        """
        batch_size, class_num = logits.shape
        lse = ops.logsumexp(logits, axes=(1,)) # logsumexp
        z_y = ops.summation(logits * init.one_hot(class_num, y), axes=1)
        return ops.summation(lse - z_y)/batch_size 
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # 初始化为恒等变换
        self.weight = Parameter(init.ones(1,dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1,dim, device=device, dtype=dtype))
        # 在深度学习框架中，通常习惯将这种非权重参数（Buffer）存储为最简洁的一维形式，使其在查看、保存或手动检查时更加直观。
        # 也更加符合语义
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size, _ = x.shape
            mean_batch = ((ops.summation(x, axes=0, keepdims=True))/batch_size) # (1,n)
            diff = x - mean_batch.broadcast_to(x.shape)
            var_batch =  ((ops.summation((diff)**2, axes=0, keepdims=True))/batch_size) # (1,n)

            # 核心目的是引入轻微随机性（不同 batch 的统计量略有差异），带来天然的正则化效果，减少过拟合；
            # 若训练时用 running_mean，会导致归一化分布过于稳定，失去 BN 的部分正则化能力，同时也无法快速适配当前 batch 的局部数据分布。
            out = diff/(var_batch.broadcast_to(x.shape)+self.eps)**0.5 # （b,n）
            out = out * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
            self.running_mean.data = (1-self.momentum) * self.running_mean.data + self.momentum * mean_batch.data.reshape(self.running_mean.shape)
            self.running_var.data = (1-self.momentum) * self.running_var.data + self.momentum * var_batch.data.reshape(self.running_var.shape)
        else:
            _, n = x.shape
            view_shape = (1, n)
            out = (x - self.running_mean.reshape(view_shape).broadcast_to(x.shape))/(self.running_var.reshape(view_shape).broadcast_to(x.shape)+self.eps)**0.5
            out = out * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        return out
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # 一开始是恒等变换
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype)) #  
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        """
        assume the input to this layer is a 2D tensor, with batches in the first dimension and features in the second
        x: (b, n)
        """
        _, feature_num = x.shape
        mu = (ops.summation(x, axes=1, keepdims=True)/feature_num).broadcast_to(x.shape) # (b,n)
        sigma = (((x - mu)**2).sum(axes=1, keepdims=True)/feature_num+self.eps)**0.5  # (b,1)
        sigma = sigma.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * (x-mu)/sigma + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x 
        else:
            keep_p = 1 - self.p
            mask = init.randb(*x.shape, p=keep_p)
            x = (x * mask) / keep_p
            return x 
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
