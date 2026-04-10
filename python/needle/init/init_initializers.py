import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    """
    why need shape parameter here?
    只要权重 shape 不是简单的 (fan_in, fan_out) 二维矩阵，就需要显式传 shape 参数。 
    Conv 就是典型场景，因为 fan_in/fan_out 是对多个维度做了聚合（乘积），丢失了原始的维度信息。
    fan_in = kernel_size * kernel_size * in_channels（一个输出神经元连接的输入数量）
    fan_out = kernel_size * kernel_size * out_channels
    """
    ### BEGIN YOUR SOLUTION
    sigma = gain * math.sqrt(6.0 / (fan_in + fan_out))
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-sigma, high=sigma, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    sigma = gain * math.sqrt(2.0 / (fan_in + fan_out))
    if shape is None:
        shape = (fan_in, fan_out)
    return randn(*shape, std=sigma, **kwargs)
    
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    var = math.sqrt(6.0 / fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-var, high=var, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    sigma = 1 / math.sqrt(fan_in/2) 
    if shape is None:
        shape = (fan_in, fan_out)
    return randn(*shape, std=sigma, **kwargs)
    ### END YOUR SOLUTION