"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for idx, p in enumerate(self.params):
            u = self.momentum * self.u.get(idx,0) + (1 - self.momentum) * (p.grad.data + self.weight_decay * p.data)
            p.data -= self.lr * u
            self.u[idx] = u.detach()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for idx, p in enumerate(self.params):
            g = ndl.Tensor(p.grad.data + self.weight_decay * p.data, dtype="float32").data
            m = self.beta1 * self.m.get(idx, 0) + (1-self.beta1)*g
            v = self.beta2 * self.v.get(idx, 0) + (1-self.beta2)*(g**2)
            self.m[idx] = ndl.Tensor(m, dtype="float32").data
            self.v[idx] = ndl.Tensor(v, dtype="float32").data
            m_hat = self.m[idx]/(1-self.beta1**self.t)
            v_hat = self.v[idx]/(1-self.beta2**self.t)
            p.data -= self.lr * m_hat/(v_hat**0.5+self.eps)
        ### END YOUR SOLUTION
