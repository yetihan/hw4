---
date: 2026-04-16
tags:
  - dl-system
  - hw4
  - convolution
  - backward-pass
  - ndarray
  - reduce
---

# HW4: Conv Backward Pass & Multi-Axis Reduce

## Summary

HW4 的核心挑战是实现 `Conv` 算子的反向传播（`dL/dZ` 和 `dL/dF`），以及修复 NDArray 底层 `reduce` 不支持多轴的问题。过程中还涉及 `compact()` 的正确使用、`dilate` 处理 stride>1、以及 `nn.Conv` 模块的 NCHW↔NHWC 转换。

## Details

### 1. Conv Forward (im2col)

`Conv.compute` 使用 `as_strided` 创建 6D 视图实现 im2col：

```python
# Z: (N, H, W, C_in) — NHWC format
# F: (k, k, C_in, C_out)
Z_6D = Z_padded.as_strided(
    shape=(N, H_out, W_out, k, k, C),
    strides=(N_s, H_s*s, W_s*s, H_s, W_s, C_s)
)
res = Z_6D.compact().reshape((N*H_out*W_out, k*k*C)) @ F.compact().reshape((k*k*C, C_out))
```

**关键**：`as_strided` 创建的视图不是 compact 的，必须 `.compact()` 再 `.reshape()`。

### 2. Conv Backward — dL/dZ

卷积的逆操作也是卷积。对输入 Z 的梯度：

```python
F_flip = flip(F, (0,1)).transpose((2,3))
dLdZ = conv(out_grad_dilated, F_flip, stride=1, padding=k-p-1)
```

- **Filter 翻转**：空间维度 180° 旋转 `flip(F, (0,1))`
- **Channel 转置**：`.transpose((2,3))` 把 `(k,k,C_in,C_out)` → `(k,k,C_out,C_in)`
- **Padding**：`k - p - 1`（确保输出和输入 Z 同 shape）

### 3. Conv Backward — dL/dF

对 filter 的梯度，本质是用 Z 作为"输入"、out_grad 作为"filter"做卷积：

```python
dLdF = conv(
    Z.transpose((0,3)),                                    # (C_in, H, W, N)
    out_grad_dilated.transpose((0,2)).transpose((0,1)),    # (H', W', N, C_out)
    stride=1, padding=p
).transpose((0,2)).transpose((0,1))                        # → (k, k, C_in, C_out)
```

维度变换思路：
- `Z.transpose((0,3))`：把 `N` 移到 channel 位，`C_in` 移到 batch 位 → 卷积自动对 batch（即 C_in）累加
- `out_grad` 做两次 transpose：把空间维度放前面，`N` 变成 in_channel，`C_out` 保持
- 卷积结果 `(C_in, k, k, C_out)` 需要两次 transpose 恢复为 `(k, k, C_in, C_out)`

### 4. Stride > 1 的处理

**核心洞察：stride 的效果在反向时被 dilate 完全吸收。**

```python
if s > 1:
    out_grad_dilated = dilate(out_grad, (1, 2), s - 1)
else:
    out_grad_dilated = out_grad
```

- `dilate` 在 out_grad 的空间维度插入 `s-1` 个零
- 之后 `dLdZ` 和 `dLdF` 的卷积**都用 stride=1**
- 直觉：dilate 把"跳着看"的梯度放回原来的空间位置，中间填零表示无贡献

### 5. nn.Conv 模块

```python
class Conv(Module):
    def forward(self, x):
        x = x.transpose((1,3)).transpose((1,2))  # NCHW → NHWC
        res = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size//2)
        if self.bias:
            res = res + self.bias.reshape((1,1,1,C_out)).broadcast_to(res.shape)
        return res.transpose((1,2)).transpose((1,3))  # NHWC → NCHW
```

- 内部算子用 NHWC，外部接口用 NCHW
- `kaiming_uniform` 需要显式传 `shape=(k, k, C_in, C_out)`，因为 `fan_in = k*k*C_in` ≠ filter shape
- Bias 的 `broadcast_to` 触发了 `BroadcastTo.gradient` 中的多轴 `summation`

### 6. NDArray 多轴 Reduce

底层 `reduce_view_out` 一次只能 reduce 一个轴（permute 到末尾再 reduce）。解决方案：

```python
def _reduce_op(self, axis, func_str, keepdims):
    if isinstance(axis, int):
        axes = (axis,)
    # 从大到小排序，先 reduce 高位轴，低位轴 index 不受影响
    for ax in sorted(axes, reverse=True):
        view, out = out.reduce_view_out(ax, keepdims=keepdims)
        func(view.compact()._handle, out._handle, view.shape[-1])
```

**从大到小排序是关键**：`(1, 4, 4, 1)` 对 `axes=(1,2)` → 先 reduce axis 2 得 `(1, 4, 1)` → 再 reduce axis 1 得 `(1, 1)`。如果先 reduce axis 1，原来的 axis 2 会变成 axis 1，导致 index 错位。

### 7. compact() 陷阱

- `flip` 和 `transpose` 返回的是非 compact 视图
- 反向传播时 `F_flip = flip(F, (0,1)).transpose((2,3))` 传入 `Conv.compute`，如果 compute 里不对 F 做 `.compact()` 就 `.reshape()`，会触发 `AssertionError: self.is_compact()`
- **修复**：`F.compact().reshape(...)` 而不是 `F.reshape(...)`

## Key Takeaways

- **卷积的反向也是卷积**：dL/dZ 用翻转后的 filter 卷 out_grad；dL/dF 用 Z 卷 out_grad（维度需要巧妙变换）
- **Stride 在反向时被 dilate 吸收**：dilate out_grad 后一切回归 stride=1
- **维度变换是最易出错的地方**：需要仔细跟踪每一步的 shape 变化，特别是多次 transpose 的顺序
- **`as_strided` 产出的视图必须 compact() 后再 reshape()**
- **NDArray 多轴 reduce 需要从大到小排序逐轴处理**
- **初始化器的 shape 参数**：Conv 层的权重 shape 与 `(fan_in, fan_out)` 不同，必须显式传入

