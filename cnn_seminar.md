<!-- $theme: default -->

# 卷积神经网络的主要结构
---
<!-- page_number: true -->
# 卷积层

torch中的卷积操作有两种，分别为时域卷积和空间卷积：

- [时域卷机](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalconvolution) ：对于一个输入序列，进行卷积操作。

- [空间卷积](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution)：对输入的2D或3D矩阵进行卷积。

---
<!-- page_number: true -->
## 空间卷积
在卷积神经网络中使用的为空间卷积，在THNN中对应[SpatialConvolutionMM.c函数](https://github.com/torch/nn/blob/master/lib/THNN/generic/SpatialConvolutionMM.c)。

---
<!-- page_number: true -->
卷积函数的输入参数如下：

```lua 
module = nn.SpatialConvolution(nInputPlane, nOutputPlane, 
                               kW, kH, [dW], [dH], [padW], [padH])
```

- 输入数据：nInputPlane，输入数据为3位矩阵，三维分别为通道数n_i，宽w_i，高h_i
- 输出数据：nOutputPlane，输入数据为3位矩阵，三维分别为通道数n_o，宽w_o，高h_o，其中输入与输出的通道数可能不等
- 卷积核参数：kW, kH，卷积核尺寸通常为正方形，但在某些网络中，可以使用1xN与Nx1的卷积核组合替代NxN的卷积核，从而减少运算量。
- 步长参数：dW, dH，卷积核在运算时的步长。
- 边缘填充参数：padW，padH，在卷积操作前在边缘填充0的数量。

---
<!-- page_number: true -->
其中，输入尺寸与输出尺寸的关系如下：
```lua
w_o = floor((w_i + 2*padW - kW) / dW + 1)
h_o = floor((h_i + 2*padH - kH) / dH + 1)
```
在卷积操作运算前，通常要确定输出数据的尺寸，确保输出数据尺寸不为0，从而判断判断输入数据是否能够继续卷积

---
<!-- page_number: true -->
torch中对于卷积函数的实现方式为：

> nn(lua)->THNN(C)->THTensor(C)->THBlas(C)->LAPACK(Fortran)

目前主流框架计算卷积时，会将数据进行展开，采用BLAS（基础线性代数子程序库），使用其中的数学运算操作实现多维矩阵的计算。对于采用CUDA的系统，则会使用NVIDIA提供的cuBLAS库实现在CUDA平台的加速运算。

---
<!-- page_number: true -->
### 计算流程
整个卷积的流程被封装在`THNN_(SpatialConvolutionMM_updateOutput_frame`这个函数里，分三步实现多维平面的卷积。分别为矩阵展开，偏移填充，矩阵相乘。

---
<!-- page_number: true -->
### 降维展开
由于输入数据通常是三维矩阵，而卷积则是对其中的平面进行操作，则可以将输入数据进行展开，生产一个较大的二位矩阵，从而实现加速计算。实现展开操作的函数是[unfold.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/unfold.c)中的`THNN_(unfolded_copy)`。
```cpp
THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH,
                   padW, padH,
		   nInputPlane, inputWidth, inputHeight,
		   outputWidth, outputHeight);
```
其中，展开操作将输入input展开为finput，输出尺寸outputWidth, outputHeight可以根据卷积参数计算出来。finput的尺寸为`[n*kW*kH]*[outW*outH]`。

---
<!-- page_number: true -->
### 偏移填充
填充的目的是利用BLAS矩阵乘法的特点，先行构建输出矩阵，并将偏移值(bias)填充进去，实现卷积操作。
```cpp
// 定义一个[outN]*[outW*outW]的输出矩阵
output2d = THTensor_(newWithStorage2d)(output->storage,
                    output->storageOffset,
                    nOutputPlane, -1,
                    outputHeight*outputWidth, -1);
// 如果有bias不为0则填充bias数值，为零则填充为0。
if (bias) {
  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data 
                    + output->storageOffset 
                    + output->stride[0] * i,
                    THTensor_(get1d)(bias, i), 
                    outputHeight*outputWidth);
} else {
  THTensor_(zero)(output);
}
```

---
<!-- page_number: true -->
### 矩阵相乘
将平移展开后的输入finput，通过与展开后的weight矩阵做乘法，得到卷积结果output，weight，finput，output矩阵关系如图所示：

![avatar](http://img.blog.csdn.net/20160831173531082)

---
<!-- page_number: true -->
矩阵相乘则使用[THTensorMath.c](https://github.com/torch/torch7/blob/master/lib/TH/generic/THTensorMath.c)中的`THTensor_(addmm)`函数实现
```cpp
THTensor_(addmm)(output2d, 
                 1, output2d, 
                 1, weight, finput);
```
该函数完成`output2d = 1*output2d + 1*weight*finput`的运算，其中output2d填充了bias，weight为复制展开的卷积核，finput为降维展开后的输入矩阵。
相乘步骤中的计算量为`[outN]*[outW*outW]`的偏移加法，`[outN]*[outW*outH]*[inN*kW*kH]`的乘法，`([inN*kW*kH]-1)*[outN]*[outW*outH]`的加法。

---
<!-- page_number: true -->
更深层次的计算，则依靠[THBlas_(gemm)](https://github.com/torch/torch7/blob/master/lib/TH/generic/THBlas.c)的实现。再经过一系列的尺寸，参数的合法性检查后，根据数据的浮点精度，判断传递给单精度矩阵计算`sgemm`，或者双精度矩阵计算`dgemm`。

---
<!-- page_number: true -->
# 传递函数(Transfer Function)
传递函数通常在参数转化层之后引入非线性关系，[将问题空间分割成更复杂的区域](https://github.com/torch/nn/blob/master/doc/transfer.md#transfer-function-layers)。在神经网络中则被称为激活函数(Activation Function)。

---
<!-- page_number: true -->
早期的激活函数(sigmoid, tanh)在浅层的神经网络中，具有很好的效果，但随着网络的复杂化，在卷积神经网络(CNN)中，ReLU(线性整流函数)具有结构简单，正向计算和反向传递运算量低等优点。

---
<!-- page_number: true -->
## ReLU
线性整流函数（Rectified Linear Unit, ReLU）,又称修正线性单元, 是一种人工神经网络中常用的激活函数，通常指代以斜坡函数及其变种为代表的非线性函数。

`ReLU`的定义如下：
```lua
f(x) = max(0, x)
```
---
<!-- page_number: true -->
在THNN中，使用nn.ReLU调用函数，传递给Threshold.c执行计算：
```lua
f = nn.ReLU([inplace]) 。
```
- 输入数据(THTensor *input)：输入数据，
- 输出数据(THTensor *output)：输出数据，
- 阈值(accreal threshold_)：在ReLU中，阈值为0
- 阈值外数值(accreal val_)：在ReLU中，阈值外数值为0
- 计算模式(inplace)：为true则在原变量上操作，为false则在其他变量上操作

---
<!-- page_number: true -->
C语言核心代码如下：
### inplace模式
```cpp
TH_TENSOR_APPLY(real, input,
    if (*input_data <= threshold)
         *input_data = val;
);
THTensor_(set)(output, input);
```

---
<!-- page_number: true -->
### 在外部变量中操作
```cpp
THTensor_(resizeAs)(output, input);
TH_TENSOR_APPLY2(real, output, real, input,
  *output_data = (*input_data > threshold) ? 
                 *input_data : val;
);
```

---
<!-- page_number: true -->
两种方式都需要对每一个输入元素进行比较，所以比较的计算量为输入元素尺寸乘积。

---
<!-- page_number: true -->
## SoftMax
[Softmax函数](https://en.wikipedia.org/wiki/Softmax_function)，或称归一化指数函数，是逻辑函数的一种推广。它能将一个含任意实数的K维的向量 
 _z_ 的“压缩”到另一个K维实向量 _σ(z)_ 中，使得每一个元素的范围都在(0, 1)之间，并且所有元素的和为1。

---
<!-- page_number: true -->
`Softmax` 定义如下:

```lua
f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
```
其中`shift = max_i(x_i)`，将指数计算固定在负数域，避免溢出。

---
<!-- page_number: true -->
在THNN中，使用nn.SoftMax调用函数，传递给[SoftMax.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/SoftMax.c)执行计算
```lua
f = nn.SoftMax()
```

---
<!-- page_number: true -->
根据输入数据纬度的不同，softmax运算中涉及到的参数也不同

- 1-D：一维向量，在卷积神经网络中通常为
- 2-D：带有时序的一维向量，时序nframe，向量维度dim
- 3-D：三维矩阵，参数有维度，每个维度的长度和高度
- 4-D：带有时序的三位矩阵

在卷积神经网络中，Softmax常用于给予概率的多分类问题。

---
<!-- page_number: true -->
C语言的核心代码如下：
### 指针的定义
```cpp
// 使用指针直接对输入输出数据进行寻址
real *input_ptr = input_data 
                  + (t/stride)*dim*stride 
                  + t % stride;
real *output_ptr = output_data 
                   + (t/stride)*dim*stride 
                   + t % stride;
```

---
<!-- page_number: true -->
### 最大值查询
在对一维向量的操作中，需要对向量中的每个元素进行遍历，分别进行比较，比较的操作次数为dim的数值
```cpp
real inputMax = -THInf;
accreal sum;
// 遍历输入向量寻找最大值
 ptrdiff_t d;
for (d = 0; d < dim; d++)
{
  if (input_ptr[d*stride] >= inputMax) 
    inputMax = input_ptr[d*stride];
}
```

---
<!-- page_number: true -->
### 完成Softmax函数的计算
计算指数时，需要先计算输入向量减去最大值的数值，在进行指数运算，之后对指数进行求和。在向量归一化中，需要对向量进行除法操作。则计算量为dim*(2* 加减 + 1* 指数 + 1* 除法)
```cpp
sum = 0;
// 计算向量元素指数，并求和
for (d = 0; d < dim; d++)
{
  real z = exp(input_ptr[d*stride] - inputMax);
  output_ptr[d*stride] = z;
  sum += z;
}
// 向量归一化
for (d = 0; d < dim; d++)
{
  output_ptr[d*stride] *= 1/sum;
}
```

---
<!-- page_number: true -->
# 简单非线性层

---
<!-- page_number: true -->
## 批归一化(BatchNormalization)
从字面意思看来Batch Normalization（简称BN）就是对每一批数据进行归一化
批归一化层只接受二维的输入。
```lua
              x - mean(x)
y =  ----------------------------- * gamma + beta
      standard-deviation(x) + eps
```

其中，每一个输入的通道N对应一个标准差，而gamma和beta为可学习的和通道数N 有关的数值。gamma和beta的学习性时可选参数。

---
<!-- page_number: true -->
在THNN中，使用`nn.BatchNormalization`调用函数，传递给[BatchNormalization.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/BatchNormalization.c)执行计算：
```lua
module = nn.BatchNormalization(N [, eps] 
                               [, momentum] [,affine])
```
其中 `N` 为输入的维度
`eps` 是加在标准差中的一个极小值，以防止除0的情况发生. 默认值为 `1e-5`.
`affine` 是一个boolean值. 当设为false时, 仿射变换的可学习性被关闭. 默认为true

---
<!-- page_number: true -->
批归一化主要的计算量在于计算平均值和方差，其中平均值由于训练数据读取的方式，很难直接求得，根据目前神经网络数据的送入方式(mini-batch)，可采用滑动平均。
所以，在训练过程中，做归一化时需要大量的计算，而在训练完成后，滑动平均值会保存在模型里，在应用模型时，只需要读取保存好的滑动平均，计算方差。

---
<!-- page_number: true -->
C语言核心代码如下：
### 计算平均值和标准差倒数
```cpp
// 计算平均值
accreal sum = 0;
TH_TENSOR_APPLY(real, in, sum += *in_data;);

mean = (real) sum / n;
THTensor_(set1d)(save_mean, f, (real) mean);
```

---
<!-- page_number: true -->
为了简化计算，直接求得标准差的倒数。
```cpp
// 计算标准差倒数
sum = 0;
TH_TENSOR_APPLY(real, in,
sum += (*in_data - mean) * (*in_data - mean););

if (sum == 0 && eps == 0.0) {
  invstd = 0;
} else {
  invstd = (real) (1 / sqrt(sum/n + eps));
}
THTensor_(set1d)(save_std, f, (real) invstd);
```

---
<!-- page_number: true -->
### 更新滑动平均与滑动方差
```cpp
// 更新滑动平均
THTensor_(set1d)(running_mean, f,
  (real) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));
//更新滑动方差
accreal unbiased_var = sum / (n - 1);
THTensor_(set1d)(running_var, f,
  (real) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));
```

---
<!-- page_number: true -->
### 读取滑动的平均值与标准差倒数
```cpp
mean = THTensor_(get1d)(running_mean, f);
invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
```

---
<!-- page_number: true -->
### 利用向量乘法实现归一化
```cpp
// 计算归一化参数
real w = weight ? THTensor_(get1d)(weight, f) : 1;
real b = bias ? THTensor_(get1d)(bias, f) : 0;

TH_TENSOR_APPLY2(real, in, real, out,
*out_data = (real) (((*in_data - mean) * invstd) * w + b););
```

---
<!-- page_number: true -->
批归一化的计算量和当前卷积层的输出数据尺寸有关，batch大小为N，通道数为N，特征图大小为W和H，则做批归一化的数据量为`M*N*W*H`。其中计算标准差倒数时需要分别实现N个通道的标准差计算。而输入数据的每一个元素则需要进行两次加减法，两次乘法的计算。

---
<!-- page_number: true -->
# 感谢大家的参与