#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BLOCK_SIZE 256
#define REGISTER_TILE 8 
#define SM_TILE (REGISTER_TILE*16) // share memory tile size

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); } //析构(destructor)函数 ,构造(constructor)函数的反面.
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim.block = dim3(BLOCK_SIZE, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t M, size_t P) {
  /**
   * Utility function to get cuda dimensions for 2D call
   * blockDim = (SM_TILE/REGISTER_TILE, SM_TILE/REGISTER_TILE)
   * 每个线程计算 REGISTER_TILE x REGISTER_TILE 个输出元素
   * 每个 block 覆盖 SM_TILE x SM_TILE 个输出元素
   */
  CudaDims dim;
  dim.block = dim3(SM_TILE/REGISTER_TILE, SM_TILE/REGISTER_TILE, 1);
  size_t num_blocks_x = (P+SM_TILE-1)/SM_TILE;
  size_t num_blocks_y = (M+SM_TILE-1)/SM_TILE;
  dim.grid = dim3(num_blocks_x, num_blocks_y, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; // gid 映射到 out_idx

  /// BEGIN SOLUTION
  // 遍历 indices，从 a[a_idx] 复制到 out[out_idx++]; 
  // non-compact -> compact, strides 和 offset 描述的是 a,所以有
  // a_idx = offset + ∑_d(indices[d] * strides[d])
  if(gid >= size) return;
  
  // convert gid to indices by 混合进制下的辗转相除取余法
  int indices[MAX_VEC_SIZE];
  int temp = gid;
  // TODO ,在 GPU 上：% / 都是 高延迟指令,比 + * 慢一个数量级。
  for(int i=shape.size-1; i>=0; i--){
    indices[i] = temp % shape.data[i];
    temp = temp / shape.data[i];
  }

  int a_idx = offset;
   for(int i=shape.size-1; i>=0; i--){
    a_idx += strides.data[i] * indices[i];
  }                 
  out[gid]=a[a_idx];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; // gid 映射到 a_idx

  // 遍历 indices，从 a[gid] 复制到 out[dst_idx]; 
  // compact -> non-compact, strides 和 offset 描述的是 out,所以有
  // dst_idx = offset + ∑_d(indices[d] * strides[d])
  if(gid >= size) return;
  
  // convert gid to indices by 混合进制下的辗转相除取余法
  int indices[MAX_VEC_SIZE];
  int temp = gid;
  for(int i=shape.size-1; i>=0; i--){
    indices[i] = temp % shape.data[i];
    temp = temp / shape.data[i];
  }

  int dst_idx = offset;
   for(int i=shape.size-1; i>=0; i--){
    dst_idx += strides.data[i] * indices[i];
  }                 
  out[dst_idx]=a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}


__global__ void EwiseSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; // gid 映射到 a_idx

  // 遍历 indices，把 out[dst_idx]设置为 val; 
  // dst_idx = offset + ∑_d(indices[d] * strides[d])
  if(gid >= size) return;
  
  // convert gid to indices by 混合进制下的辗转相除取余法
  int indices[MAX_VEC_SIZE];
  int temp = gid;
  for(int i=shape.size-1; i>=0; i--){
    indices[i] = temp % shape.data[i];
    temp = temp / shape.data[i];
  }

  int dst_idx = offset;
   for(int i=shape.size-1; i>=0; i--){
    dst_idx += strides.data[i] * indices[i];
  }                 
  out[dst_idx]=val;
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}



/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}



__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Device 层代码
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Host 层代码
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a * b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}




__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide every element of a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Device 层代码
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Host 层代码
   * Divide two CUDA arrays element-wise.
   *   out: Output array to store the result of 'a / b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::pow(a[gid],val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   *  Element-wise scalar power for CUDA array
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a**val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if(a[gid]>val) out[gid]=a[gid];
    else out[gid]=val;
  }
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Device 层代码
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if(a[gid]>b[gid]) out[gid]=a[gid];
    else out[gid]=b[gid];
  }
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if(a[gid]==val) out[gid]=1.0f;
    else out[gid]=0.0f;
  }
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide every element of a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if(a[gid]==b[gid]) out[gid]=1.0f;
    else out[gid]=0.0f;
  }
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if(a[gid]>=val) out[gid]=1.0f;
    else out[gid]=0.0f;
  }
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if(a[gid]>=b[gid]) out[gid]=1.0f;
    else out[gid]=0.0f;
  }
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid]=log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid]=exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid]=tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


////////////////////////////////////////////////////////////////////////////////



__global__ void MatmulKernelNative1D(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
            uint32_t P) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; // 对应 out (M*P) 中一个元素的 id,对应的二维索引为(i,j), gid = i*P+j
  if (gid >= M * P) return;
  int i = gid / P;
  int j = gid % P;
  scalar_t temp_s = 0;
  for(int k=0; k<N;k++){
    temp_s += a[i*N+k] * b[k*P+j];
  }
  out[gid] = temp_s;
}

__global__ void MatmulKernelNative2D(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
  // (i,j)对应 (M,P) 和(y,x), row-major  第二个维度(列,col)变化最快, 一个 row 在内存上连续
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j>= P) return;
  scalar_t temp_s = 0;
  for(int k=0; k<N;k++){
    temp_s += a[i*N+k] * b[k*P+j];
  }
  out[i*P+j] = temp_s;
}


__global__ void MatmulKernelTiling(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * 使用分块计算矩阵乘法，按照L*L大小分块,再按 V*V 分块, 
   * 一个 cuda block 对应 out(矩阵 C)L*L , block tiling, shared memory tiling 
   * 一个 thread 对应 V*V,  register tiling, 
   * a: M x N, A (M,N) (L,S)
   * b: N x P, B (N,P) (S,L)
   * c: M x P
   * 
   * Grid配置: dim3(P/TILE, M/TILE, 1), (y, x) ->(M, P)
   * blockIdx.y 对应行方向 (M)
   * blockIdx.x 对应列方向 (P)
   */
  // (i,j)对应 (M,P) 和(y,x),一组 (yblock=M/L, xblock=P/L)对应结果矩阵 C 中的 L*L 的一小块, 
  size_t yblock = blockIdx.y; 
  size_t xblock = blockIdx.x;
  // size_t j = xblock * blockDim.x + threadIdx.x;
  // size_t i = yblock * blockDim.y + threadIdx.y;
  
  #define L SM_TILE
  #define S 8
  #define V REGISTER_TILE

  __shared__ scalar_t A_shared[S][L], B_shared[S][L]; //为了避免访问 A_shared的 bank conflict,需要对 A 做了转置.
  scalar_t a_reg[V], b_reg[V], C_reg[V][V]={0};

  for (int start = 0; start < N; start += S) {  // A,B 上各自L*S 的小块同步地在在 N 维度上移动,就是\sum 下标的 k
    __syncthreads();
    // int nthreads = blockDim.y * blockDim.x;  // how many thread in one block
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 把 CUDA 线程块内的二维线程索引（y,x） 转换成一维线性编号（tid）, (0,1)->(1,)
    
    // 循环让每个线程加载多个元素
    int nthreads = blockDim.y * blockDim.x;
    for (int idx = tid; idx < L * S; idx += nthreads) {
      int y = idx / L;
      int x = idx % L;
      
      // 加载 A_shared, 为了避免后续访问 shared memory 的 bank conflict,需要做转置
      // 牺牲了 Global Memory 的合并访问（外层），换取计算时无 Bank Conflict（内层）
      if (start + y < N && yblock * L + x < M) { //处理最后一个不完整的 L*S
        A_shared[y][x] = a[(yblock * L + x) * N + ( start + y)]; // A[yblock*L+x][start+y] (M, N) 
      }else {
        A_shared[y][x] = 0; 
      }
      
      // 加载 B_shared
      if (start + y < N && xblock * L + x < P) {
        B_shared[y][x] = b[(start + y) * P + (xblock * L + x)]; // B[start+y][xblock*L+x] (N, P)
      }else {
        B_shared[y][x] = 0;
      }
    }
    __syncthreads();

    // the following is register tiling:
    // 一个 block 下的所有 threads 完成A_shared * B_shared  = C_shared,   (L,N) * (N,L) -> (L,L)
    // 其中的一个 thread 负责其中的 V*V方块计算,在 start for loop中一次计算,完成 (V,S) * (S,V) -> (V,  V) ,相当于把 N加法得到的内积,拆成S大小的若干块. (加法结合律)
    // 把L*S拆成若干个 L*1的细长条,再拆成若干个 V*1 的细长条, V*1矩阵乘 1*V 得到 V*V
    // 只有前 L/V × L/V 个线程参与计算，多余线程跳过
    // if(threadIdx.y * V >= L && threadIdx.x * V >= L) continue;
      for(int k=0; k<S; k++){ 
        for(int i=0; i<V; i++){
          a_reg[i] = A_shared[k][i+threadIdx.y*V]; //一个 wrap,32 个 thread,考虑 0~31, blockDim.x=16, 
                                                  // 这里的 y 在一个 warp 上正好只有 0,1 两个值,只要支持两路广播就会广播
          b_reg[i] = B_shared[k][i+threadIdx.x*V]; //V=8,一个 block 下的 threads 访问bank 是 0,8,24,0,8,24...,每四个就会冲突.
          // TODO Swizzling 可以解决 bank conflict
        }
        
        for(int i=0; i<V; i++){
          for (int j=0; j<V; j++){
            C_reg[i][j] += a_reg[i]*b_reg[j];
          }
        }
      }
    
    
  } // end for start loop
  
  // 结果写入 out
  // if(threadIdx.y * V >= L || threadIdx.x * V >= L) return;
  
  for(int i=0;i<V;i++){
    for(int j=0;j<V;j++){
      int global_row=yblock*L+threadIdx.y*V+i;
      int global_col=xblock*L+threadIdx.x*V+j;
      if(global_row < M && global_col < P)out[global_row*P+global_col] = C_reg[i][j];
    }
  }


} // end for MatmulKernelTiling




void MatmulNative1D(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION

  CudaDims dim = CudaOneDim(out->size);
  MatmulKernelNative1D<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
} 


void MatmulNative2D(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  CudaDims dim = CudaTwoDim(M, P);
  MatmulKernelNative2D<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  CudaDims dim = CudaTwoDim(M, P);
  MatmulKernelTiling<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);

}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////



__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  int offset = gid*reduce_size;
  scalar_t temp=a[offset];
  for(int i=1;i<reduce_size;i++){
    scalar_t val = a[offset+i];
    if(val>temp)temp=val;
  }
  out[gid] = temp;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  scalar_t temp=0;
  for(int i=0;i<reduce_size;i++){
    temp+=a[gid*reduce_size+i];
  }
  out[gid] = temp;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = SM_TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);

  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul1d", MatmulNative1D);
  m.def("matmul2d", MatmulNative2D);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
