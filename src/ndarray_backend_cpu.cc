#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    // posix_memalign: POSIX标准的内存分配函数, 专门创建对齐内存
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}


int indice_addone(std::vector<int32_t>& cur_indice, const std::vector<int32_t>& shape){
  int ndim = shape.size();
  int i = ndim - 1; // 最末位
  cur_indice[i] += 1;
  while (i >= 0 && cur_indice[i] >= shape[i]) { //需要进位了且还没突破最高位(0), i>=0其实被下面的 if 拦截了.
    // 进位(carry):低位设置成 1,左边的高位+1
    cur_indice[i] = 0;
    i--;
    if (i >= 0) {cur_indice[i] += 1;} 
    else {return 0;} //已经是最高位无法进位了,完成了遍历.
  }
  return 1;
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  // 遍历 indices，从 a[a_idx] 复制到 out[out_idx++]; 
  // non-compact -> compact, strides 和 offset 描述的是 a,所以有
  // a_idx = offset + ∑_d(indices[d] * strides[d])
  std::vector<int32_t> indices(shape.size());
  size_t out_idx = 0;
  
  do {
    // Calculate source index from multi-dimensional indices
    size_t a_idx = offset;
    for (size_t d = 0; d < shape.size(); d++) {
      a_idx += indices[d] * strides[d];
    }
    
    // Copy element to compact output
    out->ptr[out_idx] = a.ptr[a_idx];
    out_idx++;
  } while (indice_addone(indices, shape));
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  // 遍历 indices，从 a[a_idx] 复制到 out[dst_idx];  
  // compact -> non-compact, strides 和 offset 描述的是 out,所以有
  // dst_idx = offset + ∑_d(indices[d] * strides[d])
  std::vector<int32_t> indices(shape.size());
  size_t a_idx = 0;
  
  do {
    // Calculate destination index from multi-dimensional indices
    size_t dst_idx = offset;
    for (size_t d = 0; d < shape.size(); d++) {
      dst_idx += indices[d] * strides[d];
    }
    
    // Copy element from compact source to non-compact destination
    out->ptr[dst_idx] = a.ptr[a_idx];
    a_idx++;
  } while (indice_addone(indices, shape));
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  // Fill non-compact array with scalar value
  std::vector<int32_t> indices(shape.size());
  
  do {
    // Calculate destination index from multi-dimensional indices
    size_t dst_idx = offset;
    for (size_t d = 0; d < shape.size(); d++) {
      dst_idx += indices[d] * strides[d];
    }
    
    // Set scalar value
    out->ptr[dst_idx] = val;
  } while (indice_addone(indices, shape));
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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

// Template for element-wise binary operations
template<typename BinaryOp> 
void EwiseBinaryOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, BinaryOp op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

// Template for scalar binary operations
template<typename BinaryOp>
void ScalarBinaryOp(const AlignedArray& a, scalar_t val, AlignedArray* out, BinaryOp op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}


void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x * y; });
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x / y; });
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x / y; });
}


void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) {return std::pow(x, y);});
}



 void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y){
      return ( x > y ) ? x : y ;
    });
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y){
      return ( x > y ) ? x : y ;
    });
}

 void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    EwiseBinaryOp(a, b, out,[](scalar_t x, scalar_t y) { return x==y; });
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x==y; });
}


 void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    // Gerater or equal
    EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x>=y; });
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    // Gerater or equal
    ScalarBinaryOp(a, val, out,[](scalar_t x, scalar_t y) { return x>=y; });
}

// Template for element-wise unary operations
template<typename BinaryOp> 
void EwiseUnaryOp(const AlignedArray& a, AlignedArray* out, BinaryOp op) {
    for (size_t i = 0; i < a.size; i++) {
      out->ptr[i] = op(a.ptr[i]);
    }
}

 void EwiseLog(const AlignedArray& a,  AlignedArray* out) {
    EwiseUnaryOp(a, out, [](scalar_t x){return std::log(x);});
}

 void EwiseExp(const AlignedArray& a,  AlignedArray* out) {
    EwiseUnaryOp(a, out, [](scalar_t x){return std::exp(x);});
}

 void EwiseTanh(const AlignedArray& a,  AlignedArray* out) {
    EwiseUnaryOp(a, out, [](scalar_t x){return std::tanh(x);});
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  // v1, ijk
  // for(int i=0; i<m; i++){
  //   for(int j=0; j<p; j++){
  //     out->ptr[i*p+j] = 0.0f; // 谁需要 0，谁负责初始化

  //     for(int k=0; k<n; k++){
  //         out->ptr[i*p+j] += a.ptr[i*n+k]*b.ptr[k*p+j]; //ijk 中b 的访问不连续,破坏了 cache locality
  //     }
  //   }
  // }


  // v2 i-k-j优化
  for(int i=0; i<m; i++){
    for(int j=0; j<p; j++){
      out->ptr[i*p+j] = 0.0f; // 谁需要 0，谁负责初始化
    }
    for(int k=0; k<n; k++){
      scalar_t temp = a.ptr[i*n+k];
      for(int j=0; j<p; j++){
            out->ptr[i*p+j] += temp*b.ptr[k*p+j];
        }
    }
  }
  /// END SOLUTION
}


inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for(int i=0; i<TILE; i++){
    for(int k=0; k<TILE; k++){
      float temp = a[i*TILE+k];
      for(int j=0; j<TILE; j++){
            out[i*TILE+j] += temp*b[k*TILE+j];
        }
    }
  }  
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE;  (m,n) i,k
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE;  (n,p) k,j
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to; (m,p) i,j
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  size_t TT = TILE*TILE;
  for(int i= 0; i<m/TILE;i++){
    for(int j= 0; j<p/TILE;j++){
          float* out_tile = out->ptr + (i*p/TILE+j)*TT;
          for(int t=0; t<TT;t++) out_tile[t]=0.0f;
          for(int k=0; k<n/TILE;k++){
            AlignedDot(a.ptr + (i*n/TILE+k)*TT,b.ptr + (k*p/TILE+j)*TT , out_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over, last dimension size of the view
   */

  /// BEGIN SOLUTION
  for(int i = 0; i<out->size;i++){
    float res=a.ptr[i*reduce_size];
    for(int j = 0; j<reduce_size;j++){
      res = std::max(res, a.ptr[j+i*reduce_size]);
    }
    out->ptr[i] = res;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for(int i = 0; i<out->size;i++){
    float res=0.0f;
    for(int j = 0; j<reduce_size;j++){
      res+=a.ptr[j+i*reduce_size];
    }
    out->ptr[i] = res;
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
