// src/core/OrionArray.h
#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifdef ORION_USE_CUDA
  #include <cuda_runtime.h>
#endif

namespace orion {

namespace detail {

inline void throw_if(bool cond, const std::string& msg) {
  if (cond) throw std::runtime_error(msg);
}

inline std::size_t mul_overflow_safe(std::size_t a, std::size_t b) {
  if (a == 0 || b == 0) return 0;
  if (a > (std::numeric_limits<std::size_t>::max)() / b) {
    throw std::overflow_error("OrionArray size overflow");
  }
  return a * b;
}

#ifdef ORION_USE_CUDA
inline void cuda_check(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) +
                             " at " + file + ":" + std::to_string(line));
  }
}
#define ORION_CUDA_CHECK(call) ::orion::detail::cuda_check((call), __FILE__, __LINE__)
#endif

inline void* aligned_malloc(std::size_t alignment, std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* p = nullptr;
  // posix_memalign alignment must be power-of-two and multiple of sizeof(void*)
  if (posix_memalign(&p, alignment, bytes) != 0) {
    throw std::bad_alloc();
  }
  return p;
}

inline void aligned_free(void* p) noexcept {
  std::free(p);
}

} // namespace detail

/// OrionArray<T>
/// - Host memory is always allocated upon resize()
/// - Device memory is allocated lazily upon toDevice()/fromDevice() when ORION_USE_CUDA is enabled.
/// - Layout: first dimension varies fastest (i-fast).
template <typename T>
class OrionArray
{
public:
  using value_type = T;

  OrionArray() = default;

  template <typename... Dims,
            typename = std::enable_if_t<(std::is_integral_v<Dims> && ...)>>
  explicit OrionArray(Dims... dims) {
    resize(static_cast<std::size_t>(dims)...);
  }

  OrionArray(const OrionArray&) = delete;
  OrionArray& operator=(const OrionArray&) = delete;

  OrionArray(OrionArray&& other) noexcept { move_from(std::move(other)); }
  OrionArray& operator=(OrionArray&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  ~OrionArray() { release(); }

  // -----------------------------
  // Resize / metadata
  // -----------------------------
  template <typename... Dims,
            typename = std::enable_if_t<(std::is_integral_v<Dims> && ...)>>
  void resize(Dims... dims) {
    std::vector<std::size_t> vdims{static_cast<std::size_t>(dims)...};
    resize_impl(vdims);
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t ndim() const noexcept { return dims_.size(); }
  const std::vector<std::size_t>& dims() const noexcept { return dims_; }

  std::size_t bytes() const noexcept { return size_ * sizeof(T); }

  void clear() noexcept { release(); }

  // -----------------------------
  // Host access
  // -----------------------------
  T* hostPtr() noexcept { return h_; }
  const T* hostPtr() const noexcept { return h_; }

  template <typename... Idx,
            typename = std::enable_if_t<(std::is_integral_v<Idx> && ...)>>
  T& operator()(Idx... idx) noexcept {
    return h_[linear_index(static_cast<std::size_t>(idx)...)];
  }

  template <typename... Idx,
            typename = std::enable_if_t<(std::is_integral_v<Idx> && ...)>>
  const T& operator()(Idx... idx) const noexcept {
    return h_[linear_index(static_cast<std::size_t>(idx)...)];
  }

  void fill(const T& v) {
    if (!h_ || size_ == 0) return;
    std::fill_n(h_, size_, v);
  }

  // -----------------------------
  // Device (lazy)
  // -----------------------------
#ifdef ORION_USE_CUDA
  T* devicePtr() noexcept { return d_; }
  const T* devicePtr() const noexcept { return d_; }

  bool hasDevice() const noexcept { return d_ != nullptr; }

  // Allocate device (if needed) and copy host->device
  void toDevice(cudaStream_t stream = 0) {
    ensure_device_allocated();
    ORION_CUDA_CHECK(cudaMemcpyAsync(d_, h_, bytes(), cudaMemcpyHostToDevice, stream));
  }

  // Copy device->host (requires device allocated)
  void fromDevice(cudaStream_t stream = 0) {
    detail::throw_if(d_ == nullptr, "OrionArray::fromDevice(): device buffer not allocated");
    ORION_CUDA_CHECK(cudaMemcpyAsync(h_, d_, bytes(), cudaMemcpyDeviceToHost, stream));
  }

  // Free device memory explicitly (host remains)
  void freeDevice() noexcept {
    if (d_) {
      cudaFree(d_);
      d_ = nullptr;
      d_bytes_ = 0;
    }
  }
#else
  bool hasDevice() const noexcept { return false; }
#endif

private:
  std::vector<std::size_t> dims_;
  T* h_ = nullptr;
  std::size_t size_ = 0;
  static constexpr std::size_t kAlign = 64;

#ifdef ORION_USE_CUDA
  T* d_ = nullptr;
  std::size_t d_bytes_ = 0;
#endif

  void release() noexcept {
#ifdef ORION_USE_CUDA
    freeDevice();
#endif
    if (h_) {
      detail::aligned_free(h_);
      h_ = nullptr;
    }
    dims_.clear();
    size_ = 0;
  }

  void move_from(OrionArray&& other) noexcept {
    dims_ = std::move(other.dims_);
    h_    = other.h_;
    size_ = other.size_;

    other.h_ = nullptr;
    other.size_ = 0;

#ifdef ORION_USE_CUDA
    d_ = other.d_;
    d_bytes_ = other.d_bytes_;
    other.d_ = nullptr;
    other.d_bytes_ = 0;
#endif
  }

  void resize_impl(const std::vector<std::size_t>& vdims) {
    detail::throw_if(vdims.empty(), "OrionArray::resize(): dims cannot be empty");

    std::size_t n = 1;
    for (auto d : vdims) {
      detail::throw_if(d == 0, "OrionArray::resize(): dim cannot be 0");
      n = detail::mul_overflow_safe(n, d);
    }

    dims_ = vdims;
    size_ = n;

    // re-alloc host
    if (h_) {
      detail::aligned_free(h_);
      h_ = nullptr;
    }

    const std::size_t nbytes = bytes();
    h_ = reinterpret_cast<T*>(detail::aligned_malloc(kAlign, nbytes));

    // keep semantics of vector assign(size_, T{})
    // For CFD arrays (double/float/int), memset is fine.
    std::memset(h_, 0, nbytes);

#ifdef ORION_USE_CUDA
    // IMPORTANT: per requirement, do NOT allocate device here.
    // If shape changes and device existed, free it to avoid mismatch.
    freeDevice();
#endif
  }

  template <typename... Idx>
  std::size_t linear_index(Idx... idx) const noexcept {
    constexpr std::size_t K = sizeof...(Idx);
    static_assert(K >= 1, "Need at least 1 index");
    assert(dims_.size() == K && "Index dimension mismatch");

    std::array<std::size_t, K> I{static_cast<std::size_t>(idx)...};

    for (std::size_t k = 0; k < K; ++k) {
      assert(I[k] < dims_[k] && "Index out of bounds");
    }

    // dim0 fastest:
    // offset = i0 + dim0*(i1 + dim1*(i2 + ...))
    std::size_t offset = I[0];
    std::size_t stride = dims_[0];
    for (std::size_t k = 1; k < K; ++k) {
      offset += I[k] * stride;
      stride *= dims_[k];
    }
    return offset;
  }

#ifdef ORION_USE_CUDA
  void ensure_device_allocated() {
    if (size_ == 0 || h_ == nullptr) {
      throw std::runtime_error("OrionArray::toDevice(): empty array");
    }
    const std::size_t need = bytes();
    if (d_ == nullptr) {
      ORION_CUDA_CHECK(cudaMalloc((void**)&d_, need));
      d_bytes_ = need;
    } else if (d_bytes_ != need) {
      ORION_CUDA_CHECK(cudaFree(d_));
      d_ = nullptr;
      ORION_CUDA_CHECK(cudaMalloc((void**)&d_, need));
      d_bytes_ = need;
    }
  }
#endif
};

} // namespace orion
