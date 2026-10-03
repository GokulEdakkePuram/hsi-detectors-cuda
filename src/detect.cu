#include <cuda_runtime.h>

#include <cstddef>

#include "hsi/detect_cuda.hpp"

namespace hsi {
namespace {

constexpr int kBlock = 256;
constexpr unsigned kFullMask = 0xffffffffu;

/// Is this pixel the best match in its neighbourhood?
///
/// Ties are broken by linear index so a plateau of equal angles emits exactly
/// one detection instead of all of them.
__device__ bool is_local_best(const float* __restrict__ angle, int x, int y,
                              int width, int height, int radius, float self) {
  const std::size_t self_index = static_cast<std::size_t>(y) * width + x;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int ny = y + dy;
    if (ny < 0 || ny >= height) continue;
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) continue;
      const int nx = x + dx;
      if (nx < 0 || nx >= width) continue;
      const std::size_t n = static_cast<std::size_t>(ny) * width + nx;
      const float other = angle[n];
      if (other < self) return false;
      if (other == self && n < self_index) return false;
    }
  }
  return true;
}

/// Threshold and compact, one atomic per warp rather than one per detection.
///
/// A frame over a busy scene can put tens of thousands of pixels through the
/// threshold at once. Having every one of them hit the same counter serialises
/// the tail of the kernel, so the warp elects a leader, reserves the whole
/// warp's worth of slots in a single atomicAdd, and each lane writes at its
/// own rank within that reservation.
__global__ void detect_kernel(const float* __restrict__ angle,
                              const int* __restrict__ target, int width,
                              int height, float threshold, int radius,
                              bool higher_is_better, int capacity,
                              Detection* __restrict__ out,
                              unsigned int* __restrict__ count) {
  const std::size_t p = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t pixels = static_cast<std::size_t>(width) * height;

  // No early return: every lane has to reach the ballot below, including the
  // ones past the end of the frame.
  bool pass = false;
  Detection det;
  if (p < pixels) {
    const float a = angle[p];
    if (a <= threshold) {
      const int x = static_cast<int>(p % width);
      const int y = static_cast<int>(p / width);
      if (radius <= 0 || is_local_best(angle, x, y, width, height, radius, a)) {
        pass = true;
        det.x = x;
        det.y = y;
        det.target = target ? target[p] : 0;
        det.score = a;
      }
    }
  }

  const unsigned mask = __ballot_sync(kFullMask, pass);
  if (mask == 0) return;

  const int lane = threadIdx.x & 31;
  const int leader = __ffs(mask) - 1;
  unsigned base = 0;
  if (lane == leader) base = atomicAdd(count, __popc(mask));
  base = __shfl_sync(kFullMask, base, leader);

  if (pass) {
    const unsigned rank = __popc(mask & ((1u << lane) - 1u));
    const unsigned slot = base + rank;
    // Past capacity the detection still counts, so the caller can see the
    // frame saturated, but nothing is written outside the buffer.
    if (slot < static_cast<unsigned>(capacity)) out[slot] = det;
  }
}

}  // namespace

cudaError_t launch_detect(const float* d_score,
                          const std::int32_t* d_target_id, CubeShape shape,
                          DetectionParams params, Detection* d_out,
                          unsigned int* d_count, cudaStream_t stream) {
  if (!shape.valid() || !d_score || !d_out || !d_count) {
    return cudaErrorInvalidValue;
  }
  const std::size_t pixels = shape.pixels();
  const unsigned blocks = static_cast<unsigned>((pixels + kBlock - 1) / kBlock);
  detect_kernel<<<blocks, kBlock, 0, stream>>>(
      d_score, reinterpret_cast<const int*>(d_target_id), shape.width,
      shape.height, params.threshold, params.nms_radius,
      params.polarity == ScorePolarity::HigherIsBetter, params.max_detections,
      d_out, d_count);
  return cudaGetLastError();
}

}  // namespace hsi
