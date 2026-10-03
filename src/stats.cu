#include <cuda_runtime.h>

#include <cstddef>

#include "hsi/sam_cuda.hpp"
#include "hsi/stats_cuda.hpp"

namespace hsi {
namespace {

/// Bands and pixels per tile of the second-moment accumulation.
///
/// 32x32 bands with 32 pixels per chunk puts both staging tiles in 8 KB of
/// shared memory and gives each of 256 threads a 2x2 register block, which is
/// the usual arrangement for an outer-product accumulation.
constexpr int kTileBands = 32;
constexpr int kTilePixels = 32;
constexpr int kStatsBlock = 256;  // 16 x 16

inline std::size_t div_up(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

__global__ void reset_moments_kernel(double* __restrict__ sum,
                                     double* __restrict__ outer, int bands) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < static_cast<std::size_t>(bands)) sum[i] = 0.0;
  const std::size_t n = static_cast<std::size_t>(bands) * bands;
  for (std::size_t k = i; k < n; k += gridDim.x * static_cast<std::size_t>(blockDim.x)) {
    outer[k] = 0.0;
  }
}

/// One block per band, reducing that band's plane. The plane is contiguous in
/// BSQ, so the read is perfectly coalesced and this is as cheap as a pass over
/// the cube can be.
__global__ void accumulate_mean_kernel(const float* __restrict__ cube,
                                       std::size_t pixels, std::size_t stride,
                                       double* __restrict__ sum) {
  __shared__ float partial[kStatsBlock];

  const int band = blockIdx.x;
  const float* plane = cube + static_cast<std::size_t>(band) * stride;

  // fp32 within the block, fp64 at the flush - the same hierarchy the second
  // moment needs, for the same reason.
  float local = 0.0f;
  for (std::size_t p = threadIdx.x; p < pixels; p += blockDim.x) local += plane[p];
  partial[threadIdx.x] = local;
  __syncthreads();

  for (int span = blockDim.x / 2; span > 0; span >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(span)) {
      partial[threadIdx.x] += partial[threadIdx.x + span];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&sum[band], static_cast<double>(partial[0]));
}

/// Decode a lower-triangular tile index k into (i, j) with j <= i.
///
/// The float square root gets within one of the answer; the two corrections
/// make it exact, which matters because an off-by-one here would silently
/// accumulate into the wrong matrix entry.
__device__ __forceinline__ void decode_tile(int k, int* i_tile, int* j_tile) {
  int i = static_cast<int>((sqrtf(8.0f * k + 1.0f) - 1.0f) * 0.5f);
  while ((i + 1) * (i + 2) / 2 <= k) ++i;
  while (i > 0 && i * (i + 1) / 2 > k) --i;
  *i_tile = i;
  *j_tile = k - i * (i + 1) / 2;
}

__global__ void accumulate_moments_kernel(const float* __restrict__ cube, int bands,
                                          std::size_t pixels, std::size_t stride,
                                          const float* __restrict__ reference,
                                          double* __restrict__ sum,
                                          double* __restrict__ outer) {
  int i_tile = 0;
  int j_tile = 0;
  decode_tile(blockIdx.y, &i_tile, &j_tile);
  const int i_base = i_tile * kTileBands;
  const int j_base = j_tile * kTileBands;

  // Padded by one column so that threads reading down a column of the tile land
  // on distinct shared-memory banks instead of all on one.
  __shared__ float si[kTileBands][kTilePixels + 1];
  __shared__ float sj[kTileBands][kTilePixels + 1];

  const int tx = threadIdx.x;  // 0..15, two j entries each
  const int ty = threadIdx.y;  // 0..15, two i entries each
  const int tid = ty * 16 + tx;

  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  float row_sum[2] = {0.0f, 0.0f};

  const std::size_t span = static_cast<std::size_t>(gridDim.x) * kTilePixels;
  for (std::size_t chunk = static_cast<std::size_t>(blockIdx.x) * kTilePixels;
       chunk < pixels; chunk += span) {
    // Stage both band tiles. Consecutive threads take consecutive pixels, so
    // each warp reads 32 contiguous floats out of a band plane.
#pragma unroll
    for (int rep = 0; rep < (kTileBands * kTilePixels) / kStatsBlock; ++rep) {
      const int flat = rep * kStatsBlock + tid;
      const int r = flat / kTilePixels;
      const int c = flat % kTilePixels;
      const std::size_t p = chunk + c;
      const int bi = i_base + r;
      const int bj = j_base + r;
      const bool have = p < pixels;
      si[r][c] = (have && bi < bands)
                     ? cube[static_cast<std::size_t>(bi) * stride + p] -
                           (reference ? reference[bi] : 0.0f)
                     : 0.0f;
      sj[r][c] = (have && bj < bands)
                     ? cube[static_cast<std::size_t>(bj) * stride + p] -
                           (reference ? reference[bj] : 0.0f)
                     : 0.0f;
    }
    __syncthreads();

#pragma unroll 8
    for (int p = 0; p < kTilePixels; ++p) {
      const float a0 = si[ty * 2 + 0][p];
      const float a1 = si[ty * 2 + 1][p];
      const float b0 = sj[tx * 2 + 0][p];
      const float b1 = sj[tx * 2 + 1][p];
      acc[0][0] = fmaf(a0, b0, acc[0][0]);
      acc[0][1] = fmaf(a0, b1, acc[0][1]);
      acc[1][0] = fmaf(a1, b0, acc[1][0]);
      acc[1][1] = fmaf(a1, b1, acc[1][1]);
      // The first tile column also carries the first moment, so sum(d) comes
      // free rather than costing another pass over the cube.
      if (j_tile == 0 && tx == 0) {
        row_sum[0] += a0;
        row_sum[1] += a1;
      }
    }
    __syncthreads();
  }

  // One flush per block, in fp64. Zero accumulators are skipped: on a diagonal
  // tile half the register block falls above the diagonal and is discarded.
#pragma unroll
  for (int a = 0; a < 2; ++a) {
    const int i = i_base + ty * 2 + a;
    if (i >= bands) continue;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const int j = j_base + tx * 2 + b;
      if (j >= bands || j > i) continue;
      if (acc[a][b] != 0.0f) {
        atomicAdd(&outer[static_cast<std::size_t>(i) * bands + j],
                  static_cast<double>(acc[a][b]));
      }
    }
  }
  if (j_tile == 0 && tx == 0) {
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int i = i_base + ty * 2 + a;
      if (i < bands && row_sum[a] != 0.0f) {
        atomicAdd(&sum[i], static_cast<double>(row_sum[a]));
      }
    }
  }
}

}  // namespace

int moment_cube_reads(int bands) {
  if (bands <= 0) return 0;
  return static_cast<int>(div_up(static_cast<std::size_t>(bands), kTileBands));
}

cudaError_t launch_reset_moments(double* d_sum, double* d_sum_outer, int bands,
                                 cudaStream_t stream) {
  if (!d_sum || !d_sum_outer || bands <= 0) return cudaErrorInvalidValue;
  const std::size_t n = static_cast<std::size_t>(bands) * bands;
  const unsigned blocks = static_cast<unsigned>(div_up(n, kStatsBlock));
  reset_moments_kernel<<<blocks, kStatsBlock, 0, stream>>>(d_sum, d_sum_outer, bands);
  return cudaGetLastError();
}

cudaError_t launch_accumulate_mean(const float* d_cube_bsq, CubeShape shape,
                                   double* d_sum, cudaStream_t stream) {
  if (!d_cube_bsq || !d_sum || !shape.valid()) return cudaErrorInvalidValue;
  accumulate_mean_kernel<<<static_cast<unsigned>(shape.bands), kStatsBlock, 0, stream>>>(
      d_cube_bsq, shape.pixels(), bsq_plane_stride(shape), d_sum);
  return cudaGetLastError();
}

cudaError_t launch_accumulate_moments(const float* d_cube_bsq, CubeShape shape,
                                      const float* d_reference, double* d_sum,
                                      double* d_sum_outer, cudaStream_t stream) {
  if (!d_cube_bsq || !d_sum || !d_sum_outer || !shape.valid()) {
    return cudaErrorInvalidValue;
  }
  const int tiles = moment_cube_reads(shape.bands);
  const int pairs = tiles * (tiles + 1) / 2;

  // Enough pixel groups to keep the machine busy without turning the flush
  // into a contention problem: every block adds its partial to the same few
  // thousand addresses, so more blocks means more serialised atomics.
  const std::size_t chunks = div_up(shape.pixels(), kTilePixels);
  int groups = static_cast<int>(512 / (pairs > 0 ? pairs : 1));
  if (groups < 1) groups = 1;
  if (static_cast<std::size_t>(groups) > chunks) groups = static_cast<int>(chunks);

  const dim3 block(16, 16);
  const dim3 grid(static_cast<unsigned>(groups), static_cast<unsigned>(pairs));
  accumulate_moments_kernel<<<grid, block, 0, stream>>>(
      d_cube_bsq, shape.bands, shape.pixels(), bsq_plane_stride(shape),
      d_reference, d_sum, d_sum_outer);
  return cudaGetLastError();
}

cudaError_t read_moments(const double* d_sum, const double* d_sum_outer,
                         int bands, double count, const float* reference,
                         SpectralMoments* out) {
  if (!d_sum || !d_sum_outer || !out || bands <= 0) return cudaErrorInvalidValue;

  std::vector<double> sum(static_cast<std::size_t>(bands));
  std::vector<double> outer(moment_outer_elements(bands));
  cudaError_t err = cudaMemcpy(sum.data(), d_sum, sum.size() * sizeof(double),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;
  err = cudaMemcpy(outer.data(), d_sum_outer, outer.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;

  if (reference) {
    std::vector<double> shift(static_cast<std::size_t>(bands));
    for (int i = 0; i < bands; ++i) shift[static_cast<std::size_t>(i)] = reference[i];
    moments_from_shifted(sum.data(), outer.data(), count, shift.data(), bands, out);
  } else {
    out->reset(bands);
    out->count = count;
    out->sum = sum;
    // Only the lower triangle was written; mirror it so the host sees a full
    // symmetric matrix the way SpectralMoments::add would have left it.
    for (int i = 0; i < bands; ++i) {
      for (int j = 0; j <= i; ++j) {
        out->sum_outer[static_cast<std::size_t>(i) * bands + j] =
            outer[static_cast<std::size_t>(i) * bands + j];
      }
    }
  }
  return cudaSuccess;
}

}  // namespace hsi
