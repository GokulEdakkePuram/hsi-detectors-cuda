#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "hsi/sam_cuda.hpp"

namespace hsi {
namespace {

// ---------------------------------------------------------------------------
// Constant-memory staging for the target library.
//
// Every thread in a warp reads the same target element on any given band step,
// which is precisely the broadcast pattern constant memory is built for: one
// cache line serves the whole warp at register speed. 32 KB of the 64 KB
// constant bank is enough for, say, 32 targets across 256 bands.
// ---------------------------------------------------------------------------
constexpr int kMaxConstTargetElems = 8192;
constexpr int kMaxConstTargets = 64;

__constant__ float c_targets[kMaxConstTargetElems];
__constant__ float c_target_rnorm[kMaxConstTargets];

/// Pixels handled per thread by the float32 fast path: one float4 load.
constexpr int kQuad = 4;
/// Pixels handled per thread by the fp16 fast path: one 16-byte load is eight
/// halves, so the same transaction width covers twice the pixels.
constexpr int kOct = 8;

constexpr int kBlock = 256;

inline std::size_t div_up(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Baseline: one thread per pixel, scalar loads, targets from global memory.
//
// The target loop is outermost and recomputes the pixel norm on each turn.
// That is deliberate: it keeps the single-target case down to exactly one pass
// over the cube, so a T=1 measurement isolates vectorisation and constant
// memory, while T>1 shows what failing to amortise the cube read costs.
// ---------------------------------------------------------------------------
__global__ void sam_baseline_kernel(const float* __restrict__ cube, int bands,
                                    std::size_t pixels, std::size_t stride,
                                    const float* __restrict__ targets,
                                    const float* __restrict__ norms,
                                    int num_targets,
                                    float* __restrict__ out_angle,
                                    int* __restrict__ out_target) {
  const std::size_t p = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (p >= pixels) return;

  float best_cos = -2.0f;
  int best_target = -1;

  for (int t = 0; t < num_targets; ++t) {
    float dot = 0.0f;
    float pix_sq = 0.0f;
    for (int b = 0; b < bands; ++b) {
      const float v = cube[static_cast<std::size_t>(b) * stride + p];
      dot = fmaf(v, targets[static_cast<std::size_t>(t) * bands + b], dot);
      pix_sq = fmaf(v, v, pix_sq);
    }
    // A zero-norm pixel yields rsqrt->0 and therefore cosine 0, whose acos is
    // pi/2 - exactly the "no direction, no detection" answer, with no branch.
    const float rn = (pix_sq > 0.0f) ? rsqrtf(pix_sq) : 0.0f;
    const float tn = (norms[t] > 0.0f) ? (1.0f / norms[t]) : 0.0f;
    const float c = dot * rn * tn;
    if (c > best_cos) {
      best_cos = c;
      best_target = t;
    }
  }

  out_angle[p] = acosf(fminf(fmaxf(best_cos, -1.0f), 1.0f));
  if (out_target) out_target[p] = best_target;
}

// ---------------------------------------------------------------------------
// BipDirect: identical arithmetic to the baseline over a band-interleaved-by-
// pixel cube. The only thing that changes is the address arithmetic, so the
// gap between the two is a clean measurement of the coalescing penalty.
// ---------------------------------------------------------------------------
__global__ void sam_bip_kernel(const float* __restrict__ cube, int bands,
                               std::size_t pixels,
                               const float* __restrict__ targets,
                               const float* __restrict__ norms, int num_targets,
                               float* __restrict__ out_angle,
                               int* __restrict__ out_target) {
  const std::size_t p = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (p >= pixels) return;

  float best_cos = -2.0f;
  int best_target = -1;

  for (int t = 0; t < num_targets; ++t) {
    float dot = 0.0f;
    float pix_sq = 0.0f;
    const float* spectrum = cube + p * static_cast<std::size_t>(bands);
    for (int b = 0; b < bands; ++b) {
      const float v = spectrum[b];
      dot = fmaf(v, targets[static_cast<std::size_t>(t) * bands + b], dot);
      pix_sq = fmaf(v, v, pix_sq);
    }
    const float rn = (pix_sq > 0.0f) ? rsqrtf(pix_sq) : 0.0f;
    const float tn = (norms[t] > 0.0f) ? (1.0f / norms[t]) : 0.0f;
    const float c = dot * rn * tn;
    if (c > best_cos) {
      best_cos = c;
      best_target = t;
    }
  }

  out_angle[p] = acosf(fminf(fmaxf(best_cos, -1.0f), 1.0f));
  if (out_target) out_target[p] = best_target;
}

// ---------------------------------------------------------------------------
// Optimized: float4 loads, TT targets scored from one read of the cube.
//
// Between passes `out_angle` carries the running best cosine rather than an
// angle; the pass flagged `last` clamps and converts once. Ranking on cosine
// is safe because acos is monotonically decreasing, and it buys back all but
// one of the transcendentals.
// ---------------------------------------------------------------------------
template <int TT, int PPT>
__global__ void sam_opt_kernel(const float* __restrict__ cube, int bands,
                               std::size_t pixels, std::size_t stride,
                               int t_base, bool first, bool last,
                               float* __restrict__ out_angle,
                               int* __restrict__ out_target) {
  static_assert(PPT == 2 || PPT == 4 || PPT == 8, "PPT must be 2, 4 or 8");

  const std::size_t group = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t p0 = group * PPT;
  if (p0 >= pixels) return;

  float dot[TT][PPT];
#pragma unroll
  for (int t = 0; t < TT; ++t)
#pragma unroll
    for (int k = 0; k < PPT; ++k) dot[t][k] = 0.0f;

  float nrm[PPT];
#pragma unroll
  for (int k = 0; k < PPT; ++k) nrm[k] = 0.0f;

  for (int b = 0; b < bands; ++b) {
    const float* plane = cube + static_cast<std::size_t>(b) * stride;

    // One vector load per PPT pixels. The plane stride is padded to eight
    // elements, so every one of these stays aligned on every band.
    float v[PPT];
    if constexpr (PPT == 2) {
      const float2 a = reinterpret_cast<const float2*>(plane)[group];
      v[0] = a.x; v[1] = a.y;
    } else if constexpr (PPT == 4) {
      const float4 a = reinterpret_cast<const float4*>(plane)[group];
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    } else {
      const float4 a = reinterpret_cast<const float4*>(plane)[group * 2];
      const float4 c = reinterpret_cast<const float4*>(plane)[group * 2 + 1];
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
      v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
    }

#pragma unroll
    for (int k = 0; k < PPT; ++k) nrm[k] = fmaf(v[k], v[k], nrm[k]);

    // TT reads of constant memory per band step, against PPT/4 vector loads.
    // That ratio is what the measurements point at, so it is the thing PPT
    // exists to vary.
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const float r = c_targets[static_cast<std::size_t>(t_base + t) * bands + b];
#pragma unroll
      for (int k = 0; k < PPT; ++k) dot[t][k] = fmaf(v[k], r, dot[t][k]);
    }
  }

#pragma unroll
  for (int k = 0; k < PPT; ++k) {
    const std::size_t p = p0 + k;
    if (p >= pixels) continue;

    const float rn = (nrm[k] > 0.0f) ? rsqrtf(nrm[k]) : 0.0f;
    float best_cos = first ? -2.0f : out_angle[p];
    int best_target = first ? -1 : (out_target ? out_target[p] : -1);

#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const float c = dot[t][k] * rn * c_target_rnorm[t_base + t];
      if (c > best_cos) {
        best_cos = c;
        best_target = t_base + t;
      }
    }

    out_angle[p] = last ? acosf(fminf(fmaxf(best_cos, -1.0f), 1.0f)) : best_cos;
    if (out_target) out_target[p] = best_target;
  }
}

// ---------------------------------------------------------------------------
// Half: same shape as Optimized, but the cube is fp16 so one 16-byte load
// covers eight pixels. On a kernel this bandwidth-starved, halving the bytes
// is the only lever that can get near a 2x. Accumulation stays fp32 - fp16
// sums over 100+ bands lose enough mantissa to move the angle visibly.
// ---------------------------------------------------------------------------
template <int TT>
__global__ void sam_half_kernel(const __half* __restrict__ cube, int bands,
                                std::size_t pixels, std::size_t stride,
                                int t_base, bool first, bool last,
                                float* __restrict__ out_angle,
                                int* __restrict__ out_target) {
  const std::size_t oct = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t p0 = oct * kOct;
  if (p0 >= pixels) return;

  float dot[TT][kOct];
#pragma unroll
  for (int t = 0; t < TT; ++t)
#pragma unroll
    for (int k = 0; k < kOct; ++k) dot[t][k] = 0.0f;

  float nrm[kOct] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const float4* cube4 = reinterpret_cast<const float4*>(cube);
  const std::size_t stride8 = stride / kOct;

  for (int b = 0; b < bands; ++b) {
    const float4 packed = cube4[static_cast<std::size_t>(b) * stride8 + oct];
    const __half2* h2 = reinterpret_cast<const __half2*>(&packed);

    float v[kOct];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 pair = __half22float2(h2[j]);
      v[2 * j] = pair.x;
      v[2 * j + 1] = pair.y;
    }
#pragma unroll
    for (int k = 0; k < kOct; ++k) nrm[k] = fmaf(v[k], v[k], nrm[k]);

#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const float r = c_targets[static_cast<std::size_t>(t_base + t) * bands + b];
#pragma unroll
      for (int k = 0; k < kOct; ++k) dot[t][k] = fmaf(v[k], r, dot[t][k]);
    }
  }

#pragma unroll
  for (int k = 0; k < kOct; ++k) {
    const std::size_t p = p0 + k;
    if (p >= pixels) continue;

    const float rn = (nrm[k] > 0.0f) ? rsqrtf(nrm[k]) : 0.0f;
    float best_cos = first ? -2.0f : out_angle[p];
    int best_target = first ? -1 : (out_target ? out_target[p] : -1);

#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const float c = dot[t][k] * rn * c_target_rnorm[t_base + t];
      if (c > best_cos) {
        best_cos = c;
        best_target = t_base + t;
      }
    }

    out_angle[p] = last ? acosf(fminf(fmaxf(best_cos, -1.0f), 1.0f)) : best_cos;
    if (out_target) out_target[p] = best_target;
  }
}

// ---------------------------------------------------------------------------
// BIP -> BSQ transpose, staged through shared memory so both the read and the
// write are coalesced. The tile is padded by one column to keep the 32 threads
// of a warp on 32 distinct shared-memory banks during the transposed access.
// ---------------------------------------------------------------------------
constexpr int kTile = 32;

__global__ void transpose_bip_to_bsq_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            std::size_t pixels, int bands,
                                            std::size_t stride) {
  __shared__ float tile[kTile][kTile + 1];

  const std::size_t p_base = static_cast<std::size_t>(blockIdx.y) * kTile;
  const int b_base = blockIdx.x * kTile;

  // Read: threadIdx.x walks bands, which is the contiguous axis of a BIP cube.
  const std::size_t p_read = p_base + threadIdx.y;
  const int b_read = b_base + threadIdx.x;
  tile[threadIdx.y][threadIdx.x] =
      (p_read < pixels && b_read < bands)
          ? src[p_read * static_cast<std::size_t>(bands) + b_read]
          : 0.0f;

  __syncthreads();

  // Write: threadIdx.x walks pixels, the contiguous axis of a BSQ cube.
  const std::size_t p_write = p_base + threadIdx.x;
  const int b_write = b_base + threadIdx.y;
  if (p_write < pixels && b_write < bands) {
    dst[static_cast<std::size_t>(b_write) * stride + p_write] =
        tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void narrow_to_half_kernel(const float* __restrict__ src,
                                      __half* __restrict__ dst,
                                      std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) dst[i] = __float2half(src[i]);
}

}  // namespace

const char* to_string(SamVariant variant) {
  switch (variant) {
    case SamVariant::Baseline: return "baseline";
    case SamVariant::Optimized: return "optimized";
    case SamVariant::Half: return "half";
    case SamVariant::BipDirect: return "bip";
  }
  return "unknown";
}

bool parse_sam_variant(const std::string& name, SamVariant* out) {
  if (name == "baseline") { *out = SamVariant::Baseline; return true; }
  if (name == "optimized" || name == "opt") { *out = SamVariant::Optimized; return true; }
  if (name == "half" || name == "fp16") { *out = SamVariant::Half; return true; }
  if (name == "bip") { *out = SamVariant::BipDirect; return true; }
  return false;
}

std::size_t bsq_plane_stride(CubeShape shape) {
  // Round up to 8 elements: 16-byte alignment for float4 loads on the fp32
  // path, and for eight-half loads on the fp16 path.
  const std::size_t pixels = shape.pixels();
  return ((pixels + kOct - 1) / kOct) * kOct;
}

int max_constant_target_elements() { return kMaxConstTargetElems; }

cudaError_t upload_targets(const float* values, const float* norms,
                           int num_targets, int bands) {
  if (num_targets <= 0 || bands <= 0) return cudaErrorInvalidValue;
  if (num_targets > kMaxConstTargets) return cudaErrorInvalidValue;
  const std::size_t elems = static_cast<std::size_t>(num_targets) * bands;
  if (elems > kMaxConstTargetElems) return cudaErrorInvalidValue;

  // Synchronous on purpose. Both source buffers belong to the caller and may
  // be gone by the time an async copy drained, and this runs once at setup -
  // never in the per-frame path - so there is nothing to gain by deferring it.
  cudaError_t err = cudaMemcpyToSymbol(c_targets, values, elems * sizeof(float),
                                       0, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return err;

  // Store reciprocals so the kernel multiplies instead of dividing.
  float rnorm[kMaxConstTargets] = {};
  for (int t = 0; t < num_targets; ++t) {
    rnorm[t] = (norms[t] > 0.0f) ? (1.0f / norms[t]) : 0.0f;
  }
  return cudaMemcpyToSymbol(c_target_rnorm, rnorm,
                            static_cast<std::size_t>(num_targets) * sizeof(float),
                            0, cudaMemcpyHostToDevice);
}

cudaError_t launch_sam_best(SamVariant variant, const void* d_cube,
                            CubeShape shape, const float* d_targets,
                            const float* d_target_norms, int num_targets,
                            float* d_score, std::int32_t* d_target_id,
                            cudaStream_t stream, int opt_pixels_per_thread) {
  if (!shape.valid() || num_targets <= 0) return cudaErrorInvalidValue;

  const std::size_t pixels = shape.pixels();
  const std::size_t stride = bsq_plane_stride(shape);
  int* out_target = reinterpret_cast<int*>(d_target_id);

  switch (variant) {
    case SamVariant::Baseline: {
      if (!d_targets || !d_target_norms) return cudaErrorInvalidValue;
      const std::size_t blocks = div_up(pixels, kBlock);
      sam_baseline_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
          static_cast<const float*>(d_cube), shape.bands, pixels, stride,
          d_targets, d_target_norms, num_targets, d_score, out_target);
      return cudaGetLastError();
    }
    case SamVariant::BipDirect: {
      if (!d_targets || !d_target_norms) return cudaErrorInvalidValue;
      const std::size_t blocks = div_up(pixels, kBlock);
      sam_bip_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
          static_cast<const float*>(d_cube), shape.bands, pixels, d_targets,
          d_target_norms, num_targets, d_score, out_target);
      return cudaGetLastError();
    }
    case SamVariant::Optimized: {
      const int ppt = (opt_pixels_per_thread == 2 || opt_pixels_per_thread == 8)
                          ? opt_pixels_per_thread
                          : kQuad;
      const std::size_t groups = div_up(pixels, static_cast<std::size_t>(ppt));
      const std::size_t blocks = div_up(groups, kBlock);
      int done = 0;
      while (done < num_targets) {
        const int remaining = num_targets - done;
        const int tt = (remaining >= 8) ? 8 : (remaining >= 4) ? 4
                       : (remaining >= 2) ? 2 : 1;
        const bool first = (done == 0);
        const bool last = (done + tt >= num_targets);
#define HSI_LAUNCH_OPT(N, P)                                                   \
  sam_opt_kernel<N, P><<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(   \
      static_cast<const float*>(d_cube), shape.bands, pixels, stride, done,     \
      first, last, d_score, out_target)
#define HSI_LAUNCH_OPT_TT(P)                                                   \
  do {                                                                         \
    if (tt == 8) HSI_LAUNCH_OPT(8, P);                                          \
    else if (tt == 4) HSI_LAUNCH_OPT(4, P);                                     \
    else if (tt == 2) HSI_LAUNCH_OPT(2, P);                                     \
    else HSI_LAUNCH_OPT(1, P);                                                  \
  } while (0)
        if (ppt == 2) HSI_LAUNCH_OPT_TT(2);
        else if (ppt == 8) HSI_LAUNCH_OPT_TT(8);
        else HSI_LAUNCH_OPT_TT(4);
#undef HSI_LAUNCH_OPT_TT
#undef HSI_LAUNCH_OPT
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        done += tt;
      }
      return cudaSuccess;
    }
    case SamVariant::Half: {
      const std::size_t octs = div_up(pixels, kOct);
      const std::size_t blocks = div_up(octs, kBlock);
      int done = 0;
      while (done < num_targets) {
        const int remaining = num_targets - done;
        // Capped at four: eight pixels times eight targets would need 64 live
        // accumulators and the occupancy loss costs more than the extra pass.
        const int tt = (remaining >= 4) ? 4 : (remaining >= 2) ? 2 : 1;
        const bool first = (done == 0);
        const bool last = (done + tt >= num_targets);
#define HSI_LAUNCH_HALF(N)                                                    \
  sam_half_kernel<N><<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(    \
      static_cast<const __half*>(d_cube), shape.bands, pixels, stride, done,   \
      first, last, d_score, out_target)
        if (tt == 4) HSI_LAUNCH_HALF(4);
        else if (tt == 2) HSI_LAUNCH_HALF(2);
        else HSI_LAUNCH_HALF(1);
#undef HSI_LAUNCH_HALF
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        done += tt;
      }
      return cudaSuccess;
    }
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_transpose_bip_to_bsq(const float* d_src, float* d_dst,
                                        CubeShape shape, cudaStream_t stream) {
  if (!shape.valid()) return cudaErrorInvalidValue;
  const std::size_t pixels = shape.pixels();
  const std::size_t stride = bsq_plane_stride(shape);
  const dim3 block(kTile, kTile);
  const dim3 grid(static_cast<unsigned>(div_up(static_cast<std::size_t>(shape.bands), kTile)),
                  static_cast<unsigned>(div_up(pixels, kTile)));
  transpose_bip_to_bsq_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, pixels,
                                                          shape.bands, stride);
  return cudaGetLastError();
}

cudaError_t launch_narrow_to_half(const float* d_src, void* d_dst,
                                  std::size_t count, cudaStream_t stream) {
  const std::size_t blocks = div_up(count, kBlock);
  narrow_to_half_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
      d_src, static_cast<__half*>(d_dst), count);
  return cudaGetLastError();
}

}  // namespace hsi
