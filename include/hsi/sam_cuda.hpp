#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "hsi/types.hpp"

namespace hsi {

/// Which SAM kernel to launch.
///
/// SAM moves `bands * 4` bytes per pixel and does about two FLOPs per band per
/// target, so at one target the arithmetic intensity is 0.5 FLOP/byte and the
/// kernel is pinned to DRAM bandwidth on every GPU worth running it on. Every
/// variant here is therefore an experiment in moving fewer bytes, or in moving
/// the same bytes more efficiently - not in doing less maths.
enum class SamVariant {
  /// One thread per pixel, scalar loads, targets read from global memory.
  /// The obvious implementation, and the baseline the rest are measured against.
  Baseline,

  /// Four pixels per thread via float4, targets broadcast from constant
  /// memory, several targets scored per pass out of one read of the cube.
  Optimized,

  /// As Optimized, but the cube is stored as fp16 and widened to fp32 in
  /// registers. Halves DRAM traffic, which on a bandwidth-bound kernel is the
  /// only change that can approach a 2x.
  Half,

  /// Baseline structure over a band-interleaved-by-pixel cube, so the cost of
  /// getting the layout wrong can be measured rather than asserted.
  BipDirect,
};

const char* to_string(SamVariant variant);
bool parse_sam_variant(const std::string& name, SamVariant* out);

/// Plane stride, in elements, of a BSQ cube living on the device.
///
/// Device cubes are padded so every band plane starts 16-byte aligned, which
/// is what lets the fast kernels issue float4 loads on band planes other than
/// the first. Host cubes stay tightly packed, so the upload is a 2D copy from
/// stride `shape.pixels()` to this stride.
std::size_t bsq_plane_stride(CubeShape shape);

/// Elements to allocate for a device BSQ cube of this shape.
inline std::size_t bsq_device_elements(CubeShape shape) {
  return bsq_plane_stride(shape) * static_cast<std::size_t>(shape.bands);
}

/// Largest target library that fits the constant-memory staging buffer.
/// Libraries beyond this are scored in several passes over the cube, which
/// costs a re-read and so is worth avoiding.
int max_constant_target_elements();

/// Upload a target library to constant memory for the Optimized and Half
/// kernels. `values` is target-major (num_targets * bands) and `norms` holds
/// the precomputed L2 norm of each target.
///
/// Synchronous, and takes no stream: it copies from caller-owned host memory
/// that an async copy could outlive, and it belongs to setup rather than the
/// per-frame path.
///
/// Returns cudaSuccess, or cudaErrorInvalidValue if the library is larger than
/// max_constant_target_elements().
cudaError_t upload_targets(const float* values, const float* norms,
                           int num_targets, int bands);

/// Best-match SAM over a whole frame.
///
/// `d_cube`         device cube. BSQ float32, except: BIP float32 for
///                  BipDirect, and BSQ fp16 (`__half`) for Half.
/// `d_targets`      target-major float32, only read by Baseline and BipDirect;
///                  the other variants take their targets from constant memory
///                  via upload_targets().
/// `d_target_norms` per-target L2 norms, same proviso.
/// `d_score`    out, shape.pixels() floats: smallest angle at that pixel.
/// `d_target_id`    out, shape.pixels() int32: which target won. May be null.
///
/// A single acosf runs per pixel, not per target: the kernel ranks targets on
/// cosine, which is monotonically decreasing in the angle, and converts once
/// at the end.
/// `opt_pixels_per_thread` selects how many pixels one thread of the Optimized
/// kernel handles: 2, 4 (the default) or 8. It exists because measurements on
/// sm_86 put the TT=8 bandwidth cliff on the ratio of constant-memory reads to
/// vector loads per band step, not on register pressure, and this is the knob
/// that varies that ratio. Ignored by the other variants.
cudaError_t launch_sam_best(SamVariant variant, const void* d_cube,
                            CubeShape shape, const float* d_targets,
                            const float* d_target_norms, int num_targets,
                            float* d_score, std::int32_t* d_target_id,
                            cudaStream_t stream,
                            int opt_pixels_per_thread = 0);

/// Transpose a BIP cube into BSQ on the device.
///
/// The live camera path receives BIP and the fast kernels want BSQ, so the
/// transpose has to happen somewhere. Doing it on the GPU with a tiled,
/// shared-memory staging pass is far cheaper than doing it on the Orin's CPU
/// cores, which would otherwise become the pipeline's bottleneck.
cudaError_t launch_transpose_bip_to_bsq(const float* d_src, float* d_dst,
                                        CubeShape shape, cudaStream_t stream);

/// Narrow a BSQ float32 cube to fp16 in place of a copy, for the Half variant.
cudaError_t launch_narrow_to_half(const float* d_src, void* d_dst,
                                  std::size_t count, cudaStream_t stream);

}  // namespace hsi
