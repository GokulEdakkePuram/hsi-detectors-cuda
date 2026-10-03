#pragma once

#include <cuda_runtime.h>

#include "hsi/linalg.hpp"
#include "hsi/types.hpp"

namespace hsi {

/// Device-side scene statistics accumulator.
///
/// RX, ACE and CEM need the scene mean and second moment before they can score
/// anything, and accumulating sum(x x^T) over a frame is the most arithmetic in
/// the whole pipeline: B(B+1)/2 fused multiply-adds per pixel, against 4B bytes
/// read. At 113 bands that is around 28 FLOP/byte, so unlike SAM this is a
/// compute problem.
///
/// Precision drives the interface. A single fp32 accumulator cannot hold
/// sum(x x^T) over a few hundred thousand pixels - see linalg.hpp for the
/// measurements - so the output is fp64 and every block accumulates its own
/// fp32 partial before flushing once. Accepting a reference mean to subtract is
/// the other half: it keeps the covariance subtraction from cancelling 0.1369
/// against 0.1373.

/// Elements to allocate for the second-moment accumulator.
inline std::size_t moment_outer_elements(int bands) {
  return static_cast<std::size_t>(bands) * static_cast<std::size_t>(bands);
}

/// Zero both accumulators. Must run before each accumulation pass.
cudaError_t launch_reset_moments(double* d_sum, double* d_sum_outer, int bands,
                                 cudaStream_t stream);

/// Accumulate the band means only, into `d_sum`.
///
/// Cheap - one coalesced pass with B accumulators - and it exists to provide
/// the reference mean for the pass below when no previous frame is available.
/// That is what makes the two-pass mode exact rather than merely close.
cudaError_t launch_accumulate_mean(const float* d_cube_bsq, CubeShape shape,
                                   double* d_sum, cudaStream_t stream);

/// Accumulate sum(d) into `d_sum` and the lower triangle of sum(d d^T) into
/// `d_sum_outer`, where d = x - reference.
///
/// `d_reference` is `bands` floats on the device, or null to accumulate the
/// raw moments. Null is supported for completeness but costs precision in the
/// covariance; prefer a reference from launch_accumulate_mean() or from the
/// previous frame.
///
/// Only the lower triangle of `d_sum_outer` is written. The upper triangle is
/// left untouched, and read_moments() mirrors it.
cudaError_t launch_accumulate_moments(const float* d_cube_bsq, CubeShape shape,
                                      const float* d_reference, double* d_sum,
                                      double* d_sum_outer, cudaStream_t stream);

/// Copy the accumulators back and rebuild host-side moments.
///
/// `count` is the number of pixels accumulated and `reference` the same vector
/// handed to the accumulation (or null if none was). Synchronous.
cudaError_t read_moments(const double* d_sum, const double* d_sum_outer,
                         int bands, double count, const float* reference,
                         SpectralMoments* out);

/// How many times the accumulation kernel reads the cube.
///
/// The second moment is tiled over bands, and a band plane is reloaded once per
/// band tile it appears in - so this is ceil(bands / tile). It is the figure
/// that decides which side of the roofline the accumulation lands on: at 113
/// bands and a 32-band tile the cube is read four times, which drops the
/// intensity from ~28 FLOP/byte to ~7 and makes the kernel memory-bound after
/// all. Reported so the benchmark can say so rather than imply it.
int moment_cube_reads(int bands);

}  // namespace hsi
