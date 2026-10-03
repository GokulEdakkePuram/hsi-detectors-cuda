#pragma once

#include <cstdint>
#include <vector>

#include "hsi/types.hpp"

namespace hsi {

/// Spectral Angle Mapper.
///
///     theta(x, r) = acos( <x, r> / (||x|| * ||r||) )
///
/// The angle between a pixel's spectrum and a reference spectrum, which is
/// invariant to a per-pixel multiplicative gain. That invariance is the whole
/// point: illumination, shadow and viewing geometry scale a spectrum without
/// changing its shape, so SAM keeps matching a target through them where a
/// euclidean distance would not.
///
/// Reflectance is non-negative, so angles lie in [0, pi/2]; smaller is a
/// better match. A pixel whose spectrum is all zeros has no direction at all
/// and is reported as pi/2, which never trips a sane detection threshold.

/// Largest angle SAM can return for non-negative data.
constexpr float kMaxSamAngle = 1.57079632679489661923f;  // pi/2

/// Reference implementation: every target against every pixel.
///
/// `cube_bsq`      band-sequential cube, shape.elements() floats
/// `targets`       target-major, num_targets * shape.bands floats
/// `target_norms`  L2 norm of each target, num_targets floats
/// `out_score` num_targets * shape.pixels() floats, target-major
///
/// Accumulates in double so this stays a trustworthy gold standard for the
/// GPU kernels rather than a second approximation.
void sam_cpu(const float* cube_bsq, CubeShape shape, const float* targets,
             const float* target_norms, int num_targets, float* out_score);

/// Reference implementation of the best-match reduction: for each pixel, the
/// smallest angle over all targets and which target produced it.
///
/// This is what the detector actually wants - a full per-target angle cube is
/// num_targets times the size of the output and is only useful for analysis.
void sam_best_cpu(const float* cube_bsq, CubeShape shape, const float* targets,
                  const float* target_norms, int num_targets,
                  float* out_score, std::int32_t* out_target);

/// Reference thresholding, including non-maximum suppression, in raster order.
///
/// Matches the CUDA detector's tie-breaking exactly - a plateau of equal
/// angles emits the lowest linear index and nothing else - so the two can be
/// compared element for element rather than only by count.
std::vector<Detection> detect_cpu(const float* score,
                                  const std::int32_t* target, CubeShape shape,
                                  const DetectionParams& params);

/// Mean spectrum over the pixels where `mask` is non-zero.
///
/// How a target signature is built from labelled ground truth. `mask` is
/// shape.pixels() entries in raster order. Throws if the mask selects nothing.
Spectrum mean_spectrum(const float* cube_bsq, CubeShape shape,
                       const std::uint8_t* mask, const char* name);

}  // namespace hsi
