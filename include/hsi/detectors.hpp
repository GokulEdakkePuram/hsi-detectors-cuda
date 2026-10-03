#pragma once

#include <cstdint>

#include "hsi/detector.hpp"
#include "hsi/types.hpp"

namespace hsi {

/// Reference implementations of the second-order detectors.
///
/// These are the gold standard the CUDA kernels get compared against, so they
/// accumulate in double throughout. They are also deliberately written in the
/// same algebraic form the kernels will use - whitening rather than inverting -
/// because a reference that took a different route would be testing two
/// implementations of two different formulas.

/// Accumulate scene moments over a whole frame.
///
/// This is the two-pass reference: read the cube once to build the statistics,
/// then again to score. Raw moments are accumulated in double, which carries
/// enough mantissa that forming the covariance by subtraction afterwards is
/// still accurate. The device-side accumulator cannot do that and needs a
/// different arrangement.
void accumulate_moments_cpu(const float* cube_bsq, CubeShape shape,
                            SpectralMoments* moments);

/// RX: squared Mahalanobis distance of each pixel from the scene background.
///
///     RX(x) = || L^-1 (x - mu) ||^2
///
/// Higher is more anomalous. Under a Gaussian background this is distributed
/// approximately chi-squared with `bands` degrees of freedom, which is what
/// lets a threshold be set from a false-alarm rate instead of by hand.
void rx_cpu(const float* cube_bsq, CubeShape shape,
            const DetectorStatistics& stats, float* out_score);

/// ACE, reduced over targets: the squared cosine between whitened pixel and
/// whitened target.
///
///     ACE(x) = (z . u)^2 / (||u||^2 ||z||^2),   z = L^-1 (x - mu)
///
/// Lands in [0, 1]; higher is better. `out_rx` optionally receives ||z||^2,
/// which is the RX statistic for the same pixel - the whitening is shared, so
/// getting both costs nothing extra.
void ace_best_cpu(const float* cube_bsq, CubeShape shape,
                  const DetectorStatistics& stats, float* out_score,
                  std::int32_t* out_target, float* out_rx = nullptr);

/// CEM, reduced over targets: a linear filter per target, applied to the
/// uncentred pixel. Higher is better, and equals 1 at a pure target pixel.
void cem_best_cpu(const float* cube_bsq, CubeShape shape,
                  const DetectorStatistics& stats, float* out_score,
                  std::int32_t* out_target);

/// Threshold for a given per-pixel false-alarm rate under the chi-squared
/// approximation to RX with `bands` degrees of freedom.
///
/// Only as good as the Gaussian-background assumption, which real scenes
/// violate - but it turns "what threshold?" into "how many false alarms per
/// frame can I tolerate?", which is a question an operator can answer.
double rx_threshold_for_false_alarm_rate(int bands, double rate);

}  // namespace hsi
