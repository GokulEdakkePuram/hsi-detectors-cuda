#pragma once

#include <string>
#include <vector>

#include "hsi/linalg.hpp"
#include "hsi/types.hpp"

namespace hsi {

/// The detectors this project implements.
///
/// They split cleanly by arithmetic intensity, which is the whole reason for
/// having more than one. Per pixel, over B bands:
///
///   Sam, Cem   ~2B FLOP against 4B bytes   ->  0.5 FLOP/byte, DRAM-bound
///   Rx, Ace    ~B^2 FLOP against 4B bytes  ->  B/4 FLOP/byte, compute-bound
///
/// At 113 bands that is roughly 28 FLOP/byte for the second group, against a
/// ridge point near 38 on an RTX 3090 and near 26 on an AGX Orin - so the two
/// groups sit on opposite sides of the roofline and want opposite
/// optimisations.
enum class Detector {
  /// Spectral Angle Mapper. Angle between pixel and target, invariant to a
  /// per-pixel gain. Needs targets, no scene statistics. Lower is better.
  Sam,

  /// Reed-Xiaoli anomaly detector. Squared Mahalanobis distance of a pixel
  /// from the scene background. Needs scene statistics, no targets. Higher is
  /// more anomalous.
  Rx,

  /// Adaptive Cosine Estimator. The angle between whitened pixel and whitened
  /// target, so it is SAM measured in the background's own metric. Needs
  /// both. Lands in [0, 1]; higher is better.
  Ace,

  /// Constrained Energy Minimization. A linear filter passing the target at
  /// unit gain while minimising total output energy. Needs both, but the
  /// autocorrelation rather than the covariance. Higher is better.
  Cem,
};

struct DetectorTraits {
  const char* name;
  bool needs_targets;
  bool needs_statistics;
  ScorePolarity polarity;
  /// True when per-pixel cost grows with the square of the band count.
  bool compute_bound;
};

DetectorTraits traits_of(Detector detector);
const char* to_string(Detector detector);
bool parse_detector(const std::string& name, Detector* out);

/// Scene statistics reduced to exactly what the kernels index.
///
/// Derived once per frame on the host in double, then narrowed to float. All of
/// it is small: at 113 bands the whitener is 51 KB and the rest is kilobytes,
/// against a 160 MB cube.
struct DetectorStatistics {
  int bands = 0;
  int num_targets = 0;

  /// Scene mean, `bands` entries.
  std::vector<float> mean;

  /// L^-1 where covariance = L L^T. Row-major bands*bands, lower triangular
  /// with the upper triangle zeroed. Whitening a centred pixel with this and
  /// taking the squared norm gives RX.
  std::vector<float> whitener;

  /// L^-1 (d - mean) per target, num_targets * bands, target-major.
  std::vector<float> target_white;

  /// || L^-1 (d - mean) ||^2 per target - the target half of ACE's denominator.
  std::vector<float> target_norm2;

  /// R^-1 d / (d^T R^-1 d) per target, num_targets * bands, target-major.
  /// CEM's filter, applied directly to the uncentred pixel.
  std::vector<float> cem_weight;

  bool valid = false;
};

/// Build the statistics from accumulated moments and a target library.
///
/// `diagonal_loading` is added to the diagonal of both the covariance and the
/// autocorrelation before factorising. It is required in practice rather than
/// optional: hyperspectral bands are heavily correlated, so an unloaded scene
/// covariance is close to singular and inverting it turns sensor noise into
/// detections. Returns false and fills `error` when the factorisation fails,
/// which means the loading was too small.
///
/// `library` may be empty for Rx, which needs no targets.
bool build_statistics(const SpectralMoments& moments,
                      const SpectralLibrary& library, double diagonal_loading,
                      DetectorStatistics* out, std::string* error);

}  // namespace hsi
