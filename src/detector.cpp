#include "hsi/detector.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hsi {
namespace {

/// Narrow a double buffer to float. Everything above is computed in double and
/// consumed by the kernels in float.
std::vector<float> narrow(const std::vector<double>& in) {
  std::vector<float> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) out[i] = static_cast<float>(in[i]);
  return out;
}

}  // namespace

DetectorTraits traits_of(Detector detector) {
  switch (detector) {
    case Detector::Sam:
      return {"sam", true, false, ScorePolarity::LowerIsBetter, false};
    case Detector::Rx:
      return {"rx", false, true, ScorePolarity::HigherIsBetter, true};
    case Detector::Ace:
      return {"ace", true, true, ScorePolarity::HigherIsBetter, true};
    case Detector::Cem:
      return {"cem", true, true, ScorePolarity::HigherIsBetter, false};
  }
  return {"unknown", false, false, ScorePolarity::LowerIsBetter, false};
}

const char* to_string(Detector detector) { return traits_of(detector).name; }

bool parse_detector(const std::string& name, Detector* out) {
  for (Detector candidate : {Detector::Sam, Detector::Rx, Detector::Ace, Detector::Cem}) {
    if (name == traits_of(candidate).name) {
      *out = candidate;
      return true;
    }
  }
  return false;
}

bool build_statistics(const SpectralMoments& moments,
                      const SpectralLibrary& library, double diagonal_loading,
                      DetectorStatistics* out, std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error) *error = message;
    if (out) out->valid = false;
    return false;
  };
  if (!out) return false;
  if (moments.empty()) return fail("no scene statistics accumulated yet");

  const int bands = moments.bands;
  const int targets = library.size();
  if (targets > 0 && library.bands() != bands) {
    return fail("library has " + std::to_string(library.bands()) +
                " bands but the statistics cover " + std::to_string(bands));
  }

  const std::vector<double> mean = moments.mean();

  // Covariance for RX and ACE. Factorise, then invert the factor rather than
  // the matrix: y^T S^-1 y == || L^-1 y ||^2, which halves both the per-pixel
  // arithmetic and the storage, and lets one whitening serve RX and every ACE
  // target at once.
  std::vector<double> covariance = moments.covariance(diagonal_loading);
  if (!cholesky(covariance.data(), bands)) {
    return fail("scene covariance is not positive definite at loading " +
                std::to_string(diagonal_loading) +
                "; hyperspectral bands are strongly correlated, so try a larger "
                "--loading");
  }
  std::vector<double> whitener(static_cast<std::size_t>(bands) * bands, 0.0);
  triangular_inverse_lower(covariance.data(), bands, whitener.data());

  out->bands = bands;
  out->num_targets = targets;
  out->mean = narrow(mean);
  out->whitener = narrow(whitener);
  out->target_white.assign(static_cast<std::size_t>(targets) * bands, 0.0f);
  out->target_norm2.assign(static_cast<std::size_t>(targets), 0.0f);
  out->cem_weight.assign(static_cast<std::size_t>(targets) * bands, 0.0f);

  if (targets == 0) {
    out->valid = true;
    return true;
  }

  // CEM uses the autocorrelation, not the covariance, and an uncentred pixel.
  std::vector<double> correlation = moments.correlation(diagonal_loading);
  std::vector<double> correlation_factor = correlation;
  if (!cholesky(correlation_factor.data(), bands)) {
    return fail("scene autocorrelation is not positive definite at loading " +
                std::to_string(diagonal_loading) + "; try a larger --loading");
  }

  std::vector<double> centred(static_cast<std::size_t>(bands));
  std::vector<double> whitened(static_cast<std::size_t>(bands));
  std::vector<double> solved(static_cast<std::size_t>(bands));

  for (int t = 0; t < targets; ++t) {
    const Spectrum& target = library.targets[static_cast<std::size_t>(t)];

    // ACE: whiten the centred target once. Its squared norm is
    // (d - mu)^T S^-1 (d - mu), the target half of ACE's denominator.
    for (int i = 0; i < bands; ++i) {
      centred[static_cast<std::size_t>(i)] =
          static_cast<double>(target.values[static_cast<std::size_t>(i)]) -
          mean[static_cast<std::size_t>(i)];
    }
    double norm2 = 0.0;
    for (int i = 0; i < bands; ++i) {
      double s = 0.0;
      const double* row = whitener.data() + static_cast<std::size_t>(i) * bands;
      for (int j = 0; j <= i; ++j) s += row[j] * centred[static_cast<std::size_t>(j)];
      whitened[static_cast<std::size_t>(i)] = s;
      norm2 += s * s;
    }
    for (int i = 0; i < bands; ++i) {
      out->target_white[static_cast<std::size_t>(t) * bands + i] =
          static_cast<float>(whitened[static_cast<std::size_t>(i)]);
    }
    out->target_norm2[static_cast<std::size_t>(t)] = static_cast<float>(norm2);

    // CEM: w = R^-1 d / (d^T R^-1 d), so the filter passes the target at unit
    // gain. A target orthogonal to the scene under R^-1 would divide by zero;
    // such a target cannot be detected at all, so its filter is left at zero.
    std::vector<double> raw(static_cast<std::size_t>(bands));
    for (int i = 0; i < bands; ++i) {
      raw[static_cast<std::size_t>(i)] = target.values[static_cast<std::size_t>(i)];
    }
    cholesky_solve(correlation_factor.data(), bands, raw.data(), solved.data());
    double gain = 0.0;
    for (int i = 0; i < bands; ++i) {
      gain += raw[static_cast<std::size_t>(i)] * solved[static_cast<std::size_t>(i)];
    }
    if (gain > 0.0) {
      for (int i = 0; i < bands; ++i) {
        out->cem_weight[static_cast<std::size_t>(t) * bands + i] =
            static_cast<float>(solved[static_cast<std::size_t>(i)] / gain);
      }
    }
  }

  out->valid = true;
  return true;
}

}  // namespace hsi
