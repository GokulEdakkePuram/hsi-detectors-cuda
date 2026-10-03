// The whitened formulation the RX/ACE/CEM kernels will use, checked against
// the textbook definitions it is supposed to be algebraically equal to.
//
// This is the test that matters most for the new detectors: the kernels never
// see a covariance or its inverse, only L^-1 and some precomputed per-target
// vectors. If that reduction is wrong, three detectors are wrong together and
// every one of them still produces plausible-looking numbers.

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "hsi/detector.hpp"
#include "hsi/linalg.hpp"

using namespace hsi;

namespace {

/// A scene with correlated bands, which is what makes the covariance
/// non-trivial and the whitening worth doing.
SpectralMoments make_scene(int bands, int pixels, std::mt19937& rng,
                           std::vector<std::vector<float>>* spectra) {
  std::normal_distribution<float> noise(0.0f, 0.05f);
  SpectralMoments moments;
  moments.reset(bands);
  for (int p = 0; p < pixels; ++p) {
    // A smooth ramp plus two shared factors, so neighbouring bands correlate
    // the way real reflectance does.
    const float a = noise(rng) * 4.0f;
    const float b = noise(rng) * 2.0f;
    std::vector<float> x(static_cast<std::size_t>(bands));
    for (int i = 0; i < bands; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(bands);
      x[static_cast<std::size_t>(i)] =
          0.3f + 0.2f * t + a * t * (1.0f - t) + b * std::sin(3.0f * t) + noise(rng);
    }
    moments.add(x.data());
    if (spectra) spectra->push_back(x);
  }
  return moments;
}

SpectralLibrary make_library(int bands) {
  SpectralLibrary library;
  for (int t = 0; t < 3; ++t) {
    Spectrum s;
    s.name = "t" + std::to_string(t);
    s.values.resize(static_cast<std::size_t>(bands));
    for (int i = 0; i < bands; ++i) {
      const float u = static_cast<float>(i) / static_cast<float>(bands);
      s.values[static_cast<std::size_t>(i)] =
          0.2f + 0.5f * std::sin((t + 1) * 2.0f * u) * std::sin((t + 1) * 2.0f * u);
    }
    library.targets.push_back(std::move(s));
  }
  return library;
}

void test_traits_are_consistent() {
  CHECK(!traits_of(Detector::Sam).needs_statistics);
  CHECK(traits_of(Detector::Sam).needs_targets);
  CHECK(traits_of(Detector::Rx).needs_statistics);
  CHECK(!traits_of(Detector::Rx).needs_targets);
  CHECK(traits_of(Detector::Ace).needs_statistics && traits_of(Detector::Ace).needs_targets);
  CHECK(traits_of(Detector::Cem).needs_statistics && traits_of(Detector::Cem).needs_targets);

  // Polarity: SAM is an angle, the rest are responses.
  CHECK(traits_of(Detector::Sam).polarity == ScorePolarity::LowerIsBetter);
  CHECK(traits_of(Detector::Rx).polarity == ScorePolarity::HigherIsBetter);
  CHECK(traits_of(Detector::Ace).polarity == ScorePolarity::HigherIsBetter);
  CHECK(traits_of(Detector::Cem).polarity == ScorePolarity::HigherIsBetter);

  // Only the quadratic-form detectors are compute-bound.
  CHECK(traits_of(Detector::Rx).compute_bound && traits_of(Detector::Ace).compute_bound);
  CHECK(!traits_of(Detector::Sam).compute_bound && !traits_of(Detector::Cem).compute_bound);

  Detector parsed = Detector::Sam;
  CHECK(parse_detector("rx", &parsed) && parsed == Detector::Rx);
  CHECK(parse_detector("ace", &parsed) && parsed == Detector::Ace);
  CHECK(parse_detector("cem", &parsed) && parsed == Detector::Cem);
  CHECK(parse_detector("sam", &parsed) && parsed == Detector::Sam);
  CHECK(!parse_detector("nope", &parsed));
  CHECK(std::string(to_string(Detector::Rx)) == "rx");
}

void test_whitener_reproduces_the_mahalanobis_distance() {
  const int bands = 14;
  std::mt19937 rng(101);
  std::vector<std::vector<float>> spectra;
  const SpectralMoments moments = make_scene(bands, 400, rng, &spectra);

  const double loading = 1e-5;
  DetectorStatistics stats;
  std::string error;
  CHECK(build_statistics(moments, SpectralLibrary{}, loading, &stats, &error));
  if (!stats.valid) {
    std::fprintf(stderr, "  build_statistics: %s\n", error.c_str());
    return;
  }

  // The definition, computed independently: S^-1 then y^T S^-1 y.
  std::vector<double> covariance = moments.covariance(loading);
  std::vector<double> inverse;
  CHECK(invert_spd(covariance, bands, &inverse));
  const std::vector<double> mean = moments.mean();

  double worst_relative = 0.0;
  for (const std::vector<float>& x : spectra) {
    std::vector<double> y(static_cast<std::size_t>(bands));
    for (int i = 0; i < bands; ++i) {
      y[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - mean[static_cast<std::size_t>(i)];
    }
    const double want = quadratic_form(inverse.data(), y.data(), bands);

    // What the kernel will do: whiten, then take the squared norm.
    double got = 0.0;
    for (int i = 0; i < bands; ++i) {
      double s = 0.0;
      const float* row = stats.whitener.data() + static_cast<std::size_t>(i) * bands;
      for (int j = 0; j <= i; ++j) s += row[j] * y[static_cast<std::size_t>(j)];
      got += s * s;
    }
    worst_relative = std::max(worst_relative, std::fabs(got - want) / std::max(1.0, want));
  }
  // float whitener against a double inverse, so this is a float-precision bound.
  CHECK(worst_relative < 1e-4);
}

void test_whitener_is_lower_triangular() {
  const int bands = 9;
  std::mt19937 rng(7);
  const SpectralMoments moments = make_scene(bands, 200, rng, nullptr);
  DetectorStatistics stats;
  std::string error;
  CHECK(build_statistics(moments, SpectralLibrary{}, 1e-5, &stats, &error));

  bool upper_zero = true;
  for (int i = 0; i < bands; ++i)
    for (int j = i + 1; j < bands; ++j)
      if (stats.whitener[static_cast<std::size_t>(i) * bands + j] != 0.0f) upper_zero = false;
  CHECK(upper_zero);
}

void test_ace_matches_its_definition_and_stays_in_range() {
  const int bands = 12;
  std::mt19937 rng(202);
  std::vector<std::vector<float>> spectra;
  const SpectralMoments moments = make_scene(bands, 300, rng, &spectra);
  const SpectralLibrary library = make_library(bands);

  const double loading = 1e-5;
  DetectorStatistics stats;
  std::string error;
  CHECK(build_statistics(moments, library, loading, &stats, &error));
  if (!stats.valid) {
    std::fprintf(stderr, "  build_statistics: %s\n", error.c_str());
    return;
  }

  std::vector<double> covariance = moments.covariance(loading);
  std::vector<double> inverse;
  CHECK(invert_spd(covariance, bands, &inverse));
  const std::vector<double> mean = moments.mean();

  const auto mahalanobis_dot = [&](const std::vector<double>& u,
                                   const std::vector<double>& v) {
    double total = 0.0;
    for (int i = 0; i < bands; ++i) {
      double row = 0.0;
      for (int j = 0; j < bands; ++j) row += inverse[static_cast<std::size_t>(i) * bands + j] * v[static_cast<std::size_t>(j)];
      total += u[static_cast<std::size_t>(i)] * row;
    }
    return total;
  };

  double worst = 0.0;
  bool in_range = true;
  for (int t = 0; t < library.size(); ++t) {
    std::vector<double> d(static_cast<std::size_t>(bands));
    for (int i = 0; i < bands; ++i) {
      d[static_cast<std::size_t>(i)] =
          library.targets[static_cast<std::size_t>(t)].values[static_cast<std::size_t>(i)] -
          mean[static_cast<std::size_t>(i)];
    }
    const double dd = mahalanobis_dot(d, d);

    for (const std::vector<float>& x : spectra) {
      std::vector<double> y(static_cast<std::size_t>(bands));
      for (int i = 0; i < bands; ++i) {
        y[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - mean[static_cast<std::size_t>(i)];
      }
      const double yy = mahalanobis_dot(y, y);
      const double yd = mahalanobis_dot(y, d);
      const double want = (yd * yd) / (dd * yy);

      // The kernel form: whiten the pixel once, dot with the stored whitened
      // target, and divide by the two stored norms.
      std::vector<double> z(static_cast<std::size_t>(bands));
      double zz = 0.0;
      for (int i = 0; i < bands; ++i) {
        double s = 0.0;
        const float* row = stats.whitener.data() + static_cast<std::size_t>(i) * bands;
        for (int j = 0; j <= i; ++j) s += row[j] * y[static_cast<std::size_t>(j)];
        z[static_cast<std::size_t>(i)] = s;
        zz += s * s;
      }
      double zu = 0.0;
      for (int i = 0; i < bands; ++i) {
        zu += z[static_cast<std::size_t>(i)] *
              stats.target_white[static_cast<std::size_t>(t) * bands + i];
      }
      const double uu = stats.target_norm2[static_cast<std::size_t>(t)];
      const double got = (zu * zu) / (uu * zz);

      worst = std::max(worst, std::fabs(got - want));
      if (got < -1e-5 || got > 1.0 + 1e-5) in_range = false;
    }
  }
  CHECK(worst < 1e-4);
  // ACE is a squared cosine in the whitened space, so it cannot leave [0, 1].
  CHECK(in_range);
}

void test_cem_passes_its_target_at_unit_gain() {
  const int bands = 10;
  std::mt19937 rng(303);
  const SpectralMoments moments = make_scene(bands, 300, rng, nullptr);
  const SpectralLibrary library = make_library(bands);

  DetectorStatistics stats;
  std::string error;
  CHECK(build_statistics(moments, library, 1e-5, &stats, &error));
  if (!stats.valid) return;

  // The defining property: w^T d == 1 for its own target.
  double worst = 0.0;
  for (int t = 0; t < library.size(); ++t) {
    double response = 0.0;
    for (int i = 0; i < bands; ++i) {
      response += stats.cem_weight[static_cast<std::size_t>(t) * bands + i] *
                  library.targets[static_cast<std::size_t>(t)].values[static_cast<std::size_t>(i)];
    }
    worst = std::max(worst, std::fabs(response - 1.0));
  }
  CHECK(worst < 1e-3);
}

void test_singular_covariance_is_reported_not_silently_wrong() {
  // Two bands that are exact copies: the covariance is singular, and without
  // loading the factorisation must fail loudly rather than return garbage.
  const int bands = 4;
  SpectralMoments moments;
  moments.reset(bands);
  std::mt19937 rng(5);
  std::normal_distribution<float> dist(0.5f, 0.1f);
  for (int p = 0; p < 100; ++p) {
    const float shared = dist(rng);
    const float other = dist(rng);
    const float x[bands] = {shared, shared, other, other};
    moments.add(x);
  }

  DetectorStatistics stats;
  std::string error;
  CHECK(!build_statistics(moments, SpectralLibrary{}, 0.0, &stats, &error));
  CHECK(!stats.valid);
  // And the message has to point at the fix, not just state the failure.
  CHECK(error.find("loading") != std::string::npos);

  CHECK(build_statistics(moments, SpectralLibrary{}, 1e-3, &stats, &error));
  CHECK(stats.valid);
}

void test_library_band_mismatch_is_rejected() {
  const int bands = 6;
  std::mt19937 rng(13);
  const SpectralMoments moments = make_scene(bands, 100, rng, nullptr);
  const SpectralLibrary library = make_library(bands + 2);

  DetectorStatistics stats;
  std::string error;
  CHECK(!build_statistics(moments, library, 1e-5, &stats, &error));
  CHECK(error.find("bands") != std::string::npos);
}

}  // namespace

int main() {
  test_traits_are_consistent();
  test_whitener_reproduces_the_mahalanobis_distance();
  test_whitener_is_lower_triangular();
  test_ace_matches_its_definition_and_stays_in_range();
  test_cem_passes_its_target_at_unit_gain();
  test_singular_covariance_is_reported_not_silently_wrong();
  test_library_band_mismatch_is_rejected();
  return check::finish("detector");
}
