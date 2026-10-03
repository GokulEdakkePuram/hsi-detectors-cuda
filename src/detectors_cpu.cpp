#include "hsi/detectors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hsi {
namespace {

/// Gather one pixel's spectrum out of a BSQ cube into contiguous storage.
///
/// BSQ is the right layout for the GPU and the wrong one for a scalar loop, so
/// the reference pays a strided read per band. It is a reference; it is allowed
/// to be slow.
inline void gather(const float* cube_bsq, std::size_t pixels, int bands,
                   std::size_t p, double* out) {
  for (int b = 0; b < bands; ++b) {
    out[b] = cube_bsq[static_cast<std::size_t>(b) * pixels + p];
  }
}

/// Whiten a centred spectrum: z = L^-1 y, with L^-1 lower triangular.
/// Returns ||z||^2, which is the RX statistic.
inline double whiten(const float* whitener, int bands, const double* y, double* z) {
  double norm2 = 0.0;
  for (int i = 0; i < bands; ++i) {
    const float* row = whitener + static_cast<std::size_t>(i) * bands;
    double s = 0.0;
    for (int j = 0; j <= i; ++j) s += static_cast<double>(row[j]) * y[j];
    z[i] = s;
    norm2 += s * s;
  }
  return norm2;
}

/// Regularised lower incomplete gamma P(a, x), by series below the crossover
/// and continued fraction above it.
double gamma_p(double a, double x) {
  if (x <= 0.0) return 0.0;
  const double log_gamma_a = std::lgamma(a);
  if (x < a + 1.0) {
    double term = 1.0 / a;
    double sum = term;
    for (int n = 1; n < 1000; ++n) {
      term *= x / (a + n);
      sum += term;
      if (std::fabs(term) < std::fabs(sum) * 1e-16) break;
    }
    return sum * std::exp(-x + a * std::log(x) - log_gamma_a);
  }
  // Lentz's algorithm on the continued fraction for Q(a, x).
  const double tiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / tiny;
  double d = 1.0 / b;
  double h = d;
  for (int n = 1; n < 1000; ++n) {
    const double an = -n * (n - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::fabs(delta - 1.0) < 1e-16) break;
  }
  const double q = std::exp(-x + a * std::log(x) - log_gamma_a) * h;
  return 1.0 - q;
}

}  // namespace

void accumulate_moments_cpu(const float* cube_bsq, CubeShape shape,
                            SpectralMoments* moments) {
  if (!cube_bsq || !moments) throw std::invalid_argument("accumulate_moments_cpu: null");
  if (!shape.valid()) throw std::invalid_argument("accumulate_moments_cpu: bad shape");
  if (moments->bands != shape.bands) moments->reset(shape.bands);

  const std::size_t pixels = shape.pixels();
  std::vector<double> spectrum(static_cast<std::size_t>(shape.bands));
  std::vector<float> narrow(static_cast<std::size_t>(shape.bands));
  for (std::size_t p = 0; p < pixels; ++p) {
    gather(cube_bsq, pixels, shape.bands, p, spectrum.data());
    for (int b = 0; b < shape.bands; ++b) {
      narrow[static_cast<std::size_t>(b)] = static_cast<float>(spectrum[static_cast<std::size_t>(b)]);
    }
    moments->add(narrow.data());
  }
}

void rx_cpu(const float* cube_bsq, CubeShape shape,
            const DetectorStatistics& stats, float* out_score) {
  if (!stats.valid) throw std::invalid_argument("rx_cpu: statistics are not valid");
  if (stats.bands != shape.bands) throw std::invalid_argument("rx_cpu: band mismatch");

  const std::size_t pixels = shape.pixels();
  const int bands = shape.bands;
  std::vector<double> x(static_cast<std::size_t>(bands));
  std::vector<double> y(static_cast<std::size_t>(bands));
  std::vector<double> z(static_cast<std::size_t>(bands));

  for (std::size_t p = 0; p < pixels; ++p) {
    gather(cube_bsq, pixels, bands, p, x.data());
    for (int b = 0; b < bands; ++b) {
      y[static_cast<std::size_t>(b)] = x[static_cast<std::size_t>(b)] - stats.mean[static_cast<std::size_t>(b)];
    }
    out_score[p] = static_cast<float>(whiten(stats.whitener.data(), bands, y.data(), z.data()));
  }
}

void ace_best_cpu(const float* cube_bsq, CubeShape shape,
                  const DetectorStatistics& stats, float* out_score,
                  std::int32_t* out_target, float* out_rx) {
  if (!stats.valid) throw std::invalid_argument("ace_best_cpu: statistics are not valid");
  if (stats.bands != shape.bands) throw std::invalid_argument("ace_best_cpu: band mismatch");
  if (stats.num_targets <= 0) throw std::invalid_argument("ace_best_cpu: no targets");

  const std::size_t pixels = shape.pixels();
  const int bands = shape.bands;
  std::vector<double> x(static_cast<std::size_t>(bands));
  std::vector<double> y(static_cast<std::size_t>(bands));
  std::vector<double> z(static_cast<std::size_t>(bands));

  for (std::size_t p = 0; p < pixels; ++p) {
    gather(cube_bsq, pixels, bands, p, x.data());
    for (int b = 0; b < bands; ++b) {
      y[static_cast<std::size_t>(b)] = x[static_cast<std::size_t>(b)] - stats.mean[static_cast<std::size_t>(b)];
    }
    // One whitening, then every target rides on it.
    const double zz = whiten(stats.whitener.data(), bands, y.data(), z.data());
    if (out_rx) out_rx[p] = static_cast<float>(zz);

    double best = -1.0;
    std::int32_t best_target = -1;
    for (int t = 0; t < stats.num_targets; ++t) {
      const float* u = stats.target_white.data() + static_cast<std::size_t>(t) * bands;
      const double uu = stats.target_norm2[static_cast<std::size_t>(t)];
      if (!(uu > 0.0) || !(zz > 0.0)) continue;
      double zu = 0.0;
      for (int b = 0; b < bands; ++b) zu += z[static_cast<std::size_t>(b)] * u[b];
      const double score = (zu * zu) / (uu * zz);
      if (score > best) {
        best = score;
        best_target = t;
      }
    }
    // A pixel sitting exactly at the scene mean has no direction to compare,
    // so it scores 0 - the least target-like value ACE can take.
    out_score[p] = static_cast<float>(best < 0.0 ? 0.0 : std::min(best, 1.0));
    if (out_target) out_target[p] = best_target;
  }
}

void cem_best_cpu(const float* cube_bsq, CubeShape shape,
                  const DetectorStatistics& stats, float* out_score,
                  std::int32_t* out_target) {
  if (!stats.valid) throw std::invalid_argument("cem_best_cpu: statistics are not valid");
  if (stats.bands != shape.bands) throw std::invalid_argument("cem_best_cpu: band mismatch");
  if (stats.num_targets <= 0) throw std::invalid_argument("cem_best_cpu: no targets");

  const std::size_t pixels = shape.pixels();
  const int bands = shape.bands;
  std::vector<double> x(static_cast<std::size_t>(bands));

  for (std::size_t p = 0; p < pixels; ++p) {
    gather(cube_bsq, pixels, bands, p, x.data());
    double best = -std::numeric_limits<double>::infinity();
    std::int32_t best_target = -1;
    for (int t = 0; t < stats.num_targets; ++t) {
      const float* w = stats.cem_weight.data() + static_cast<std::size_t>(t) * bands;
      double response = 0.0;
      // CEM applies its filter to the uncentred pixel: the unit-gain
      // constraint is defined against the raw signature, not a centred one.
      for (int b = 0; b < bands; ++b) response += static_cast<double>(w[b]) * x[static_cast<std::size_t>(b)];
      if (response > best) {
        best = response;
        best_target = t;
      }
    }
    out_score[p] = static_cast<float>(best);
    if (out_target) out_target[p] = best_target;
  }
}

double rx_threshold_for_false_alarm_rate(int bands, double rate) {
  if (bands <= 0) throw std::invalid_argument("rx_threshold: bands must be positive");
  if (!(rate > 0.0) || !(rate < 1.0)) {
    throw std::invalid_argument("rx_threshold: rate must be in (0, 1)");
  }
  // Solve P(X > t) = rate for X ~ chi-squared with `bands` degrees of freedom,
  // i.e. gamma_p(bands/2, t/2) = 1 - rate. Monotone in t, so bisect.
  const double a = 0.5 * bands;
  const double want = 1.0 - rate;
  double low = 0.0;
  double high = std::max(4.0 * bands, 16.0);
  while (gamma_p(a, 0.5 * high) < want && high < 1e9) high *= 2.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (low + high);
    if (gamma_p(a, 0.5 * mid) < want) low = mid; else high = mid;
  }
  return 0.5 * (low + high);
}

}  // namespace hsi
