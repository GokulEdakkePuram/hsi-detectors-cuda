#include "hsi/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hsi {

void SpectralMoments::reset(int band_count) {
  if (band_count <= 0) throw std::invalid_argument("SpectralMoments: bands must be positive");
  bands = band_count;
  count = 0.0;
  sum.assign(static_cast<std::size_t>(bands), 0.0);
  sum_outer.assign(static_cast<std::size_t>(bands) * bands, 0.0);
}

void SpectralMoments::add(const float* spectrum) {
  if (bands <= 0) throw std::logic_error("SpectralMoments::add before reset");
  for (int i = 0; i < bands; ++i) {
    const double vi = spectrum[i];
    sum[static_cast<std::size_t>(i)] += vi;
    // Only the lower triangle is accumulated; the matrix is symmetric and the
    // upper triangle is filled when it is read out.
    double* row = sum_outer.data() + static_cast<std::size_t>(i) * bands;
    for (int j = 0; j <= i; ++j) row[j] += vi * spectrum[j];
  }
  count += 1.0;
}

void SpectralMoments::merge(const SpectralMoments& other) {
  if (other.empty()) return;
  if (bands == 0) {
    *this = other;
    return;
  }
  if (bands != other.bands) {
    throw std::invalid_argument("SpectralMoments::merge: band count mismatch");
  }
  for (std::size_t i = 0; i < sum.size(); ++i) sum[i] += other.sum[i];
  for (std::size_t i = 0; i < sum_outer.size(); ++i) sum_outer[i] += other.sum_outer[i];
  count += other.count;
}

std::vector<double> SpectralMoments::mean() const {
  if (empty()) throw std::logic_error("SpectralMoments::mean: nothing accumulated");
  std::vector<double> out(static_cast<std::size_t>(bands));
  for (int i = 0; i < bands; ++i) out[static_cast<std::size_t>(i)] = sum[static_cast<std::size_t>(i)] / count;
  return out;
}

std::vector<double> SpectralMoments::correlation(double diagonal_loading) const {
  if (empty()) throw std::logic_error("SpectralMoments::correlation: nothing accumulated");
  std::vector<double> out(static_cast<std::size_t>(bands) * bands);
  for (int i = 0; i < bands; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double v = sum_outer[static_cast<std::size_t>(i) * bands + j] / count;
      out[static_cast<std::size_t>(i) * bands + j] = v;
      out[static_cast<std::size_t>(j) * bands + i] = v;
    }
    out[static_cast<std::size_t>(i) * bands + i] += diagonal_loading;
  }
  return out;
}

std::vector<double> SpectralMoments::covariance(double diagonal_loading) const {
  std::vector<double> out = correlation(0.0);
  const std::vector<double> mu = mean();
  for (int i = 0; i < bands; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double v = out[static_cast<std::size_t>(i) * bands + j] -
                       mu[static_cast<std::size_t>(i)] * mu[static_cast<std::size_t>(j)];
      out[static_cast<std::size_t>(i) * bands + j] = v;
      out[static_cast<std::size_t>(j) * bands + i] = v;
    }
    out[static_cast<std::size_t>(i) * bands + i] += diagonal_loading;
  }
  return out;
}

bool cholesky(double* a, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = a[static_cast<std::size_t>(i) * n + j];
      for (int k = 0; k < j; ++k) {
        s -= a[static_cast<std::size_t>(i) * n + k] * a[static_cast<std::size_t>(j) * n + k];
      }
      if (i == j) {
        if (!(s > 0.0)) return false;  // also catches NaN
        a[static_cast<std::size_t>(i) * n + i] = std::sqrt(s);
      } else {
        a[static_cast<std::size_t>(i) * n + j] = s / a[static_cast<std::size_t>(j) * n + j];
      }
    }
    for (int j = i + 1; j < n; ++j) a[static_cast<std::size_t>(i) * n + j] = 0.0;
  }
  return true;
}

void cholesky_solve(const double* l, int n, const double* b, double* x) {
  // Forward substitution: L z = b.
  for (int i = 0; i < n; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k) s -= l[static_cast<std::size_t>(i) * n + k] * x[k];
    x[i] = s / l[static_cast<std::size_t>(i) * n + i];
  }
  // Back substitution: L^T x = z.
  for (int i = n - 1; i >= 0; --i) {
    double s = x[i];
    for (int k = i + 1; k < n; ++k) s -= l[static_cast<std::size_t>(k) * n + i] * x[k];
    x[i] = s / l[static_cast<std::size_t>(i) * n + i];
  }
}

void cholesky_inverse(const double* l, int n, double* inv) {
  std::vector<double> e(static_cast<std::size_t>(n), 0.0);
  std::vector<double> col(static_cast<std::size_t>(n), 0.0);
  for (int k = 0; k < n; ++k) {
    std::fill(e.begin(), e.end(), 0.0);
    e[static_cast<std::size_t>(k)] = 1.0;
    cholesky_solve(l, n, e.data(), col.data());
    for (int i = 0; i < n; ++i) inv[static_cast<std::size_t>(i) * n + k] = col[static_cast<std::size_t>(i)];
  }
}

void triangular_inverse_lower(const double* l, int n, double* inv) {
  std::fill(inv, inv + static_cast<std::size_t>(n) * n, 0.0);
  for (int k = 0; k < n; ++k) {
    inv[static_cast<std::size_t>(k) * n + k] = 1.0 / l[static_cast<std::size_t>(k) * n + k];
    for (int i = k + 1; i < n; ++i) {
      double s = 0.0;
      for (int j = k; j < i; ++j) {
        s -= l[static_cast<std::size_t>(i) * n + j] * inv[static_cast<std::size_t>(j) * n + k];
      }
      inv[static_cast<std::size_t>(i) * n + k] = s / l[static_cast<std::size_t>(i) * n + i];
    }
  }
}

bool invert_spd(std::vector<double>& a, int n, std::vector<double>* inv) {
  if (static_cast<int>(a.size()) != n * n || !inv) return false;
  if (!cholesky(a.data(), n)) return false;
  inv->assign(static_cast<std::size_t>(n) * n, 0.0);
  cholesky_inverse(a.data(), n, inv->data());
  return true;
}

double quadratic_form(const double* m, const double* y, int n) {
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    double row = 0.0;
    for (int j = 0; j < n; ++j) row += m[static_cast<std::size_t>(i) * n + j] * y[j];
    total += y[i] * row;
  }
  return total;
}

}  // namespace hsi
