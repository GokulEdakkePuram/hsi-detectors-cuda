// The host linear algebra that RX, ACE and CEM all rest on. Every one of them
// inverts a scene covariance, so an error here is an error in three detectors.

#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "hsi/linalg.hpp"

using namespace hsi;

namespace {

std::vector<double> multiply(const std::vector<double>& a,
                             const std::vector<double>& b, int n) {
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        out[static_cast<std::size_t>(i) * n + j] +=
            a[static_cast<std::size_t>(i) * n + k] * b[static_cast<std::size_t>(k) * n + j];
  return out;
}

/// A random symmetric positive-definite matrix: B B^T plus a diagonal, which
/// guarantees positive definiteness without being trivially diagonal.
std::vector<double> random_spd(int n, std::mt19937& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> b(static_cast<std::size_t>(n) * n);
  for (double& v : b) v = dist(rng);
  std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += b[static_cast<std::size_t>(i) * n + k] * b[static_cast<std::size_t>(j) * n + k];
      a[static_cast<std::size_t>(i) * n + j] = s;
    }
  for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i) * n + i] += n;
  return a;
}

void test_cholesky_reconstructs_the_matrix() {
  std::mt19937 rng(11);
  for (int n : {1, 2, 5, 17}) {
    const std::vector<double> a = random_spd(n, rng);
    std::vector<double> l = a;
    CHECK(cholesky(l.data(), n));

    // L must be lower triangular, and L L^T must be the original.
    bool upper_zero = true;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (l[static_cast<std::size_t>(i) * n + j] != 0.0) upper_zero = false;
    CHECK(upper_zero);

    std::vector<double> lt(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        lt[static_cast<std::size_t>(i) * n + j] = l[static_cast<std::size_t>(j) * n + i];
    const std::vector<double> product = multiply(l, lt, n);

    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      worst = std::max(worst, std::fabs(product[i] - a[i]));
    }
    CHECK(worst < 1e-9);
  }
}

void test_cholesky_rejects_non_positive_definite() {
  // Zero matrix, a negative diagonal, and a singular rank-1 matrix.
  std::vector<double> zero(9, 0.0);
  CHECK(!cholesky(zero.data(), 3));

  std::vector<double> negative{1.0, 0.0, 0.0, -4.0};
  CHECK(!cholesky(negative.data(), 2));

  std::vector<double> singular{1.0, 1.0, 1.0, 1.0};
  CHECK(!cholesky(singular.data(), 2));
}

void test_inverse_round_trips() {
  std::mt19937 rng(23);
  for (int n : {1, 3, 9, 21}) {
    std::vector<double> a = random_spd(n, rng);
    const std::vector<double> original = a;
    std::vector<double> inv;
    CHECK(invert_spd(a, n, &inv));

    const std::vector<double> identity = multiply(original, inv, n);
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        const double want = (i == j) ? 1.0 : 0.0;
        worst = std::max(worst, std::fabs(identity[static_cast<std::size_t>(i) * n + j] - want));
      }
    CHECK(worst < 1e-9);
  }
}

void test_solve_matches_the_inverse() {
  std::mt19937 rng(37);
  const int n = 12;
  std::vector<double> a = random_spd(n, rng);
  const std::vector<double> original = a;

  std::vector<double> l = a;
  CHECK(cholesky(l.data(), n));

  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> b(static_cast<std::size_t>(n));
  for (double& v : b) v = dist(rng);

  std::vector<double> x(static_cast<std::size_t>(n));
  cholesky_solve(l.data(), n, b.data(), x.data());

  // A x must reproduce b.
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) s += original[static_cast<std::size_t>(i) * n + j] * x[static_cast<std::size_t>(j)];
    worst = std::max(worst, std::fabs(s - b[static_cast<std::size_t>(i)]));
  }
  CHECK(worst < 1e-9);
}

void test_moments_recover_mean_and_covariance() {
  const int bands = 4;
  const int pixels = 500;
  std::mt19937 rng(5);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  // Known mean and a known correlation between bands 0 and 1.
  const float offsets[bands] = {0.5f, 2.0f, -1.0f, 0.25f};
  std::vector<std::vector<float>> spectra;
  SpectralMoments moments;
  moments.reset(bands);
  for (int p = 0; p < pixels; ++p) {
    std::vector<float> x(bands);
    const float shared = dist(rng);
    x[0] = offsets[0] + shared;
    x[1] = offsets[1] + shared;  // perfectly correlated with band 0
    x[2] = offsets[2] + dist(rng);
    x[3] = offsets[3] + dist(rng);
    moments.add(x.data());
    spectra.push_back(x);
  }

  CHECK(moments.count == pixels);

  // Against the direct two-pass computation, which is the definition.
  std::vector<double> mu(bands, 0.0);
  for (const std::vector<float>& x : spectra)
    for (int i = 0; i < bands; ++i) mu[static_cast<std::size_t>(i)] += x[static_cast<std::size_t>(i)];
  for (double& v : mu) v /= pixels;

  const std::vector<double> got_mean = moments.mean();
  for (int i = 0; i < bands; ++i) CHECK_CLOSE(got_mean[static_cast<std::size_t>(i)], mu[static_cast<std::size_t>(i)], 1e-9);

  std::vector<double> want(static_cast<std::size_t>(bands) * bands, 0.0);
  for (const std::vector<float>& x : spectra)
    for (int i = 0; i < bands; ++i)
      for (int j = 0; j < bands; ++j)
        want[static_cast<std::size_t>(i) * bands + j] +=
            (x[static_cast<std::size_t>(i)] - mu[static_cast<std::size_t>(i)]) *
            (x[static_cast<std::size_t>(j)] - mu[static_cast<std::size_t>(j)]);
  for (double& v : want) v /= pixels;

  const std::vector<double> got = moments.covariance();
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
  CHECK(worst < 1e-6);

  // Bands 0 and 1 share a term, so their covariance equals each variance and
  // the matrix is singular. That is the case diagonal loading exists for.
  CHECK_CLOSE(got[0 * bands + 1], got[0 * bands + 0], 1e-6);
  std::vector<double> bare = moments.covariance();
  CHECK(!cholesky(bare.data(), bands));
  std::vector<double> loaded = moments.covariance(1e-3);
  CHECK(cholesky(loaded.data(), bands));
}

void test_merge_matches_one_pass() {
  const int bands = 3;
  std::mt19937 rng(9);
  std::normal_distribution<float> dist(1.0f, 0.5f);

  SpectralMoments whole, left, right;
  whole.reset(bands);
  left.reset(bands);
  right.reset(bands);

  for (int p = 0; p < 200; ++p) {
    std::vector<float> x(bands);
    for (float& v : x) v = dist(rng);
    whole.add(x.data());
    (p < 70 ? left : right).add(x.data());
  }

  left.merge(right);
  CHECK(left.count == whole.count);
  const std::vector<double> a = left.covariance();
  const std::vector<double> b = whole.covariance();
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
  CHECK(worst < 1e-12);
}

void test_quadratic_form() {
  // Identity gives the squared euclidean norm.
  const int n = 3;
  std::vector<double> m{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> y{1.0, 2.0, 3.0};
  CHECK_CLOSE(quadratic_form(m.data(), y.data(), n), 14.0, 1e-12);

  // And for the inverse of a covariance it is the squared Mahalanobis distance:
  // y^T S^-1 y with S = diag(4, 9, 1) should be 1/4 + 4/9 + 9.
  std::vector<double> s{4, 0, 0, 0, 9, 0, 0, 0, 1};
  std::vector<double> inv;
  CHECK(invert_spd(s, n, &inv));
  CHECK_CLOSE(quadratic_form(inv.data(), y.data(), n), 0.25 + 4.0 / 9.0 + 9.0, 1e-12);
}

}  // namespace

int main() {
  test_cholesky_reconstructs_the_matrix();
  test_cholesky_rejects_non_positive_definite();
  test_inverse_round_trips();
  test_solve_matches_the_inverse();
  test_moments_recover_mean_and_covariance();
  test_merge_matches_one_pass();
  test_quadratic_form();
  return check::finish("linalg");
}
