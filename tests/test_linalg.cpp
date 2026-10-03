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

void test_shifted_moments_reconstruct_exactly() {
  // The device accumulates about a reference point to keep fp32 alive; the
  // host has to rebuild the raw moments from that without drift.
  const int bands = 5;
  const int pixels = 300;
  std::mt19937 rng(77);
  std::normal_distribution<float> dist(0.4f, 0.05f);

  std::vector<std::vector<float>> spectra;
  SpectralMoments direct;
  direct.reset(bands);
  for (int p = 0; p < pixels; ++p) {
    std::vector<float> x(static_cast<std::size_t>(bands));
    for (float& v : x) v = dist(rng);
    direct.add(x.data());
    spectra.push_back(x);
  }

  // A reference that is close but deliberately not the exact mean, which is
  // the realistic case: it comes from the previous frame.
  std::vector<double> reference(static_cast<std::size_t>(bands), 0.41);

  std::vector<double> s1(static_cast<std::size_t>(bands), 0.0);
  std::vector<double> s2(static_cast<std::size_t>(bands) * bands, 0.0);
  for (const std::vector<float>& x : spectra) {
    for (int i = 0; i < bands; ++i) {
      const double di = x[static_cast<std::size_t>(i)] - reference[static_cast<std::size_t>(i)];
      s1[static_cast<std::size_t>(i)] += di;
      for (int j = 0; j <= i; ++j) {
        s2[static_cast<std::size_t>(i) * bands + j] +=
            di * (x[static_cast<std::size_t>(j)] - reference[static_cast<std::size_t>(j)]);
      }
    }
  }

  SpectralMoments rebuilt;
  moments_from_shifted(s1.data(), s2.data(), pixels, reference.data(), bands, &rebuilt);

  CHECK(rebuilt.count == direct.count);

  const std::vector<double> want_mean = direct.mean();
  const std::vector<double> got_mean = rebuilt.mean();
  for (int i = 0; i < bands; ++i) {
    CHECK_CLOSE(got_mean[static_cast<std::size_t>(i)], want_mean[static_cast<std::size_t>(i)], 1e-12);
  }

  const std::vector<double> want_cov = direct.covariance();
  const std::vector<double> got_cov = rebuilt.covariance();
  double worst = 0.0;
  for (std::size_t i = 0; i < want_cov.size(); ++i) {
    worst = std::max(worst, std::fabs(got_cov[i] - want_cov[i]));
  }
  CHECK(worst < 1e-14);

  // The correlation matrix CEM needs has to survive the round trip too.
  const std::vector<double> want_corr = direct.correlation();
  const std::vector<double> got_corr = rebuilt.correlation();
  worst = 0.0;
  for (std::size_t i = 0; i < want_corr.size(); ++i) {
    worst = std::max(worst, std::fabs(got_corr[i] - want_corr[i]));
  }
  CHECK(worst < 1e-14);
}

void test_block_accumulation_is_what_makes_fp32_viable() {
  // This test fixes the device accumulator's design, so it measures all three
  // strategies rather than asserting one.
  //
  // The quantity the detectors consume is the variance, and it is where the
  // errors differ most. Measured over 400k samples of reflectance near 0.37:
  //
  //   unshifted, one fp32 accumulator   1.7e-05 on sum(x^2)
  //                                     ~5e-03 on the variance, because
  //                                     sum(x^2)/N - mean^2 subtracts 0.1369
  //                                     from 0.1373 and amplifies the relative
  //                                     error by about 300x
  //   shifted, one fp32 accumulator     1.3e-04 on both - no amplification,
  //                                     but a single accumulator absorbs the
  //                                     small increments
  //   shifted, fp32 blocks -> fp64      2.4e-08 on both
  //
  // So the hierarchy is the fix and the shift is what protects the subtraction
  // that follows it. The kernel needs both, and it needs the hierarchy more.
  const int pixels = 400000;
  const int block = 1024;
  const double reference = 0.37;
  std::mt19937 rng(91);
  std::normal_distribution<double> dist(0.37, 0.02);

  long double exact_s1 = 0.0L, exact_s2 = 0.0L;
  float flat_s1 = 0.0f, flat_s2 = 0.0f;
  double blocked_s1 = 0.0, blocked_s2 = 0.0;
  float partial_s1 = 0.0f, partial_s2 = 0.0f;

  for (int p = 0; p < pixels; ++p) {
    const double d = dist(rng) - reference;
    exact_s1 += static_cast<long double>(d);
    exact_s2 += static_cast<long double>(d) * static_cast<long double>(d);
    flat_s1 += static_cast<float>(d);
    flat_s2 += static_cast<float>(d * d);
    partial_s1 += static_cast<float>(d);
    partial_s2 += static_cast<float>(d * d);
    if ((p + 1) % block == 0) {
      blocked_s1 += partial_s1;
      blocked_s2 += partial_s2;
      partial_s1 = 0.0f;
      partial_s2 = 0.0f;
    }
  }
  blocked_s1 += partial_s1;
  blocked_s2 += partial_s2;

  const auto variance = [&](double s1, double s2) {
    return s2 / pixels - (s1 / pixels) * (s1 / pixels);
  };
  const double want = variance(static_cast<double>(exact_s1), static_cast<double>(exact_s2));
  const double flat_error = std::fabs(variance(flat_s1, flat_s2) - want) / want;
  const double blocked_error = std::fabs(variance(blocked_s1, blocked_s2) - want) / want;

  // What the kernel will do has to be accurate in absolute terms.
  CHECK(blocked_error < 1e-6);
  // And it has to be decisively better than the single-accumulator version,
  // which is the reason for the extra machinery.
  CHECK(blocked_error < flat_error / 100.0);
}

}  // namespace

int main() {
  test_shifted_moments_reconstruct_exactly();
  test_block_accumulation_is_what_makes_fp32_viable();
  test_cholesky_reconstructs_the_matrix();
  test_cholesky_rejects_non_positive_definite();
  test_inverse_round_trips();
  test_solve_matches_the_inverse();
  test_moments_recover_mean_and_covariance();
  test_merge_matches_one_pass();
  test_quadratic_form();
  return check::finish("linalg");
}
