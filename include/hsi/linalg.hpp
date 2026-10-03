#pragma once

#include <cstddef>
#include <vector>

namespace hsi {

/// First and second moments of a set of spectra.
///
/// RX, ACE and CEM all need second-order scene statistics, and one accumulation
/// serves all three: with M1 = sum(x) and M2 = sum(x x^T) over N pixels,
///
///     mean        mu = M1 / N
///     correlation R  = M2 / N              (what CEM uses)
///     covariance  S  = M2 / N - mu mu^T    (what RX and ACE use)
///
/// Accumulated in double throughout. That is not fastidiousness: for
/// reflectance in [0,1] the entries of M2/N are around 0.14 while the entries
/// of the covariance can be around 0.01, so forming S by subtraction loses most
/// of a float's mantissa to cancellation. The device-side accumulator has the
/// same problem and solves it differently - see stats_cuda.hpp.
struct SpectralMoments {
  int bands = 0;
  double count = 0.0;
  std::vector<double> sum;        ///< M1, `bands` entries
  std::vector<double> sum_outer;  ///< M2, row-major bands*bands, symmetric

  void reset(int bands);

  /// Accumulate one spectrum of `bands` contiguous values.
  void add(const float* spectrum);

  /// Merge another accumulator covering different pixels.
  void merge(const SpectralMoments& other);

  bool empty() const { return count <= 0.0; }

  std::vector<double> mean() const;

  /// M2/N - mu mu^T, plus `diagonal_loading` on the diagonal.
  ///
  /// Loading is not optional in practice. A scene with fewer independent
  /// spectra than bands - any scene with strong band-to-band correlation, which
  /// is all of them - gives a covariance that is singular or nearly so, and
  /// inverting it amplifies noise into false detections. Adding a small
  /// multiple of the identity is the standard remedy.
  std::vector<double> covariance(double diagonal_loading = 0.0) const;

  /// M2/N, plus `diagonal_loading` on the diagonal.
  std::vector<double> correlation(double diagonal_loading = 0.0) const;
};

/// In-place Cholesky factorisation of a symmetric positive-definite matrix.
///
/// `a` is row-major n*n on entry and holds the lower triangle of L on exit;
/// the strict upper triangle is zeroed. Returns false if `a` is not positive
/// definite, which for a scene covariance means the loading was too small.
bool cholesky(double* a, int n);

/// Solve A x = b for one right-hand side, given L from cholesky().
void cholesky_solve(const double* l, int n, const double* b, double* x);

/// Invert A given L from cholesky(). `inv` is row-major n*n.
void cholesky_inverse(const double* l, int n, double* inv);

/// Invert a lower-triangular matrix, given L from cholesky().
///
/// This is what the detectors actually want. With S = L L^T,
///
///     y^T S^-1 y = || L^-1 y ||^2
///
/// so whitening a pixel with L^-1 and taking its squared norm gives the RX
/// statistic for half the arithmetic of the full symmetric quadratic form, and
/// for half the storage since L^-1 is triangular. ACE falls out of the same
/// transform: its numerator is the dot product of two whitened vectors, so one
/// whitening per pixel serves RX and every ACE target together.
///
/// `inv` is row-major n*n with the strict upper triangle zeroed.
void triangular_inverse_lower(const double* l, int n, double* inv);

/// Invert a symmetric positive-definite matrix. `a` is consumed.
/// Returns false if the factorisation fails.
bool invert_spd(std::vector<double>& a, int n, std::vector<double>* inv);

/// y^T M y for symmetric M, the quadratic form RX and ACE are built from.
double quadratic_form(const double* m, const double* y, int n);

}  // namespace hsi
