// Properties of the spectral angle, checked against the reference
// implementation. These are the invariants the CUDA kernels also have to hold.

#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "hsi/sam.hpp"

using namespace hsi;

namespace {

CubeShape make_shape(int h, int w, int b) {
  CubeShape s;
  s.height = h;
  s.width = w;
  s.bands = b;
  return s;
}

void test_identical_spectrum_is_zero_angle() {
  const CubeShape shape = make_shape(1, 1, 8);
  std::vector<float> target{0.1f, 0.4f, 0.9f, 0.3f, 0.2f, 0.7f, 0.5f, 0.6f};
  std::vector<float> cube = target;  // 1x1 cube, BSQ == the spectrum itself

  SpectralLibrary library;
  library.targets.push_back({"t", target, {}});
  const std::vector<float> flat = library.flatten();
  const std::vector<float> norms = library.norms();

  float angle = -1.0f;
  std::int32_t which = -1;
  sam_best_cpu(cube.data(), shape, flat.data(), norms.data(), 1, &angle, &which);

  CHECK_CLOSE(angle, 0.0, 1e-5);
  CHECK(which == 0);
}

void test_scale_invariance() {
  // The reason SAM is used at all: a pixel brightened or dimmed by
  // illumination keeps the same angle. If this ever fails the detector stops
  // working in shadow, which is the case it is supposed to survive.
  const int bands = 16;
  const CubeShape shape = make_shape(1, 1, bands);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(0.05f, 1.0f);
  std::vector<float> target(bands), pixel(bands);
  for (int b = 0; b < bands; ++b) {
    target[b] = dist(rng);
    pixel[b] = dist(rng);
  }

  SpectralLibrary library;
  library.targets.push_back({"t", target, {}});
  const std::vector<float> flat = library.flatten();
  const std::vector<float> norms = library.norms();

  float base_angle = 0.0f;
  std::int32_t which = 0;
  sam_best_cpu(pixel.data(), shape, flat.data(), norms.data(), 1, &base_angle, &which);

  for (float gain : {0.01f, 0.5f, 3.0f, 250.0f}) {
    std::vector<float> scaled(bands);
    for (int b = 0; b < bands; ++b) scaled[b] = pixel[b] * gain;
    float angle = 0.0f;
    sam_best_cpu(scaled.data(), shape, flat.data(), norms.data(), 1, &angle, &which);
    CHECK_CLOSE(angle, base_angle, 1e-4);
  }
}

void test_orthogonal_and_zero_pixels() {
  const int bands = 4;
  const CubeShape shape = make_shape(1, 2, bands);

  // Pixel 0 is orthogonal to the target, pixel 1 is all zeros.
  std::vector<float> cube(shape.elements(), 0.0f);
  const float values[4] = {0.0f, 1.0f, 0.0f, 1.0f};
  for (int b = 0; b < bands; ++b) cube[static_cast<std::size_t>(b) * 2 + 0] = values[b];

  SpectralLibrary library;
  library.targets.push_back({"t", {1.0f, 0.0f, 1.0f, 0.0f}, {}});
  const std::vector<float> flat = library.flatten();
  const std::vector<float> norms = library.norms();

  std::vector<float> angle(2);
  std::vector<std::int32_t> which(2);
  sam_best_cpu(cube.data(), shape, flat.data(), norms.data(), 1, angle.data(),
               which.data());

  CHECK_CLOSE(angle[0], kMaxSamAngle, 1e-5);
  // A directionless pixel must land on the maximum angle, not on NaN, or it
  // would sail through any threshold comparison.
  CHECK_CLOSE(angle[1], kMaxSamAngle, 1e-5);
  CHECK(!std::isnan(angle[1]));
}

void test_best_match_picks_the_nearest_target() {
  const int bands = 6;
  const CubeShape shape = make_shape(1, 1, bands);
  std::vector<float> pixel{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

  SpectralLibrary library;
  library.targets.push_back({"far", {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {}});
  library.targets.push_back({"near", {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f}, {}});
  library.targets.push_back({"middling", {0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f}, {}});
  const std::vector<float> flat = library.flatten();
  const std::vector<float> norms = library.norms();

  float angle = 0.0f;
  std::int32_t which = -1;
  sam_best_cpu(pixel.data(), shape, flat.data(), norms.data(), 3, &angle, &which);

  // "near" is the pixel scaled by two, so it should win outright at zero angle.
  CHECK(which == 1);
  CHECK_CLOSE(angle, 0.0, 1e-5);

  // The best-match reduction must agree with the full per-target cube.
  std::vector<float> all(3);
  sam_cpu(pixel.data(), shape, flat.data(), norms.data(), 3, all.data());
  CHECK_CLOSE(all[1], angle, 1e-6);
  CHECK(all[0] > angle && all[2] > angle);
}

void test_mean_spectrum_over_mask() {
  const CubeShape shape = make_shape(2, 2, 3);
  std::vector<float> cube(shape.elements());
  for (int b = 0; b < 3; ++b) {
    for (std::size_t p = 0; p < 4; ++p) {
      cube[static_cast<std::size_t>(b) * 4 + p] = static_cast<float>(b * 10 + p);
    }
  }
  const std::uint8_t mask[4] = {1, 0, 0, 1};  // pixels 0 and 3
  const Spectrum s = mean_spectrum(cube.data(), shape, mask, "blood");

  CHECK(s.name == "blood");
  CHECK(s.bands() == 3);
  for (int b = 0; b < 3; ++b) CHECK_CLOSE(s.values[b], b * 10 + 1.5, 1e-5);
}

void test_nms_keeps_one_pixel_per_blob() {
  const CubeShape shape = make_shape(9, 9, 1);
  std::vector<float> angle(81, 1.0f);
  std::vector<std::int32_t> which(81, 0);

  // A 3x3 blob well under threshold with a unique minimum at its centre.
  for (int y = 3; y <= 5; ++y) {
    for (int x = 3; x <= 5; ++x) angle[y * 9 + x] = 0.05f;
  }
  angle[4 * 9 + 4] = 0.01f;

  DetectionParams params;
  params.threshold = 0.1f;

  params.nms_radius = 0;
  CHECK(detect_cpu(angle.data(), which.data(), shape, params).size() == 9);

  params.nms_radius = 2;
  const std::vector<Detection> peaks =
      detect_cpu(angle.data(), which.data(), shape, params);
  CHECK(peaks.size() == 1);
  if (peaks.size() == 1) {
    CHECK(peaks[0].x == 4 && peaks[0].y == 4);
    CHECK_CLOSE(peaks[0].score, 0.01, 1e-6);
  }
}

void test_nms_breaks_ties_by_index() {
  // A plateau of equal angles has no unique minimum; the tie-break has to pick
  // exactly one, and the CUDA kernel has to pick the same one.
  const CubeShape shape = make_shape(5, 5, 1);
  std::vector<float> angle(25, 1.0f);
  std::vector<std::int32_t> which(25, 0);
  for (int y = 1; y <= 2; ++y) {
    for (int x = 1; x <= 2; ++x) angle[y * 5 + x] = 0.02f;
  }

  DetectionParams params;
  params.threshold = 0.1f;
  params.nms_radius = 2;
  const std::vector<Detection> peaks =
      detect_cpu(angle.data(), which.data(), shape, params);

  CHECK(peaks.size() == 1);
  if (peaks.size() == 1) CHECK(peaks[0].x == 1 && peaks[0].y == 1);
}

void test_polarity_inverts_the_detector() {
  // The same score map, read both ways round. RX and ACE return responses
  // where larger is better, so thresholding and suppression both have to
  // invert - and nothing else about the stage changes.
  const CubeShape shape = make_shape(7, 7, 1);
  std::vector<float> score(49, 0.5f);
  std::vector<std::int32_t> which(49, 0);

  // A low-valued well at (2,2) and a high-valued peak at (5,5).
  score[2 * 7 + 2] = 0.01f;
  score[5 * 7 + 5] = 0.99f;

  DetectionParams params;
  params.nms_radius = 1;

  params.threshold = 0.1f;
  params.polarity = ScorePolarity::LowerIsBetter;
  const std::vector<Detection> low = detect_cpu(score.data(), which.data(), shape, params);
  CHECK(low.size() == 1);
  if (low.size() == 1) CHECK(low[0].x == 2 && low[0].y == 2);

  params.threshold = 0.9f;
  params.polarity = ScorePolarity::HigherIsBetter;
  const std::vector<Detection> high = detect_cpu(score.data(), which.data(), shape, params);
  CHECK(high.size() == 1);
  if (high.size() == 1) CHECK(high[0].x == 5 && high[0].y == 5);
}

void test_polarity_picks_the_right_extremum_in_a_blob() {
  // A plateau with both a dip and a bump inside one suppression window, so a
  // detector that inverted the threshold but not the suppression would still
  // pass the test above while failing this one.
  const CubeShape shape = make_shape(9, 9, 1);
  std::vector<float> score(81, 0.5f);
  std::vector<std::int32_t> which(81, 0);
  for (int y = 3; y <= 5; ++y)
    for (int x = 3; x <= 5; ++x) score[y * 9 + x] = 0.8f;
  score[3 * 9 + 3] = 0.6f;  // lowest of the raised block
  score[5 * 9 + 5] = 0.95f; // highest of the raised block

  DetectionParams params;
  params.nms_radius = 3;
  params.threshold = 0.55f;
  params.polarity = ScorePolarity::HigherIsBetter;

  const std::vector<Detection> found = detect_cpu(score.data(), which.data(), shape, params);
  CHECK(found.size() == 1);
  if (found.size() == 1) {
    CHECK(found[0].x == 5 && found[0].y == 5);
    CHECK_CLOSE(found[0].score, 0.95, 1e-6);
  }
}

}  // namespace

int main() {
  test_polarity_inverts_the_detector();
  test_polarity_picks_the_right_extremum_in_a_blob();
  test_identical_spectrum_is_zero_angle();
  test_scale_invariance();
  test_orthogonal_and_zero_pixels();
  test_best_match_picks_the_nearest_target();
  test_mean_spectrum_over_mask();
  test_nms_keeps_one_pixel_per_blob();
  test_nms_breaks_ties_by_index();
  return check::finish("sam");
}
