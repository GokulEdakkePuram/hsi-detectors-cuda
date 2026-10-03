// The CPU reference detectors. Each is checked against a property that would
// be violated by a plausible implementation error, not just against itself.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "hsi/detectors.hpp"
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

/// A BSQ scene of correlated background with a handful of planted anomalies.
struct Scene {
  CubeShape shape;
  std::vector<float> cube;
  std::vector<std::size_t> anomalies;
  std::vector<float> anomaly_spectrum;
};

Scene make_scene(int h, int w, int bands, std::mt19937& rng) {
  Scene scene;
  scene.shape = make_shape(h, w, bands);
  const std::size_t pixels = scene.shape.pixels();
  scene.cube.assign(scene.shape.elements(), 0.0f);

  std::normal_distribution<float> noise(0.0f, 0.01f);
  std::normal_distribution<float> factor(0.0f, 1.0f);

  for (std::size_t p = 0; p < pixels; ++p) {
    const float a = factor(rng) * 0.05f;
    const float b = factor(rng) * 0.03f;
    for (int i = 0; i < bands; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(bands);
      scene.cube[static_cast<std::size_t>(i) * pixels + p] =
          0.30f + 0.15f * t + a * t * (1.0f - t) + b * std::sin(4.0f * t) + noise(rng);
    }
  }

  // Something spectrally unlike the background: an inverted ramp.
  scene.anomaly_spectrum.resize(static_cast<std::size_t>(bands));
  for (int i = 0; i < bands; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(bands);
    scene.anomaly_spectrum[static_cast<std::size_t>(i)] = 0.75f - 0.55f * t;
  }
  for (std::size_t p : {pixels / 7, pixels / 3, (2 * pixels) / 3}) {
    scene.anomalies.push_back(p);
    for (int i = 0; i < bands; ++i) {
      scene.cube[static_cast<std::size_t>(i) * pixels + p] =
          scene.anomaly_spectrum[static_cast<std::size_t>(i)] + noise(rng);
    }
  }
  return scene;
}

DetectorStatistics stats_for(const Scene& scene, const SpectralLibrary& library,
                             double loading) {
  SpectralMoments moments;
  accumulate_moments_cpu(scene.cube.data(), scene.shape, &moments);
  DetectorStatistics stats;
  std::string error;
  if (!build_statistics(moments, library, loading, &stats, &error)) {
    std::fprintf(stderr, "  build_statistics failed: %s\n", error.c_str());
  }
  return stats;
}

void test_rx_ranks_anomalies_above_background() {
  std::mt19937 rng(1);
  const Scene scene = make_scene(40, 48, 16, rng);
  const DetectorStatistics stats = stats_for(scene, SpectralLibrary{}, 1e-6);
  CHECK(stats.valid);
  if (!stats.valid) return;

  std::vector<float> score(scene.shape.pixels());
  rx_cpu(scene.cube.data(), scene.shape, stats, score.data());

  // Every planted anomaly must outrank every background pixel. RX needs no
  // signature, which is the whole point of it.
  float worst_anomaly = std::numeric_limits<float>::infinity();
  for (std::size_t p : scene.anomalies) worst_anomaly = std::min(worst_anomaly, score[p]);

  float best_background = 0.0f;
  for (std::size_t p = 0; p < scene.shape.pixels(); ++p) {
    const bool planted = std::find(scene.anomalies.begin(), scene.anomalies.end(), p) !=
                         scene.anomalies.end();
    if (!planted) best_background = std::max(best_background, score[p]);
  }
  CHECK(worst_anomaly > best_background);
}

void test_rx_background_mean_matches_the_band_count() {
  // For a background drawn from the distribution the covariance was estimated
  // from, RX averages to the number of bands. That is the chi-squared
  // expectation, and it catches a whitener that is off by a scale factor -
  // which ranking tests cannot see.
  std::mt19937 rng(2);
  const int bands = 12;
  const Scene scene = make_scene(60, 60, bands, rng);
  const DetectorStatistics stats = stats_for(scene, SpectralLibrary{}, 1e-9);
  CHECK(stats.valid);
  if (!stats.valid) return;

  std::vector<float> score(scene.shape.pixels());
  rx_cpu(scene.cube.data(), scene.shape, stats, score.data());

  double total = 0.0;
  std::size_t counted = 0;
  for (std::size_t p = 0; p < scene.shape.pixels(); ++p) {
    const bool planted = std::find(scene.anomalies.begin(), scene.anomalies.end(), p) !=
                         scene.anomalies.end();
    if (planted) continue;
    total += score[p];
    ++counted;
  }
  const double mean = total / static_cast<double>(counted);
  // Three anomalies inflate the covariance slightly, so allow a loose band.
  CHECK(mean > bands * 0.80 && mean < bands * 1.05);
}

void test_ace_is_bounded_and_peaks_on_its_target() {
  std::mt19937 rng(3);
  const Scene scene = make_scene(40, 48, 14, rng);

  SpectralLibrary library;
  library.targets.push_back({"anomaly", scene.anomaly_spectrum, {}});
  // A decoy that looks like the background, so the best-match reduction has
  // something to reject.
  Spectrum decoy;
  decoy.name = "background-like";
  decoy.values.resize(static_cast<std::size_t>(scene.shape.bands));
  for (int i = 0; i < scene.shape.bands; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(scene.shape.bands);
    decoy.values[static_cast<std::size_t>(i)] = 0.30f + 0.15f * t;
  }
  library.targets.push_back(std::move(decoy));

  const DetectorStatistics stats = stats_for(scene, library, 1e-6);
  CHECK(stats.valid);
  if (!stats.valid) return;

  std::vector<float> score(scene.shape.pixels());
  std::vector<std::int32_t> which(scene.shape.pixels());
  std::vector<float> rx(scene.shape.pixels());
  ace_best_cpu(scene.cube.data(), scene.shape, stats, score.data(), which.data(), rx.data());

  bool in_range = true;
  for (float v : score) {
    if (!(v >= 0.0f && v <= 1.0f)) in_range = false;
  }
  CHECK(in_range);

  // The planted pixels should score high and name target 0.
  bool found_all = true;
  for (std::size_t p : scene.anomalies) {
    if (score[p] < 0.9f) found_all = false;
    if (which[p] != 0) found_all = false;
  }
  CHECK(found_all);

  // The RX statistic handed back alongside must equal a standalone RX run -
  // they share the whitening, so a mismatch means the sharing is broken.
  std::vector<float> rx_alone(scene.shape.pixels());
  rx_cpu(scene.cube.data(), scene.shape, stats, rx_alone.data());
  double worst = 0.0;
  for (std::size_t p = 0; p < scene.shape.pixels(); ++p) {
    worst = std::max(worst, std::fabs(static_cast<double>(rx[p]) - rx_alone[p]));
  }
  CHECK(worst < 1e-3);
}

void test_cem_is_near_unity_on_its_target() {
  std::mt19937 rng(4);
  const Scene scene = make_scene(40, 48, 14, rng);
  SpectralLibrary library;
  library.targets.push_back({"anomaly", scene.anomaly_spectrum, {}});

  const DetectorStatistics stats = stats_for(scene, library, 1e-6);
  CHECK(stats.valid);
  if (!stats.valid) return;

  std::vector<float> score(scene.shape.pixels());
  std::vector<std::int32_t> which(scene.shape.pixels());
  cem_best_cpu(scene.cube.data(), scene.shape, stats, score.data(), which.data());

  // Unit gain on the target, and the background should be suppressed well
  // below it - that is what "minimum output energy" buys.
  for (std::size_t p : scene.anomalies) {
    CHECK(score[p] > 0.8f && score[p] < 1.2f);
  }
  double background_max = -1e30;
  for (std::size_t p = 0; p < scene.shape.pixels(); ++p) {
    const bool planted = std::find(scene.anomalies.begin(), scene.anomalies.end(), p) !=
                         scene.anomalies.end();
    if (!planted) background_max = std::max<double>(background_max, score[p]);
  }
  CHECK(background_max < 0.8);
}

void test_chi_squared_threshold_matches_the_closed_form() {
  // For two degrees of freedom the survival function is exp(-t/2) exactly, so
  // the quantile is -2 ln(rate) and there is a closed form to check against.
  for (double rate : {0.5, 0.1, 1e-3, 1e-6}) {
    const double want = -2.0 * std::log(rate);
    const double got = rx_threshold_for_false_alarm_rate(2, rate);
    CHECK_CLOSE(got, want, 1e-6 * std::max(1.0, want));
  }

  // Monotone in the rate, and for many bands it should sit near the mean.
  CHECK(rx_threshold_for_false_alarm_rate(50, 1e-6) >
        rx_threshold_for_false_alarm_rate(50, 1e-2));
  const double median = rx_threshold_for_false_alarm_rate(100, 0.5);
  CHECK(median > 90.0 && median < 100.0);
}

void test_detectors_reject_mismatched_statistics() {
  std::mt19937 rng(5);
  const Scene scene = make_scene(8, 8, 10, rng);
  DetectorStatistics empty;
  std::vector<float> score(scene.shape.pixels());

  bool threw = false;
  try {
    rx_cpu(scene.cube.data(), scene.shape, empty, score.data());
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

int main() {
  test_rx_ranks_anomalies_above_background();
  test_rx_background_mean_matches_the_band_count();
  test_ace_is_bounded_and_peaks_on_its_target();
  test_cem_is_near_unity_on_its_target();
  test_chi_squared_threshold_matches_the_closed_form();
  test_detectors_reject_mismatched_statistics();
  return check::finish("detectors");
}
