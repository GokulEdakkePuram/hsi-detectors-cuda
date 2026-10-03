// The synthetic source is both a load generator and, because it knows where it
// planted its targets, a self-validating end-to-end check of the detector.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "hsi/frame_source.hpp"
#include "hsi/sam.hpp"

using namespace hsi;

namespace {

SyntheticOptions small_options() {
  SyntheticOptions options;
  options.shape.height = 96;
  options.shape.width = 128;
  options.shape.bands = 24;
  options.num_targets = 3;
  options.target_radius = 5;
  options.seed = 4242;
  return options;
}

void test_shape_and_determinism() {
  const SyntheticOptions options = small_options();
  auto a = make_synthetic_source(options);
  auto b = make_synthetic_source(options);

  CHECK(a->shape() == options.shape);
  CHECK(a->interleave() == Interleave::Bsq);
  CHECK(a->library() != nullptr);
  CHECK(a->library()->size() == options.num_targets);

  std::vector<float> fa(options.shape.elements());
  std::vector<float> fb(options.shape.elements());
  FrameMeta ma, mb;
  CHECK(a->read_into(fa.data(), options.shape.pixels(), &ma));
  CHECK(b->read_into(fb.data(), options.shape.pixels(), &mb));

  // Same seed, same frame - including the noise, which is generated per band
  // so the result cannot depend on how the work was split across threads.
  CHECK(fa == fb);
  CHECK(ma.index == 0 && mb.index == 0);
}

void test_length_bounds_the_stream() {
  SyntheticOptions options = small_options();
  options.length = 3;
  auto source = make_synthetic_source(options);

  std::vector<float> frame(options.shape.elements());
  int produced = 0;
  while (source->read_into(frame.data(), options.shape.pixels(), nullptr)) ++produced;
  CHECK(produced == 3);

  source->reset();
  CHECK(source->read_into(frame.data(), options.shape.pixels(), nullptr));
}

void test_targets_move_between_frames() {
  auto source = make_synthetic_source(small_options());
  const CubeShape shape = source->shape();
  std::vector<float> frame(shape.elements());

  source->read_into(frame.data(), shape.pixels(), nullptr);
  const std::vector<std::uint8_t> first(source->truth_mask(),
                                        source->truth_mask() + shape.pixels());
  source->read_into(frame.data(), shape.pixels(), nullptr);
  const std::uint8_t* second = source->truth_mask();

  bool moved = false;
  for (std::size_t p = 0; p < shape.pixels(); ++p) {
    if (first[p] != second[p]) moved = true;
  }
  CHECK(moved);
}

void test_detector_finds_the_planted_targets() {
  // The whole chain on real-ish data: generate a frame with known targets,
  // score it with the reference SAM, threshold it, and check every detection
  // lands on a pixel the generator actually marked.
  auto source = make_synthetic_source(small_options());
  const CubeShape shape = source->shape();
  const SpectralLibrary& library = *source->library();

  std::vector<float> frame(shape.elements());
  CHECK(source->read_into(frame.data(), shape.pixels(), nullptr));
  const std::uint8_t* truth = source->truth_mask();

  const std::vector<float> flat = library.flatten();
  const std::vector<float> norms = library.norms();

  std::vector<float> angle(shape.pixels());
  std::vector<std::int32_t> which(shape.pixels());
  sam_best_cpu(frame.data(), shape, flat.data(), norms.data(), library.size(),
               angle.data(), which.data());

  DetectionParams params;
  params.threshold = 0.05f;
  params.nms_radius = 4;
  const std::vector<Detection> found =
      detect_cpu(angle.data(), which.data(), shape, params);

  // Recall and precision, not an exact count. Sensor noise puts several local
  // minima inside a blob wider than the suppression window, so pinning the
  // count would only be testing that the noise happens not to do that.
  std::vector<bool> seen(static_cast<std::size_t>(library.size()), false);
  bool all_on_target = true;
  bool all_correctly_labelled = true;
  for (const Detection& d : found) {
    const std::size_t p = static_cast<std::size_t>(d.y) * shape.width + d.x;
    if (truth[p] == 0) {
      all_on_target = false;  // a false positive
      continue;
    }
    if (truth[p] - 1 != d.target) all_correctly_labelled = false;
    seen[static_cast<std::size_t>(d.target)] = true;
  }
  CHECK(all_on_target);
  CHECK(all_correctly_labelled);

  bool every_target_found = true;
  for (bool hit : seen) {
    if (!hit) every_target_found = false;
  }
  CHECK(every_target_found);

  // Suppression still has to do its job: without it every pixel of every blob
  // would be emitted, which is hundreds.
  CHECK(found.size() < static_cast<std::size_t>(library.size()) * 8);

  // And the background must not be a near miss: unmasked pixels should sit far
  // above the detection threshold, otherwise the margin is luck.
  double worst_background = 1e9;
  for (std::size_t p = 0; p < shape.pixels(); ++p) {
    if (truth[p] == 0) worst_background = std::min<double>(worst_background, angle[p]);
  }
  CHECK(worst_background > params.threshold);
}

/// Write a BSQ ENVI pair whose value at (b, y, x) identifies all three, so a
/// mis-strided or transposed replay cannot pass.
std::filesystem::path write_cube(const std::filesystem::path& dir,
                                 const std::string& name, CubeShape shape,
                                 float offset) {
  std::vector<float> data(shape.elements());
  for (int b = 0; b < shape.bands; ++b)
    for (int y = 0; y < shape.height; ++y)
      for (int x = 0; x < shape.width; ++x)
        data[(static_cast<std::size_t>(b) * shape.height + y) * shape.width + x] =
            offset + b * 100.0f + y * 10.0f + x;

  const std::filesystem::path stem = dir / name;
  {
    std::ofstream out(stem.string() + ".float", std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
  }
  {
    std::ofstream out(stem.string() + ".hdr");
    out << "ENVI\nsamples = " << shape.width << "\nlines = " << shape.height
        << "\nbands = " << shape.bands
        << "\ndata type = 4\ninterleave = bsq\nbyte order = 0\n";
  }
  return stem.string() + ".hdr";
}

void test_envi_replay_cycles_and_honours_stride() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "hsi_test_replay";
  fs::create_directories(dir);

  CubeShape shape;
  shape.height = 3;
  shape.width = 5;  // not a multiple of 8, so pixels() != a padded stride
  shape.bands = 4;

  EnviReplayOptions options;
  options.hdr_paths = {write_cube(dir, "one", shape, 0.0f).string(),
                       write_cube(dir, "two", shape, 1000.0f).string()};
  options.loop = true;
  auto source = make_envi_replay_source(options);
  CHECK(source->shape() == shape);
  CHECK(source->interleave() == Interleave::Bsq);

  // Replay with a padded plane stride, the way the pipeline does on Jetson.
  // Every band plane has to land at its own stride, and the gaps between them
  // must be left alone rather than written through.
  const std::size_t stride = shape.pixels() + 3;
  std::vector<float> buffer(stride * shape.bands, -1.0f);

  for (int round = 0; round < 2; ++round) {
    for (float offset : {0.0f, 1000.0f}) {
      CHECK(source->read_into(buffer.data(), stride, nullptr));
      bool correct = true;
      for (int b = 0; b < shape.bands; ++b) {
        for (std::size_t p = 0; p < shape.pixels(); ++p) {
          const int y = static_cast<int>(p) / shape.width;
          const int x = static_cast<int>(p) % shape.width;
          const float want = offset + b * 100.0f + y * 10.0f + x;
          if (buffer[static_cast<std::size_t>(b) * stride + p] != want) correct = false;
        }
        // The padding between planes must still be untouched.
        for (std::size_t p = shape.pixels(); p < stride; ++p) {
          if (buffer[static_cast<std::size_t>(b) * stride + p] != -1.0f) correct = false;
        }
      }
      CHECK(correct);
    }
  }

  // A cube of a different shape would silently break every preallocated
  // buffer downstream, so it has to be refused rather than reshaped.
  CubeShape other = shape;
  other.bands = 6;
  EnviReplayOptions bad = options;
  bad.hdr_paths.push_back(write_cube(dir, "odd", other, 0.0f).string());
  bool refused = false;
  try {
    make_envi_replay_source(bad);
  } catch (const std::exception&) {
    refused = true;
  }
  CHECK(refused);

  fs::remove_all(dir);
}

}  // namespace

int main() {
  test_envi_replay_cycles_and_honours_stride();
  test_shape_and_determinism();
  test_length_bounds_the_stream();
  test_targets_move_between_frames();
  test_detector_finds_the_planted_targets();
  return check::finish("sources");
}
