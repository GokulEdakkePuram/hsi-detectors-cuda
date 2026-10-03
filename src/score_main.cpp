// hsi_score - score one hyperspectral cube against a target library on the CPU.
//
// The reference scorer. It runs anywhere, needs no GPU, and produces exactly
// the angle map the CUDA pipeline should produce for the same inputs, so it
// serves three purposes: checking detection quality on real data before any
// hardware is rented, generating the expected output to diff a GPU run
// against, and giving the evaluation scripts something to consume on a laptop.
//
// It is not fast and is not trying to be. That is what the kernels are for.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "hsi/envi.hpp"
#include "hsi/sam.hpp"
#include "hsi/spectra_io.hpp"

namespace {

void usage() {
  std::puts(R"(hsi_score - CPU reference scorer for one hyperspectral cube

  hsi_score --cube=FILE.hdr --library=FILE.csv [options]

  --cube=FILE.hdr        ENVI cube to score
  --library=FILE.csv     target spectra
  --target=NAME          score against this one target only. Needed for a
                         single-class ROC: with a full library the angle map
                         holds the best match over all targets, which is a
                         different quantity from "how blood-like is this pixel"
  --threshold=RAD        detection threshold                        [0.10]
  --nms=N                non-maximum suppression radius, 0 off          [2]
  --dump-score=FILE      write the angle map as raw float32
  --dump-target=FILE     write the winning target index as raw int32
  --print=N              print the first N detections                 [10]
  --help)");
}

std::map<std::string, std::string> parse(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string token = argv[i];
    if (token.rfind("--", 0) != 0) continue;
    token = token.substr(2);
    const std::size_t eq = token.find('=');
    if (eq == std::string::npos) args[token] = "";
    else args[token.substr(0, eq)] = token.substr(eq + 1);
  }
  return args;
}

template <typename T>
void dump(const std::string& path, const std::vector<T>& values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
  const auto args = parse(argc, argv);
  const auto get = [&](const char* key, const std::string& fallback) {
    const auto it = args.find(key);
    return (it == args.end() || it->second.empty()) ? fallback : it->second;
  };
  if (args.count("help") || !args.count("cube") || !args.count("library")) {
    usage();
    return args.count("help") ? 0 : 2;
  }

  try {
    const hsi::HsiFrame frame = hsi::read_envi(get("cube", ""), hsi::Interleave::Bsq);
    hsi::SpectralLibrary library = hsi::read_spectra_csv(get("library", ""));

    // Restricting to one target matters more than it looks. With several
    // targets the angle map holds min over all of them, so a pixel that is
    // slightly more ketchup-like than blood-like records the ketchup angle.
    // Scoring that as "blood-likeness" measures the wrong thing.
    if (args.count("target")) {
      const std::string wanted = get("target", "");
      hsi::SpectralLibrary filtered;
      for (const hsi::Spectrum& target : library.targets) {
        if (target.name == wanted) filtered.targets.push_back(target);
      }
      if (filtered.targets.empty()) {
        std::fprintf(stderr, "no target named '%s' in the library. Available:",
                     wanted.c_str());
        for (const hsi::Spectrum& target : library.targets) {
          std::fprintf(stderr, " %s", target.name.c_str());
        }
        std::fputc('\n', stderr);
        return 1;
      }
      library = std::move(filtered);
    }

    if (library.bands() != frame.shape.bands) {
      std::fprintf(stderr,
                   "band mismatch: the cube has %d bands, the library has %d.\n"
                   "The two have to come from the same cleaning - see\n"
                   "scripts/prepare_hyperblood.py, which removes the fifteen\n"
                   "noisy bands and writes a library to match.\n",
                   frame.shape.bands, library.bands());
      return 1;
    }

    std::printf("cube     %dx%dx%d from %s\n", frame.shape.height, frame.shape.width,
                frame.shape.bands, get("cube", "").c_str());
    std::printf("library  %d targets:", library.size());
    for (const hsi::Spectrum& target : library.targets) {
      std::printf(" %s", target.name.c_str());
    }
    std::puts("");

    const std::vector<float> flat = library.flatten();
    const std::vector<float> norms = library.norms();
    const std::size_t pixels = frame.shape.pixels();
    std::vector<float> angle(pixels);
    std::vector<std::int32_t> target(pixels);

    const auto begin = std::chrono::steady_clock::now();
    hsi::sam_best_cpu(frame.data.data(), frame.shape, flat.data(), norms.data(),
                      library.size(), angle.data(), target.data());
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - begin).count();
    std::printf("scored   %zu pixels x %d targets in %.2f s\n", pixels,
                library.size(), seconds);

    hsi::DetectionParams params;
    params.threshold = std::stof(get("threshold", "0.10"));
    params.nms_radius = std::stoi(get("nms", "2"));
    const std::vector<hsi::Detection> detections =
        hsi::detect_cpu(angle.data(), target.data(), frame.shape, params);
    std::printf("found    %zu detections at threshold %.3f rad, nms radius %d\n",
                detections.size(), params.threshold, params.nms_radius);

    // Per target, so a library whose signatures overlap is visible here rather
    // than discovered later as a confusing detection count.
    std::vector<int> per_target(static_cast<std::size_t>(library.size()), 0);
    for (const hsi::Detection& d : detections) {
      if (d.target >= 0 && d.target < library.size()) ++per_target[static_cast<std::size_t>(d.target)];
    }
    for (int t = 0; t < library.size(); ++t) {
      std::printf("  %-22s %d\n", library.targets[static_cast<std::size_t>(t)].name.c_str(),
                  per_target[static_cast<std::size_t>(t)]);
    }

    const int print = std::stoi(get("print", "10"));
    for (int i = 0; i < print && i < static_cast<int>(detections.size()); ++i) {
      const hsi::Detection& d = detections[static_cast<std::size_t>(i)];
      std::printf("  (%4d,%4d) %-22s angle %.4f rad\n", d.x, d.y,
                  library.targets[static_cast<std::size_t>(d.target)].name.c_str(),
                  d.score);
    }

    if (args.count("dump-score")) {
      dump(get("dump-score", "angle.f32"), angle);
      std::printf("wrote angle map to %s\n", get("dump-score", "angle.f32").c_str());
    }
    if (args.count("dump-target")) {
      dump(get("dump-target", "target.i32"), target);
      std::printf("wrote target map to %s\n", get("dump-target", "target.i32").c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
