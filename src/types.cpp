#include "hsi/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hsi {

const char* to_string(Interleave interleave) {
  switch (interleave) {
    case Interleave::Bsq: return "bsq";
    case Interleave::Bil: return "bil";
    case Interleave::Bip: return "bip";
  }
  return "unknown";
}

const char* to_string(ScorePolarity polarity) {
  switch (polarity) {
    case ScorePolarity::LowerIsBetter: return "lower-is-better";
    case ScorePolarity::HigherIsBetter: return "higher-is-better";
  }
  return "unknown";
}

namespace {

std::size_t offset_for(Interleave interleave, const CubeShape& s, int band,
                       int y, int x) {
  const std::size_t w = static_cast<std::size_t>(s.width);
  const std::size_t b = static_cast<std::size_t>(s.bands);
  switch (interleave) {
    case Interleave::Bsq:
      return static_cast<std::size_t>(band) * s.pixels() +
             static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x);
    case Interleave::Bil:
      return (static_cast<std::size_t>(y) * b + static_cast<std::size_t>(band)) *
                 w + static_cast<std::size_t>(x);
    case Interleave::Bip:
      return (static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)) *
                 b + static_cast<std::size_t>(band);
  }
  return 0;
}

}  // namespace

std::size_t HsiFrame::offset(int band, int y, int x) const {
  return offset_for(interleave, shape, band, y, x);
}

std::vector<float> SpectralLibrary::flatten() const {
  if (targets.empty()) return {};
  const int b = bands();
  for (const Spectrum& t : targets) {
    if (t.bands() != b) {
      throw std::invalid_argument(
          "SpectralLibrary: target '" + t.name + "' has " +
          std::to_string(t.bands()) + " bands, expected " + std::to_string(b));
    }
  }
  std::vector<float> out;
  out.reserve(static_cast<std::size_t>(size()) * b);
  for (const Spectrum& t : targets) {
    out.insert(out.end(), t.values.begin(), t.values.end());
  }
  return out;
}

std::vector<float> SpectralLibrary::norms() const {
  std::vector<float> out;
  out.reserve(targets.size());
  for (const Spectrum& t : targets) {
    double acc = 0.0;
    for (float v : t.values) acc += static_cast<double>(v) * v;
    out.push_back(static_cast<float>(std::sqrt(acc)));
  }
  return out;
}

void convert_interleave(const float* src, float* dst, CubeShape shape,
                        Interleave from, Interleave to) {
  if (!shape.valid()) throw std::invalid_argument("convert_interleave: bad shape");
  if (from == to) {
    std::copy(src, src + shape.elements(), dst);
    return;
  }
  // Walk the destination linearly so writes stay sequential; the strided side
  // is then the read, which the hardware prefetcher handles far better.
  for (int b = 0; b < shape.bands; ++b) {
    for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {
        dst[offset_for(to, shape, b, y, x)] =
            src[offset_for(from, shape, b, y, x)];
      }
    }
  }
}

void convert_frame(HsiFrame& frame, Interleave to) {
  if (frame.interleave == to) return;
  std::vector<float> out(frame.shape.elements());
  convert_interleave(frame.data.data(), out.data(), frame.shape,
                     frame.interleave, to);
  frame.data = std::move(out);
  frame.interleave = to;
}

}  // namespace hsi
