#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hsi {

/// Band ordering of a hyperspectral cube in memory.
///
/// The GPU path works exclusively in BSQ. With one thread per pixel and a
/// band-major layout, neighbouring threads touch neighbouring addresses on
/// every band step, which is the only arrangement that fully coalesces.
/// Sensors and files commonly deliver BIL or BIP, so anything read from disk
/// or from a camera is transposed once on ingest.
enum class Interleave {
  Bsq,  ///< band[b][y][x] - band-major planes
  Bil,  ///< line[y][b][x] - band-interleaved by line
  Bip,  ///< pixel[y][x][b] - band-interleaved by pixel
};

const char* to_string(Interleave interleave);

/// Dimensions of a cube. Kept separate from the storage so kernels and
/// launch configuration can be reasoned about without owning any memory.
struct CubeShape {
  int height = 0;
  int width = 0;
  int bands = 0;

  std::size_t pixels() const {
    return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  }
  std::size_t elements() const {
    return pixels() * static_cast<std::size_t>(bands);
  }
  std::size_t bytes_f32() const { return elements() * sizeof(float); }

  bool valid() const { return height > 0 && width > 0 && bands > 0; }
  bool operator==(const CubeShape& other) const {
    return height == other.height && width == other.width &&
           bands == other.bands;
  }
  bool operator!=(const CubeShape& other) const { return !(*this == other); }
};

/// One frame of hyperspectral video, owned on the host.
///
/// Reflectance is stored as float32 and, by convention, already normalised by
/// the sensor's full-scale value. SAM is scale invariant per pixel, so an
/// unknown global gain does not change the result - but it does change what a
/// fixed detection threshold means, so ingest still normalises.
struct HsiFrame {
  CubeShape shape;
  Interleave interleave = Interleave::Bsq;
  std::vector<float> data;

  std::uint64_t index = 0;      ///< monotonic frame counter
  double timestamp_s = 0.0;     ///< seconds, source clock
  std::vector<float> wavelengths_nm;  ///< optional, size == shape.bands

  /// Index of element (band, y, x) for this frame's interleave.
  std::size_t offset(int band, int y, int x) const;

  float at(int band, int y, int x) const { return data[offset(band, y, x)]; }
};

/// A single reference (target) spectrum.
struct Spectrum {
  std::string name;
  std::vector<float> values;
  std::vector<float> wavelengths_nm;  ///< optional

  int bands() const { return static_cast<int>(values.size()); }
};

/// A set of target spectra evaluated against every pixel in one pass.
///
/// Reading a pixel's spectrum costs `bands * 4` bytes; scoring it against one
/// target costs roughly two FLOPs per band. A single target is therefore
/// hopelessly bandwidth bound, and the cost of adding targets is nearly free
/// until arithmetic catches up - which is exactly how a real detector is used,
/// so the library is the primary API rather than an afterthought.
struct SpectralLibrary {
  std::vector<Spectrum> targets;

  int size() const { return static_cast<int>(targets.size()); }
  int bands() const { return targets.empty() ? 0 : targets.front().bands(); }

  /// Flatten to target-major float32: out[t * bands + b].
  /// Throws std::invalid_argument if the targets disagree on band count.
  std::vector<float> flatten() const;

  /// L2 norm of each target spectrum, precomputed once on the host so the
  /// kernel never recomputes a constant.
  std::vector<float> norms() const;
};

/// One detected target pixel.
struct Detection {
  int x = 0;
  int y = 0;
  int target = 0;       ///< index into SpectralLibrary::targets
  float score = 0;  ///< spectral angle; smaller is a better match
};

/// Which direction of a detector score means "more like a target".
///
/// SAM returns an angle, so smaller is better. RX returns a squared distance
/// and ACE and CEM return filter responses, so larger is better. The
/// thresholding stage has to be told which, since it both compares against the
/// threshold and picks local extrema.
enum class ScorePolarity {
  LowerIsBetter,
  HigherIsBetter,
};

const char* to_string(ScorePolarity polarity);

/// Tuning for the thresholding stage.
struct DetectionParams {
  /// Pixels whose spectral angle is at or below this are candidates. Radians.
  float threshold = 0.10f;

  /// Radius of the non-maximum suppression window, in pixels. 0 disables it.
  ///
  /// A target bigger than one pixel trips the threshold across its whole
  /// footprint, so without suppression the output is thousands of pixels
  /// rather than a handful of targets. Suppression keeps the local best of
  /// each blob and turns the readback from a mask into a short list.
  int nms_radius = 2;

  /// Capacity of the output list. Detections beyond this are counted but not
  /// written, so a saturated frame degrades instead of corrupting memory.
  int max_detections = 4096;

  /// Which way round the score runs. Set from the detector's traits.
  ScorePolarity polarity = ScorePolarity::LowerIsBetter;
};

/// Convert a cube between interleaves. `src` and `dst` must not alias.
void convert_interleave(const float* src, float* dst, CubeShape shape,
                        Interleave from, Interleave to);

/// Convert a frame in place to the requested interleave (no-op if it matches).
void convert_frame(HsiFrame& frame, Interleave to);

}  // namespace hsi
