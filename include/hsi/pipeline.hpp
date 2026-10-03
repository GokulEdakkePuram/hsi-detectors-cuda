#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hsi/frame_source.hpp"
#include "hsi/sam_cuda.hpp"
#include "hsi/types.hpp"

namespace hsi {

/// How frames get from the host into the kernels.
enum class MemoryMode {
  /// Pinned host staging plus an explicit asynchronous H2D copy. The only
  /// option on a discrete GPU, where host and device memory are genuinely
  /// separate and the copy crosses PCIe.
  Copy,

  /// Pinned, device-mapped host memory read in place by the kernels.
  ///
  /// The AGX Orin's CPU and GPU share one physical LPDDR5 pool, so an H2D copy
  /// there moves bytes from memory to the same memory - pure overhead, and at
  /// ~160 MB per frame the largest single cost in the pipeline. Mapping the
  /// staging buffer removes the copy entirely. On a discrete GPU this same
  /// setting is a trap: every kernel read becomes a PCIe transaction.
  ZeroCopy,
};

const char* to_string(MemoryMode mode);
bool parse_memory_mode(const std::string& name, MemoryMode* out);

struct PipelineOptions {
  SamVariant variant = SamVariant::Optimized;
  MemoryMode memory = MemoryMode::Copy;
  DetectionParams detection;

  /// Frames in flight. Each slot owns a stream, a staging buffer and a device
  /// cube, so this trades memory for overlap between the host's frame
  /// production and the GPU's work on earlier frames.
  int num_streams = 3;

  /// Non-zero fills the staging ring once at setup and then replays those
  /// num_streams frames without calling the source again.
  ///
  /// The synthetic generator writes tens of millions of floats per frame and
  /// can easily be slower than the GPU it feeds. Pre-staging separates "how
  /// fast can this GPU detect" from "how fast can this CPU invent data".
  /// 0 calls the source every frame, which is the honest end-to-end number.
  int prefill = 0;

  /// Record per-stage CUDA events. Costs a little, so it is opt-in.
  bool time_stages = true;

  /// Also read back the full per-pixel angle map with each frame.
  ///
  /// For scoring detection quality offline - ROC curves need every pixel's
  /// score, not just the ones over threshold. It adds a megabytes-per-frame
  /// D2H that the real-time path exists to avoid, so it is off by default and
  /// should stay off when measuring throughput.
  bool return_score_map = false;
};

struct FrameResult {
  FrameMeta meta;
  std::vector<Detection> detections;
  unsigned int detections_found = 0;  ///< may exceed detections.size() if saturated

  /// Per-pixel spectral angle, raster order. Empty unless the pipeline was
  /// built with return_score_map.
  std::vector<float> score_map;

  float ms_source = 0;    ///< host time spent producing the frame
  float ms_upload = 0;    ///< 0 in ZeroCopy mode, by construction
  float ms_score = 0;
  float ms_detect = 0;
  float ms_download = 0;
  float ms_gpu = 0;       ///< upload through download, on the device
  float ms_wall = 0;      ///< wall clock attributable to this frame
};

struct PipelineStats {
  std::uint64_t frames = 0;
  double wall_s = 0;
  double fps = 0;
  double cube_gb_per_s = 0;   ///< cube bytes retired per second
  double mpixel_per_s = 0;
  float ms_p50 = 0, ms_p95 = 0, ms_p99 = 0, ms_max = 0;
  float mean_source = 0, mean_upload = 0, mean_score = 0, mean_detect = 0,
        mean_download = 0, mean_gpu = 0;
  std::uint64_t total_detections = 0;
};

/// A frame-rate SAM detector.
///
/// Slots run ahead of each other: while the GPU works on frame N the host is
/// already producing frame N+1 into another slot's staging buffer. Results
/// come back in order.
class Pipeline {
 public:
  virtual ~Pipeline() = default;

  /// Build a pipeline for `source`, detecting the spectra in `library`.
  /// Returns null and fills `error` on failure.
  static std::unique_ptr<Pipeline> create(FrameSource* source,
                                          const SpectralLibrary& library,
                                          const PipelineOptions& options,
                                          std::string* error);

  /// Process `frames` frames, calling `on_frame` once per frame in order.
  /// Pass 0 to run until the source is exhausted. `on_frame` may be null.
  virtual bool run(std::uint64_t frames,
                   const std::function<void(const FrameResult&)>& on_frame,
                   std::string* error) = 0;

  virtual PipelineStats stats() const = 0;
  virtual std::string describe() const = 0;
};

}  // namespace hsi
