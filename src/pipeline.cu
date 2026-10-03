#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

#include "hsi/detect_cuda.hpp"
#include "hsi/pipeline.hpp"

namespace hsi {
namespace {

#define HSI_CUDA_TRY(expr, what)                                       \
  do {                                                                 \
    const cudaError_t hsi_err = (expr);                                \
    if (hsi_err != cudaSuccess) {                                      \
      if (error) {                                                     \
        *error = std::string(what) + ": " + cudaGetErrorString(hsi_err); \
      }                                                                \
      return false;                                                    \
    }                                                                  \
  } while (0)

float percentile(std::vector<float> values, double q) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  const double position = q * (static_cast<double>(values.size()) - 1.0);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double frac = position - static_cast<double>(lower);
  return static_cast<float>(values[lower] * (1.0 - frac) + values[upper] * frac);
}

/// One frame in flight: its own stream so the copies and kernels of different
/// frames can overlap, and its own buffers so nothing is reused underneath a
/// launch that has not finished.
struct Slot {
  cudaStream_t stream = nullptr;
  float* host_cube = nullptr;      ///< pinned; device-mapped in ZeroCopy mode
  float* device_cube = nullptr;    ///< aliases host_cube in ZeroCopy mode
  void* half_cube = nullptr;       ///< fp16 copy, only for SamVariant::Half
  float* d_score = nullptr;
  std::int32_t* d_target = nullptr;
  Detection* d_detections = nullptr;
  unsigned int* d_count = nullptr;
  Detection* h_detections = nullptr;  ///< pinned readback
  unsigned int* h_count = nullptr;
  float* h_score = nullptr;           ///< pinned, only when scoring offline

  cudaEvent_t ev_begin = nullptr, ev_uploaded = nullptr, ev_score = nullptr,
              ev_detect = nullptr, ev_done = nullptr;

  FrameMeta meta;
  bool busy = false;
  float ms_source = 0;
};

class CudaPipeline final : public Pipeline {
 public:
  CudaPipeline(FrameSource* source, const PipelineOptions& options)
      : source_(source), opt_(options) {}

  ~CudaPipeline() override {
    for (Slot& slot : slots_) {
      if (slot.stream) cudaStreamSynchronize(slot.stream);
    }
    for (Slot& slot : slots_) {
      if (slot.ev_begin) cudaEventDestroy(slot.ev_begin);
      if (slot.ev_uploaded) cudaEventDestroy(slot.ev_uploaded);
      if (slot.ev_score) cudaEventDestroy(slot.ev_score);
      if (slot.ev_detect) cudaEventDestroy(slot.ev_detect);
      if (slot.ev_done) cudaEventDestroy(slot.ev_done);
      if (slot.stream) cudaStreamDestroy(slot.stream);
      if (slot.host_cube) cudaFreeHost(slot.host_cube);
      if (opt_.memory == MemoryMode::Copy && slot.device_cube) cudaFree(slot.device_cube);
      if (slot.half_cube) cudaFree(slot.half_cube);
      if (slot.d_score) cudaFree(slot.d_score);
      if (slot.d_target) cudaFree(slot.d_target);
      if (slot.d_detections) cudaFree(slot.d_detections);
      if (slot.d_count) cudaFree(slot.d_count);
      if (slot.h_detections) cudaFreeHost(slot.h_detections);
      if (slot.h_count) cudaFreeHost(slot.h_count);
      if (slot.h_score) cudaFreeHost(slot.h_score);
    }
    if (d_targets_) cudaFree(d_targets_);
    if (d_target_norms_) cudaFree(d_target_norms_);
  }

  bool setup(const SpectralLibrary& library, std::string* error) {
    if (opt_.variant == SamVariant::BipDirect) {
      if (error) {
        *error =
            "the bip variant is a kernel-level comparison only; sources deliver "
            "BSQ, so run it through the benchmark rather than the pipeline";
      }
      return false;
    }

    // Before anything allocates and pins the context to its default flags.
    if (opt_.memory == MemoryMode::ZeroCopy) {
      int can_map = 0;
      HSI_CUDA_TRY(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0),
                   "query host-mapped memory support");
      if (!can_map) {
        if (error) *error = "this device cannot map host memory; use --memory=copy";
        return false;
      }
      HSI_CUDA_TRY(cudaSetDeviceFlags(cudaDeviceMapHost), "enable host mapping");
    }

    shape_ = source_->shape();
    if (!shape_.valid()) {
      if (error) *error = "source reported an invalid cube shape";
      return false;
    }
    if (library.size() == 0) {
      if (error) *error = "empty spectral library";
      return false;
    }
    if (library.bands() != shape_.bands) {
      if (error) {
        *error = "library has " + std::to_string(library.bands()) +
                 " bands but the source produces " + std::to_string(shape_.bands);
      }
      return false;
    }

    num_targets_ = library.size();
    stride_ = bsq_plane_stride(shape_);
    const std::size_t cube_elements = bsq_device_elements(shape_);
    const std::size_t pixels = shape_.pixels();

    // Targets live in constant memory for the vectorised kernels and in global
    // memory for the baseline, so upload both and let the launcher pick.
    const std::vector<float> flat = library.flatten();
    const std::vector<float> norms = library.norms();
    HSI_CUDA_TRY(cudaMalloc(&d_targets_, flat.size() * sizeof(float)), "alloc targets");
    HSI_CUDA_TRY(cudaMalloc(&d_target_norms_, norms.size() * sizeof(float)),
                 "alloc target norms");
    HSI_CUDA_TRY(cudaMemcpy(d_targets_, flat.data(), flat.size() * sizeof(float),
                            cudaMemcpyHostToDevice), "upload targets");
    HSI_CUDA_TRY(cudaMemcpy(d_target_norms_, norms.data(), norms.size() * sizeof(float),
                            cudaMemcpyHostToDevice), "upload target norms");
    if (opt_.variant != SamVariant::Baseline) {
      const cudaError_t err = upload_targets(flat.data(), norms.data(),
                                             num_targets_, shape_.bands);
      if (err != cudaSuccess) {
        if (error) {
          *error = std::string("upload targets to constant memory: ") +
                   cudaGetErrorString(err) +
                   " (library is " + std::to_string(flat.size()) +
                   " elements, capacity " +
                   std::to_string(max_constant_target_elements()) + ")";
        }
        return false;
      }
    }

    slots_.resize(std::max(1, opt_.num_streams));
    for (Slot& slot : slots_) {
      HSI_CUDA_TRY(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking),
                   "create stream");

      const unsigned host_flags = (opt_.memory == MemoryMode::ZeroCopy)
                                      ? cudaHostAllocMapped
                                      : cudaHostAllocDefault;
      HSI_CUDA_TRY(cudaHostAlloc(&slot.host_cube, cube_elements * sizeof(float),
                                 host_flags), "alloc pinned staging cube");

      if (opt_.memory == MemoryMode::ZeroCopy) {
        // No device cube at all: the kernels read the staging buffer in place.
        HSI_CUDA_TRY(cudaHostGetDevicePointer(&slot.device_cube, slot.host_cube, 0),
                     "map staging cube onto the device");
      } else {
        HSI_CUDA_TRY(cudaMalloc(&slot.device_cube, cube_elements * sizeof(float)),
                     "alloc device cube");
      }

      if (opt_.variant == SamVariant::Half) {
        HSI_CUDA_TRY(cudaMalloc(&slot.half_cube, cube_elements * sizeof(short)),
                     "alloc fp16 cube");
      }

      HSI_CUDA_TRY(cudaMalloc(&slot.d_score, pixels * sizeof(float)), "alloc angle map");
      HSI_CUDA_TRY(cudaMalloc(&slot.d_target, pixels * sizeof(std::int32_t)),
                   "alloc target map");
      HSI_CUDA_TRY(cudaMalloc(&slot.d_detections,
                              static_cast<std::size_t>(opt_.detection.max_detections) *
                                  sizeof(Detection)), "alloc detection list");
      HSI_CUDA_TRY(cudaMalloc(&slot.d_count, sizeof(unsigned int)), "alloc counter");
      HSI_CUDA_TRY(cudaHostAlloc(&slot.h_detections,
                                 static_cast<std::size_t>(opt_.detection.max_detections) *
                                     sizeof(Detection), cudaHostAllocDefault),
                   "alloc pinned detection readback");
      HSI_CUDA_TRY(cudaHostAlloc(&slot.h_count, sizeof(unsigned int), cudaHostAllocDefault),
                   "alloc pinned counter readback");
      if (opt_.return_score_map) {
        HSI_CUDA_TRY(cudaHostAlloc(&slot.h_score, pixels * sizeof(float),
                                   cudaHostAllocDefault),
                     "alloc pinned angle map readback");
      }

      const unsigned event_flags = opt_.time_stages ? cudaEventDefault
                                                    : cudaEventDisableTiming;
      HSI_CUDA_TRY(cudaEventCreateWithFlags(&slot.ev_begin, event_flags), "create event");
      HSI_CUDA_TRY(cudaEventCreateWithFlags(&slot.ev_uploaded, event_flags), "create event");
      HSI_CUDA_TRY(cudaEventCreateWithFlags(&slot.ev_score, event_flags), "create event");
      HSI_CUDA_TRY(cudaEventCreateWithFlags(&slot.ev_detect, event_flags), "create event");
      HSI_CUDA_TRY(cudaEventCreateWithFlags(&slot.ev_done, event_flags), "create event");
    }

    if (opt_.prefill > 0) {
      // Fill every slot's staging buffer once and then never call the source
      // again, so the measurement is of the GPU rather than of the generator.
      prefilled_ = true;
      for (Slot& slot : slots_) {
        if (!source_->read_into(slot.host_cube, stride_, &slot.meta)) {
          if (error) *error = "source ran dry while prefilling";
          return false;
        }
      }
    }
    return true;
  }

  bool run(std::uint64_t frames,
           const std::function<void(const FrameResult&)>& on_frame,
           std::string* error) override {
    const std::size_t pixels = shape_.pixels();
    const std::size_t cube_elements = bsq_device_elements(shape_);

    // Each run is measured on its own, so a warm-up run does not drag the
    // percentiles of the run that follows it.
    latencies_.clear();
    t_source_.clear();
    t_upload_.clear();
    t_sam_.clear();
    t_detect_.clear();
    t_download_.clear();
    t_gpu_.clear();
    stats_ = PipelineStats{};

    const auto wall_begin = std::chrono::steady_clock::now();
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    bool source_dry = false;

    while (completed < frames || (frames == 0 && !source_dry) ||
           (submitted > completed)) {
      const bool can_submit =
          !source_dry && (frames == 0 || submitted < frames);

      if (can_submit) {
        Slot& slot = slots_[submitted % slots_.size()];

        // Reclaim the slot before overwriting it. This is where the pipeline
        // actually blocks: with enough slots the GPU is still busy on earlier
        // frames while the host fills this one.
        if (slot.busy) {
          HSI_CUDA_TRY(cudaEventSynchronize(slot.ev_done), "wait for slot");
          if (!harvest(slot, on_frame, error)) return false;
          ++completed;
        }

        const auto source_begin = std::chrono::steady_clock::now();
        if (!prefilled_) {
          if (!source_->read_into(slot.host_cube, stride_, &slot.meta)) {
            source_dry = true;
            continue;
          }
        } else {
          slot.meta.index = submitted;
          slot.meta.timestamp_s = static_cast<double>(submitted) / 30.0;
        }
        slot.ms_source = std::chrono::duration<float, std::milli>(
                             std::chrono::steady_clock::now() - source_begin).count();

        HSI_CUDA_TRY(cudaEventRecord(slot.ev_begin, slot.stream), "record begin");

        if (opt_.memory == MemoryMode::Copy) {
          HSI_CUDA_TRY(cudaMemcpyAsync(slot.device_cube, slot.host_cube,
                                       cube_elements * sizeof(float),
                                       cudaMemcpyHostToDevice, slot.stream),
                       "upload cube");
        }
        HSI_CUDA_TRY(cudaEventRecord(slot.ev_uploaded, slot.stream), "record uploaded");

        const void* cube = slot.device_cube;
        if (opt_.variant == SamVariant::Half) {
          HSI_CUDA_TRY(launch_narrow_to_half(slot.device_cube, slot.half_cube,
                                             cube_elements, slot.stream),
                       "narrow cube to fp16");
          cube = slot.half_cube;
        }
        HSI_CUDA_TRY(launch_sam_best(opt_.variant, cube, shape_, d_targets_,
                                     d_target_norms_, num_targets_, slot.d_score,
                                     slot.d_target, slot.stream), "sam");
        HSI_CUDA_TRY(cudaEventRecord(slot.ev_score, slot.stream), "record sam");

        HSI_CUDA_TRY(cudaMemsetAsync(slot.d_count, 0, sizeof(unsigned int), slot.stream),
                     "reset detection counter");
        HSI_CUDA_TRY(launch_detect(slot.d_score, slot.d_target, shape_,
                                   opt_.detection, slot.d_detections, slot.d_count,
                                   slot.stream), "detect");
        HSI_CUDA_TRY(cudaEventRecord(slot.ev_detect, slot.stream), "record detect");

        // Only the compacted list comes back, never the angle map. At this
        // resolution the map is megabytes and the list is a few kilobytes.
        HSI_CUDA_TRY(cudaMemcpyAsync(slot.h_count, slot.d_count, sizeof(unsigned int),
                                     cudaMemcpyDeviceToHost, slot.stream),
                     "read back count");
        HSI_CUDA_TRY(cudaMemcpyAsync(slot.h_detections, slot.d_detections,
                                     static_cast<std::size_t>(opt_.detection.max_detections) *
                                         sizeof(Detection),
                                     cudaMemcpyDeviceToHost, slot.stream),
                     "read back detections");
        if (opt_.return_score_map) {
          HSI_CUDA_TRY(cudaMemcpyAsync(slot.h_score, slot.d_score,
                                       pixels * sizeof(float),
                                       cudaMemcpyDeviceToHost, slot.stream),
                       "read back angle map");
        }
        HSI_CUDA_TRY(cudaEventRecord(slot.ev_done, slot.stream), "record done");

        slot.busy = true;
        ++submitted;
        continue;
      }

      // Nothing more to submit: drain whatever is still in flight, in order.
      if (submitted == completed) break;
      Slot& slot = slots_[completed % slots_.size()];
      if (!slot.busy) break;
      HSI_CUDA_TRY(cudaEventSynchronize(slot.ev_done), "wait for slot");
      if (!harvest(slot, on_frame, error)) return false;
      ++completed;
    }

    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - wall_begin).count();
    stats_.frames = completed;
    stats_.wall_s = wall;
    stats_.fps = (wall > 0) ? static_cast<double>(completed) / wall : 0.0;
    const double cube_bytes = static_cast<double>(shape_.bytes_f32());
    stats_.cube_gb_per_s = (wall > 0) ? cube_bytes * completed / wall / 1e9 : 0.0;
    stats_.mpixel_per_s =
        (wall > 0) ? static_cast<double>(pixels) * completed / wall / 1e6 : 0.0;
    stats_.ms_p50 = percentile(latencies_, 0.50);
    stats_.ms_p95 = percentile(latencies_, 0.95);
    stats_.ms_p99 = percentile(latencies_, 0.99);
    stats_.ms_max = latencies_.empty() ? 0.0f
                                       : *std::max_element(latencies_.begin(),
                                                           latencies_.end());
    const auto mean = [&](const std::vector<float>& v) {
      return v.empty() ? 0.0f
                       : static_cast<float>(std::accumulate(v.begin(), v.end(), 0.0) /
                                            static_cast<double>(v.size()));
    };
    stats_.mean_source = mean(t_source_);
    stats_.mean_upload = mean(t_upload_);
    stats_.mean_score = mean(t_sam_);
    stats_.mean_detect = mean(t_detect_);
    stats_.mean_download = mean(t_download_);
    stats_.mean_gpu = mean(t_gpu_);
    return true;
  }

  PipelineStats stats() const override { return stats_; }

  std::string describe() const override {
    return std::string("pipeline(") + to_string(opt_.variant) + ", " +
           to_string(opt_.memory) + ", " + std::to_string(slots_.size()) +
           " streams, " + std::to_string(num_targets_) + " targets, " +
           std::to_string(shape_.height) + "x" + std::to_string(shape_.width) +
           "x" + std::to_string(shape_.bands) + ")";
  }

 private:
  bool harvest(Slot& slot, const std::function<void(const FrameResult&)>& on_frame,
               std::string* error) {
    FrameResult result;
    result.meta = slot.meta;
    result.ms_source = slot.ms_source;
    result.detections_found = *slot.h_count;

    const unsigned kept = std::min<unsigned>(
        result.detections_found, static_cast<unsigned>(opt_.detection.max_detections));
    result.detections.assign(slot.h_detections, slot.h_detections + kept);
    if (opt_.return_score_map && slot.h_score) {
      result.score_map.assign(slot.h_score, slot.h_score + shape_.pixels());
    }

    if (opt_.time_stages) {
      float ms = 0;
      HSI_CUDA_TRY(cudaEventElapsedTime(&ms, slot.ev_begin, slot.ev_uploaded),
                   "time upload");
      result.ms_upload = ms;
      HSI_CUDA_TRY(cudaEventElapsedTime(&ms, slot.ev_uploaded, slot.ev_score), "time sam");
      result.ms_score = ms;
      HSI_CUDA_TRY(cudaEventElapsedTime(&ms, slot.ev_score, slot.ev_detect), "time detect");
      result.ms_detect = ms;
      HSI_CUDA_TRY(cudaEventElapsedTime(&ms, slot.ev_detect, slot.ev_done),
                   "time download");
      result.ms_download = ms;
      HSI_CUDA_TRY(cudaEventElapsedTime(&ms, slot.ev_begin, slot.ev_done), "time gpu");
      result.ms_gpu = ms;
    }
    result.ms_wall = result.ms_source + result.ms_gpu;

    latencies_.push_back(result.ms_wall);
    t_source_.push_back(result.ms_source);
    t_upload_.push_back(result.ms_upload);
    t_sam_.push_back(result.ms_score);
    t_detect_.push_back(result.ms_detect);
    t_download_.push_back(result.ms_download);
    t_gpu_.push_back(result.ms_gpu);
    stats_.total_detections += result.detections.size();

    slot.busy = false;
    if (on_frame) on_frame(result);
    return true;
  }

  FrameSource* source_;
  PipelineOptions opt_;
  CubeShape shape_;
  std::size_t stride_ = 0;
  int num_targets_ = 0;
  bool prefilled_ = false;
  float* d_targets_ = nullptr;
  float* d_target_norms_ = nullptr;
  std::vector<Slot> slots_;
  std::vector<float> latencies_, t_source_, t_upload_, t_sam_, t_detect_,
      t_download_, t_gpu_;
  PipelineStats stats_;
};

#undef HSI_CUDA_TRY

}  // namespace

const char* to_string(MemoryMode mode) {
  switch (mode) {
    case MemoryMode::Copy: return "copy";
    case MemoryMode::ZeroCopy: return "zerocopy";
  }
  return "unknown";
}

bool parse_memory_mode(const std::string& name, MemoryMode* out) {
  if (name == "copy") { *out = MemoryMode::Copy; return true; }
  if (name == "zerocopy" || name == "zero-copy") { *out = MemoryMode::ZeroCopy; return true; }
  return false;
}

std::unique_ptr<Pipeline> Pipeline::create(FrameSource* source,
                                           const SpectralLibrary& library,
                                           const PipelineOptions& options,
                                           std::string* error) {
  if (!source) {
    if (error) *error = "null frame source";
    return nullptr;
  }
  auto pipeline = std::make_unique<CudaPipeline>(source, options);
  if (!pipeline->setup(library, error)) return nullptr;
  return pipeline;
}

}  // namespace hsi
