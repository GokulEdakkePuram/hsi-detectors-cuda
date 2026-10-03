// hsi_detect - run SAM target detection over a stream of hyperspectral frames.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "hsi/frame_source.hpp"
#include "hsi/hot_loader.hpp"
#include "hsi/pipeline.hpp"
#include "hsi/spectra_io.hpp"

namespace {

void usage() {
  std::puts(R"(hsi_detect - real-time spectral angle mapper target detection

  hsi_detect [options]

Source
  --source=synthetic|envi|hot   where frames come from            [synthetic]
  --envi=A.hdr,B.hdr            ENVI cubes to replay
  --hot-dir=DIR                 directory of snapshot-mosaic PNG frames
  --mosaic=N                    mosaic period for --source=hot            [4]
  --resident=N                  frames or cubes held in RAM               [4]
  --width=N --height=N          synthetic frame size               [512x512]
  --bands=N                     synthetic band count                    [128]
  --targets=N                   synthetic planted targets                 [3]
  --noise=F                     synthetic noise sigma                  [0.01]

Detection
  --library=FILE.csv            target spectra; required unless the source
                                supplies its own (--source=synthetic does)
  --threshold=RAD               spectral angle threshold               [0.10]
  --nms=N                       non-maximum suppression radius, 0 off      [2]
  --max-detections=N            detection list capacity                [4096]

Execution
  --variant=baseline|optimized|half                              [optimized]
  --memory=copy|zerocopy        zerocopy is for Jetson; on a discrete GPU it
                                turns every kernel read into a PCIe access
  --streams=N                   frames in flight                           [3]
  --frames=N                    frames to process                        [300]
  --warmup=N                    unmeasured frames first                   [30]
  --prefill                     stage the ring once and replay it, to measure
                                the GPU rather than the frame generator
  --csv=FILE                    write per-frame timings
  --print-detections=N          print the first N detections of the last frame
  --dump-score=FILE             write the last frame's angle map as raw float32
                                (height*width, raster order) for offline scoring
  --quiet
  --help)");
}

struct Args {
  std::map<std::string, std::string> values;

  bool has(const std::string& key) const { return values.count(key) != 0; }
  std::string str(const std::string& key, const std::string& fallback) const {
    const auto it = values.find(key);
    return (it == values.end() || it->second.empty()) ? fallback : it->second;
  }
  int integer(const std::string& key, int fallback) const {
    const auto it = values.find(key);
    return (it == values.end() || it->second.empty()) ? fallback : std::stoi(it->second);
  }
  float real(const std::string& key, float fallback) const {
    const auto it = values.find(key);
    return (it == values.end() || it->second.empty()) ? fallback : std::stof(it->second);
  }
};

Args parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string token = argv[i];
    if (token.rfind("--", 0) != 0) continue;
    token = token.substr(2);
    const std::size_t eq = token.find('=');
    if (eq == std::string::npos) args.values[token] = "";
    else args.values[token.substr(0, eq)] = token.substr(eq + 1);
  }
  return args;
}

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::string field;
  std::istringstream in(s);
  while (std::getline(in, field, ',')) {
    if (!field.empty()) out.push_back(field);
  }
  return out;
}

void print_device() {
  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    std::puts("device: unavailable");
    return;
  }
  // Peak theoretical bandwidth, which for a kernel this memory-bound is the
  // ceiling every number below should be read against.
  const double bandwidth =
      2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6;
  std::printf("device: %s  sm_%d%d  %d SMs  %.1f GB/s peak  %s memory\n",
              props.name, props.major, props.minor, props.multiProcessorCount,
              bandwidth, props.integrated ? "integrated" : "discrete");
  if (!props.integrated) {
    std::puts("note:   discrete GPU - --memory=zerocopy will be slow here");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  if (args.has("help") || args.has("h")) {
    usage();
    return 0;
  }
  const bool quiet = args.has("quiet");

  try {
    // ---- source -----------------------------------------------------------
    const std::string source_kind = args.str("source", "synthetic");
    std::unique_ptr<hsi::FrameSource> source;

    if (source_kind == "synthetic") {
      hsi::SyntheticOptions options;
      options.shape.height = args.integer("height", 512);
      options.shape.width = args.integer("width", 512);
      options.shape.bands = args.integer("bands", 128);
      options.num_targets = args.integer("targets", 3);
      options.noise_sigma = args.real("noise", 0.01f);
      source = hsi::make_synthetic_source(options);
    } else if (source_kind == "envi") {
      hsi::EnviReplayOptions options;
      options.hdr_paths = split_commas(args.str("envi", ""));
      options.max_resident = args.integer("resident", 4);
      if (options.hdr_paths.empty()) {
        std::fputs("--source=envi needs --envi=<header>[,<header>...]\n", stderr);
        return 2;
      }
      source = hsi::make_envi_replay_source(options);
    } else if (source_kind == "hot") {
      hsi::HotOptions options;
      options.directory = args.str("hot-dir", "");
      options.mosaic_period = args.integer("mosaic", 4);
      options.bands = args.integer("bands", options.mosaic_period * options.mosaic_period);
      options.max_resident = args.integer("resident", 64);
      if (options.directory.empty()) {
        std::fputs("--source=hot needs --hot-dir=<directory>\n", stderr);
        return 2;
      }
      source = hsi::make_hot_source(options);
    } else {
      std::fprintf(stderr, "unknown --source=%s\n", source_kind.c_str());
      return 2;
    }

    // ---- target library ---------------------------------------------------
    hsi::SpectralLibrary library;
    if (args.has("library")) {
      library = hsi::read_spectra_csv(args.str("library", ""));
    } else if (source->library()) {
      // The synthetic source knows what it planted, so it can supply its own
      // signatures and the tool stays runnable with no arguments at all.
      library = *source->library();
    } else {
      std::fputs("no target spectra: pass --library=<csv>\n", stderr);
      return 2;
    }

    // ---- pipeline ---------------------------------------------------------
    hsi::PipelineOptions options;
    if (!hsi::parse_sam_variant(args.str("variant", "optimized"), &options.variant)) {
      std::fputs("--variant must be baseline, optimized or half\n", stderr);
      return 2;
    }
    if (!hsi::parse_memory_mode(args.str("memory", "copy"), &options.memory)) {
      std::fputs("--memory must be copy or zerocopy\n", stderr);
      return 2;
    }
    options.num_streams = args.integer("streams", 3);
    options.prefill = args.has("prefill") ? 1 : 0;
    options.detection.threshold = args.real("threshold", 0.10f);
    options.detection.nms_radius = args.integer("nms", 2);
    options.detection.max_detections = args.integer("max-detections", 4096);
    options.return_score_map = args.has("dump-score");

    std::string error;
    auto pipeline = hsi::Pipeline::create(source.get(), library, options, &error);
    if (!pipeline) {
      std::fprintf(stderr, "pipeline setup failed: %s\n", error.c_str());
      return 1;
    }

    if (!quiet) {
      print_device();
      std::printf("source: %s\n", source->describe().c_str());
      std::printf("run:    %s\n", pipeline->describe().c_str());
    }

    // ---- warm up ----------------------------------------------------------
    const int warmup = args.integer("warmup", 30);
    if (warmup > 0) {
      // Clocks ramp, allocators settle and the first launch of each kernel
      // pays its JIT and module load. None of that is the steady state.
      if (!pipeline->run(static_cast<std::uint64_t>(warmup), nullptr, &error)) {
        std::fprintf(stderr, "warm-up failed: %s\n", error.c_str());
        return 1;
      }
    }

    // ---- measured run -----------------------------------------------------
    std::vector<hsi::Detection> last_detections;
    std::vector<float> last_angle_map;
    std::ofstream csv;
    if (args.has("csv")) {
      csv.open(args.str("csv", "timings.csv"));
      csv << "frame,detections,ms_source,ms_upload,ms_score,ms_detect,ms_download,"
             "ms_gpu,ms_wall\n";
    }

    const auto on_frame = [&](const hsi::FrameResult& result) {
      last_detections = result.detections;
      if (!result.score_map.empty()) last_angle_map = result.score_map;
      if (csv.is_open()) {
        csv << result.meta.index << "," << result.detections_found << ","
            << result.ms_source << "," << result.ms_upload << "," << result.ms_score
            << "," << result.ms_detect << "," << result.ms_download << ","
            << result.ms_gpu << "," << result.ms_wall << "\n";
      }
    };

    const std::uint64_t frames = static_cast<std::uint64_t>(args.integer("frames", 300));
    if (!pipeline->run(frames, on_frame, &error)) {
      std::fprintf(stderr, "run failed: %s\n", error.c_str());
      return 1;
    }

    const hsi::PipelineStats stats = pipeline->stats();
    std::printf(
        "\nframes %llu in %.3f s\n"
        "  throughput   %.1f fps   %.2f GB/s cube   %.1f Mpixel/s\n"
        "  latency ms   p50 %.3f   p95 %.3f   p99 %.3f   max %.3f\n"
        "  stage ms     source %.3f  upload %.3f  sam %.3f  detect %.3f  "
        "download %.3f  gpu %.3f\n"
        "  detections   %llu total, %.1f per frame\n",
        static_cast<unsigned long long>(stats.frames), stats.wall_s, stats.fps,
        stats.cube_gb_per_s, stats.mpixel_per_s, stats.ms_p50, stats.ms_p95,
        stats.ms_p99, stats.ms_max, stats.mean_source, stats.mean_upload,
        stats.mean_score, stats.mean_detect, stats.mean_download, stats.mean_gpu,
        static_cast<unsigned long long>(stats.total_detections),
        stats.frames ? static_cast<double>(stats.total_detections) / stats.frames : 0.0);

    if (args.has("dump-score") && !last_angle_map.empty()) {
      const std::string path = args.str("dump-score", "angle.f32");
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(last_angle_map.data()),
                static_cast<std::streamsize>(last_angle_map.size() * sizeof(float)));
      std::printf("\nwrote %zu x float32 angle map to %s (%dx%d)\n",
                  last_angle_map.size(), path.c_str(), source->shape().height,
                  source->shape().width);
    }

    const int print = args.integer("print-detections", 0);
    if (print > 0) {
      std::printf("\nlast frame, first %d of %zu detections:\n", print,
                  last_detections.size());
      for (int i = 0; i < print && i < static_cast<int>(last_detections.size()); ++i) {
        const hsi::Detection& d = last_detections[static_cast<std::size_t>(i)];
        std::printf("  (%4d,%4d) target %d  angle %.4f rad\n", d.x, d.y, d.target,
                    d.score);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
