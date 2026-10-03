#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hsi/types.hpp"

namespace hsi {

/// Threshold an angle map into a compacted detection list on the device.
///
/// `d_count` must be zeroed before each call. On return it holds the number of
/// detections *found*, which may exceed `params.max_detections`; the caller
/// should clamp before reading `d_out`.
cudaError_t launch_detect(const float* d_score,
                          const std::int32_t* d_target_id, CubeShape shape,
                          DetectionParams params, Detection* d_out,
                          unsigned int* d_count, cudaStream_t stream);

}  // namespace hsi
