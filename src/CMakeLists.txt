# Host-only core: data types, ENVI I/O, frame sources, CPU reference SAM.
add_library(hsi_core
  types.cpp
  envi.cpp
  frame_source.cpp
  hot_loader.cpp
  sam_cpu.cpp
  linalg.cpp
  spectra_io.cpp
)
target_include_directories(hsi_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(hsi_core PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
# zlib decompresses the IDAT stream of the snapshot-mosaic PNGs. It ships with
# every platform this targets, so it is a cheaper dependency than vendoring an
# inflate implementation.
find_package(ZLIB REQUIRED)
target_link_libraries(hsi_core PUBLIC Threads::Threads PRIVATE ZLIB::ZLIB)

# The reference scorer is host-only on purpose: it runs on a laptop, on a
# rented GPU box and on the Orin alike, which is what makes it useful as the
# thing a GPU run gets diffed against.
add_executable(hsi_score score_main.cpp)
target_link_libraries(hsi_score PRIVATE hsi_core)

if(NOT HSI_HAVE_CUDA)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

# CUDA kernels and the streaming pipeline.
add_library(hsi_cuda
  sam.cu
  detect.cu
  pipeline.cu
)
# PUBLIC cudart so plain C++ consumers such as main.cpp get the headers too.
target_link_libraries(hsi_cuda PUBLIC hsi_core CUDA::cudart)
set_target_properties(hsi_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(hsi_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

add_executable(hsi_detect main.cpp)
target_link_libraries(hsi_detect PRIVATE hsi_cuda)
