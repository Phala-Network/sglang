cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(phala_kvcache_incremental LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(Torch REQUIRED)

# Torch's legacy CMake package injects a broad auto-detected architecture list
# into CMAKE_CUDA_FLAGS when no GPU is visible. CUDA 13 no longer accepts some
# of those targets (for example sm_50). This extension supplies its explicit
# Hopper/Blackwell targets below, so remove only Torch's injected gencode pairs.
string(
  REGEX REPLACE " -gencode arch=[^ ]+" ""
  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}"
)

set(PHALA_FIXED_AOT_ROOT "" CACHE PATH "Pinned sgl-kernel AOT source root")
if(NOT PHALA_FIXED_AOT_ROOT)
  message(FATAL_ERROR "Set PHALA_FIXED_AOT_ROOT to the pinned kernel-aot-source directory")
endif()

set(TRANSFER_SOURCE "${PHALA_FIXED_AOT_ROOT}/csrc/kvcacheio/transfer.cu")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_BINDING "${GENERATED_DIR}/phala_kvcache_binding.cu")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_binding.py"
    --source "${TRANSFER_SOURCE}"
    --lock "${CMAKE_CURRENT_SOURCE_DIR}/source-lock.json"
    --output "${GENERATED_BINDING}"
  RESULT_VARIABLE GENERATE_RESULT
  OUTPUT_VARIABLE GENERATE_STDOUT
  ERROR_VARIABLE GENERATE_STDERR
)
if(NOT GENERATE_RESULT EQUAL 0)
  message(FATAL_ERROR "binding generation failed: ${GENERATE_STDOUT}\n${GENERATE_STDERR}")
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

add_library(phala_kvcache_ops SHARED "${GENERATED_BINDING}")
set_target_properties(phala_kvcache_ops PROPERTIES
  PREFIX ""
  OUTPUT_NAME "phala_kvcache_ops"
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES OFF
  CXX_STANDARD 17
  CUDA_STANDARD 17
)
target_compile_definitions(phala_kvcache_ops PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)
target_compile_options(phala_kvcache_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-DNDEBUG;-Xcompiler=-fPIC;-gencode=arch=compute_90,code=sm_90;-gencode=arch=compute_100f,code=sm_100f>
)
target_include_directories(phala_kvcache_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(phala_kvcache_ops PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
)

install(
  TARGETS phala_kvcache_ops
  LIBRARY DESTINATION phala_kvcache_incremental/_native
)
