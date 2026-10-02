cmake_minimum_required(VERSION 3.20)

project(scala_ollama_direct LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED LLAMA_CPP_DIR)
  message(FATAL_ERROR "LLAMA_CPP_DIR must point to a llama.cpp source checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(
  "${LLAMA_CPP_DIR}"
  "${CMAKE_BINARY_DIR}/llama-build"
  EXCLUDE_FROM_ALL
)

add_library(
  scala_ollama_direct
  SHARED
  scala_ollama_direct.cpp
)

target_include_directories(
  scala_ollama_direct
  PRIVATE
  "${LLAMA_CPP_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  scala_ollama_direct
  PRIVATE
  llama
)

target_compile_features(scala_ollama_direct PRIVATE cxx_std_17)

set_target_properties(
  scala_ollama_direct
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ffi"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ffi"
)
