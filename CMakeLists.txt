cmake_minimum_required(VERSION 3.16)
project(hex9 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(hex_core INTERFACE)
target_include_directories(hex_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(HEX_SANITIZE "Build tests with ASan and UBSan" OFF)
option(HEX_TSAN "Build with ThreadSanitizer (phase 3 onwards)" OFF)

# std::jthread and std::atomic link against pthreads on Linux. Requested here
# so the option exists before the threaded targets are added.
if(NOT EMSCRIPTEN)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(hex_core INTERFACE Threads::Threads)
endif()

# ASan and TSan are mutually exclusive; catch the mistake at configure time
# rather than with a confusing link error.
if(HEX_SANITIZE AND HEX_TSAN)
  message(FATAL_ERROR "HEX_SANITIZE and HEX_TSAN cannot be enabled together")
endif()

enable_testing()

add_executable(test_board tests/test_board.cpp)
target_link_libraries(test_board PRIVATE hex_core)
add_test(NAME board COMMAND test_board)

if(HEX_SANITIZE)
  target_compile_options(test_board PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(test_board PRIVATE -fsanitize=address,undefined)
endif()

if(HEX_TSAN)
  target_compile_options(test_board PRIVATE -fsanitize=thread -g -O1)
  target_link_options(test_board PRIVATE -fsanitize=thread)
endif()

add_executable(test_mcts tests/test_mcts.cpp)
target_link_libraries(test_mcts PRIVATE hex_core)
add_test(NAME mcts COMMAND test_mcts)

if(HEX_SANITIZE)
  target_compile_options(test_mcts PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(test_mcts PRIVATE -fsanitize=address,undefined)
endif()

if(HEX_TSAN)
  target_compile_options(test_mcts PRIVATE -fsanitize=thread -g -O1)
  target_link_options(test_mcts PRIVATE -fsanitize=thread)
endif()

add_executable(bench_board bench/bench_board.cpp)
target_link_libraries(bench_board PRIVATE hex_core)

# Native-only tuning. Emscripten gets its own flags below.
if(NOT EMSCRIPTEN)
  target_compile_options(bench_board PRIVATE -O3 -march=native)
endif()

# Flags for the WebAssembly target, recorded here so the native and browser
# builds stay in sync.
if(EMSCRIPTEN)
  set(HEX_WASM_FLAGS
      -O3
      -msimd128
      -sMODULARIZE=1
      -sEXPORT_ES6=1
      -sALLOW_MEMORY_GROWTH=1
      -sENVIRONMENT=web,worker)
endif()
