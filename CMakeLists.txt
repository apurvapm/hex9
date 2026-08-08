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

add_executable(test_alphabeta tests/test_alphabeta.cpp)
target_link_libraries(test_alphabeta PRIVATE hex_core)
add_test(NAME alphabeta COMMAND test_alphabeta)

if(HEX_SANITIZE)
  target_compile_options(test_alphabeta PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(test_alphabeta PRIVATE -fsanitize=address,undefined)
endif()

if(HEX_TSAN)
  target_compile_options(test_alphabeta PRIVATE -fsanitize=thread -g -O1)
  target_link_options(test_alphabeta PRIVATE -fsanitize=thread)
endif()

add_executable(test_encoding tests/test_encoding.cpp)
target_link_libraries(test_encoding PRIVATE hex_core)
add_test(NAME encoding COMMAND test_encoding)

if(HEX_SANITIZE)
  target_compile_options(test_encoding PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(test_encoding PRIVATE -fsanitize=address,undefined)
endif()

add_executable(test_puct tests/test_puct.cpp)
target_link_libraries(test_puct PRIVATE hex_core)
add_test(NAME puct COMMAND test_puct)

if(HEX_SANITIZE)
  target_compile_options(test_puct PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(test_puct PRIVATE -fsanitize=address,undefined)
endif()

if(HEX_TSAN)
  target_compile_options(test_puct PRIVATE -fsanitize=thread -g -O1)
  target_link_options(test_puct PRIVATE -fsanitize=thread)
endif()

# Native-only: constructs std::thread. The browser build is single-threaded by
# design -- GitHub Pages cannot set COOP/COEP, so there is no SharedArrayBuffer and
# no pthreads -- and a target that cannot link there is better excluded than
# silently broken.
if(NOT EMSCRIPTEN)
# Pins the parallel driver's reproducibility and bookkeeping. Runs under TSan too,
# so the ordering argument in parallel_selfplay.hpp is checked by a test target
# rather than only by a CI shell step.
add_executable(test_parallel tests/test_parallel.cpp)
target_link_libraries(test_parallel PRIVATE hex_core)
add_test(NAME parallel COMMAND test_parallel)

if(HEX_SANITIZE)
  target_compile_options(test_parallel PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(test_parallel PRIVATE -fsanitize=address,undefined)
endif()

if(HEX_TSAN)
  target_compile_options(test_parallel PRIVATE -fsanitize=thread -g -O1)
  target_link_options(test_parallel PRIVATE -fsanitize=thread)
endif()
endif()  # NOT EMSCRIPTEN

add_executable(play tools/play.cpp)
target_link_libraries(play PRIVATE hex_core)

# Native-only: constructs std::thread. The browser build is single-threaded by
# design -- GitHub Pages cannot set COOP/COEP, so there is no SharedArrayBuffer and
# no pthreads -- and a target that cannot link there is better excluded than
# silently broken.
if(NOT EMSCRIPTEN)
add_executable(selfplay tools/selfplay.cpp)
target_link_libraries(selfplay PRIVATE hex_core)

# The parallel driver shares block bookkeeping across workers as plain values and
# relies on std::barrier for the ordering. That reasoning is exactly what a race
# detector exists to check, so this target is TSan-able.
if(HEX_TSAN)
  target_compile_options(selfplay PRIVATE -fsanitize=thread -g -O1)
  target_link_options(selfplay PRIVATE -fsanitize=thread)
endif()
endif()  # NOT EMSCRIPTEN

# Writes the solved-position fixture that training/gate.py scores a checkpoint
# against. Needs no ONNX Runtime: it only runs the exhaustive solver, so the
# gate's ground truth is always buildable in CI.
add_executable(gate tools/gate.cpp)
target_link_libraries(gate PRIVATE hex_core)

# ONNX Runtime is optional. Without it the driver still runs in --heuristic
# mode, which is what the Python record tests use, so CI needs no extra
# dependency. Point ONNXRUNTIME_ROOT at an extracted release to enable it.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime C++ distribution")
if(ONNXRUNTIME_ROOT)
  target_compile_definitions(selfplay PRIVATE HEX_WITH_ONNX)
  target_include_directories(selfplay PRIVATE ${ONNXRUNTIME_ROOT}/include)
  target_link_directories(selfplay PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(selfplay PRIVATE onnxruntime)
  set_target_properties(selfplay PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
  message(STATUS "selfplay: ONNX Runtime enabled at ${ONNXRUNTIME_ROOT}")

  # Scores PUCT-with-network against the solved fixture. Added only when ORT is
  # available: unlike the fixture generator, this one cannot fall back to a
  # heuristic, because the whole point is to measure the trained agent.
  add_executable(gate_search tools/gate_search.cpp)
  target_link_libraries(gate_search PRIVATE hex_core)
  target_include_directories(gate_search PRIVATE ${ONNXRUNTIME_ROOT}/include)
  target_link_directories(gate_search PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(gate_search PRIVATE onnxruntime)
  set_target_properties(gate_search PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
else()
  message(STATUS "selfplay: building without ONNX Runtime (--heuristic only)")
  message(STATUS "gate_search: skipped (needs ONNXRUNTIME_ROOT)")
endif()

# Head-to-head matches between checkpoints. Builds without ONNX Runtime, in which
# case both sides must be the heuristic, which is enough to smoke-test the pairing.
# Native-only: constructs std::thread. The browser build is single-threaded by
# design -- GitHub Pages cannot set COOP/COEP, so there is no SharedArrayBuffer and
# no pthreads -- and a target that cannot link there is better excluded than
# silently broken.
if(NOT EMSCRIPTEN)
add_executable(arena tools/arena.cpp)
target_link_libraries(arena PRIVATE hex_core)
if(ONNXRUNTIME_ROOT)
  target_compile_definitions(arena PRIVATE HEX_WITH_ONNX)
  target_include_directories(arena PRIVATE ${ONNXRUNTIME_ROOT}/include)
  target_link_directories(arena PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(arena PRIVATE onnxruntime)
  set_target_properties(arena PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
endif()

endif()  # NOT EMSCRIPTEN -- arena

add_executable(bench_board bench/bench_board.cpp)
target_link_libraries(bench_board PRIVATE hex_core)

# Scaling measurements for the parallel driver. Builds without ONNX Runtime, in
# which case it sweeps the heuristic evaluator only; with ORT it also compares
# workers against session threads at a fixed core budget.
# Native-only: constructs std::thread. The browser build is single-threaded by
# design -- GitHub Pages cannot set COOP/COEP, so there is no SharedArrayBuffer and
# no pthreads -- and a target that cannot link there is better excluded than
# silently broken.
if(NOT EMSCRIPTEN)
add_executable(bench_parallel bench/bench_parallel.cpp)
target_link_libraries(bench_parallel PRIVATE hex_core)
if(NOT EMSCRIPTEN)
  target_compile_options(bench_parallel PRIVATE -O3 -march=native)
endif()
if(ONNXRUNTIME_ROOT)
  target_compile_definitions(bench_parallel PRIVATE HEX_WITH_ONNX)
  target_include_directories(bench_parallel PRIVATE ${ONNXRUNTIME_ROOT}/include)
  target_link_directories(bench_parallel PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(bench_parallel PRIVATE onnxruntime)
  set_target_properties(bench_parallel PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
endif()
endif()  # NOT EMSCRIPTEN -- bench_parallel

# Native-only tuning. Emscripten gets its own flags below.
if(NOT EMSCRIPTEN)
  target_compile_options(bench_board PRIVATE -O3 -march=native)
endif()

# Flags for the WebAssembly targets, recorded here so the native and browser builds
# stay in sync.
#
# Two sets, because they serve different consumers. The engine module is an ES6
# module loaded by a web worker. The test executables are run by node from the
# command line, where MODULARIZE and EXPORT_ES6 would produce something that cannot
# simply be executed -- and running the suites under wasm is the whole point, since
# it is what checks the random streams did not drift between libc++ and libstdc++.
#
# Deliberately absent everywhere: -pthread, -sUSE_PTHREADS, SharedArrayBuffer.
if(EMSCRIPTEN)
  set(HEX_WASM_FLAGS
      -O3
      -msimd128
      -sMODULARIZE=1
      -sEXPORT_ES6=1
      -sALLOW_MEMORY_GROWTH=1
      # node is in the list purely so the bindings are testable without a browser.
      # tests/test_wasm_engine.mjs drives the resumable search loop under node, which
      # is how the interface gets checked in CI; the browser only ever uses the web
      # and worker paths. The cost is a few kilobytes of environment shims.
      -sENVIRONMENT=web,worker,node)

  set(HEX_WASM_TEST_FLAGS -O3 -msimd128 -sALLOW_MEMORY_GROWTH=1)

  # The engine module the browser loads. An ES6 module so a worker can import it,
  # and embind because the interface is an object with methods rather than a handful
  # of C functions.
  add_executable(hex_engine wasm/hex_wasm.cpp)
  target_link_libraries(hex_engine PRIVATE hex_core)
  target_compile_options(hex_engine PRIVATE -O3 -msimd128)
  target_link_options(hex_engine PRIVATE ${HEX_WASM_FLAGS} --bind
                      -sEXPORT_NAME=createHexEngine
                      -sSTACK_SIZE=8388608)
  # .mjs rather than .js: the module is emitted with `export default`, and both node
  # and a browser <script type="module"> need the extension to agree with that.
  # Loading it as .js makes node treat ESM syntax as CommonJS and fail to parse.
  set_target_properties(hex_engine PROPERTIES OUTPUT_NAME "hex_engine" SUFFIX ".mjs")

  # NODERAWFS gives the suites the real filesystem so they can write and read the
  # golden fixtures by path, exactly as the native builds do. The larger stack is for
  # the 4x4 exhaustive solve's recursion depth.
  foreach(target test_board test_mcts test_alphabeta test_puct test_encoding gate
                 bench_board)
    target_compile_options(${target} PRIVATE ${HEX_WASM_TEST_FLAGS})
    target_link_options(${target} PRIVATE ${HEX_WASM_TEST_FLAGS}
                        -sNODERAWFS=1 -sSTACK_SIZE=8388608 -sEXIT_RUNTIME=1)
  endforeach()
endif()
