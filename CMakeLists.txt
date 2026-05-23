cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(model_test)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

file(GLOB INCLUDE_FILES "include/*.hpp")

add_executable(model_test
  ${INCLUDE_FILES}
  main.cpp)

target_include_directories(model_test PUBLIC
  include/
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(model_test PUBLIC
  ${OpenCV_LIBS}
  onnxruntime::onnxruntime
)

target_compile_options(model_test PUBLIC
  "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
