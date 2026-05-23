cmake_minimum_required(VERSION 3.24)

project(model_test)

find_package(OpenCV REQUIRED)

find_library(ONNXRUNTIME_LIB onnxruntime)

file(GLOB INCLUDE_FILES "include/*.hpp")

add_executable(model_test
  ${INCLUDE_FILES}
  main.cpp)

target_include_directories(model_test PUBLIC
  include/
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(model_test PUBLIC
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)
