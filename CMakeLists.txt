cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(model_test)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# 默认路径，避免每次手动传 -D 参数
if(NOT DEFINED OpenCV_DIR)
  set(OpenCV_DIR "E:/opencv/opencv/build" CACHE PATH "OpenCV install prefix")
endif()
if(NOT DEFINED onnxruntime_DIR)
  set(onnxruntime_DIR "E:/onnxruntime-win-x64-1.26.0/onnxruntime-win-x64-1.26.0" CACHE PATH "onnxruntime install prefix")
endif()

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_library(asio INTERFACE)
target_include_directories(asio INTERFACE ${CMAKE_SOURCE_DIR}/asio/include)

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
  asio
)

add_custom_command(TARGET model_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:onnxruntime::onnxruntime>
          $<TARGET_FILE_DIR:model_test>
)

target_compile_options(model_test PUBLIC
  "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)

# ---- weapon 检测程序 ----
add_executable(weapon_test
  ${INCLUDE_FILES}
  main_weapon.cpp)

target_include_directories(weapon_test PUBLIC
  include/
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(weapon_test PUBLIC
  ${OpenCV_LIBS}
  onnxruntime::onnxruntime
  asio
)

add_custom_command(TARGET weapon_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:onnxruntime::onnxruntime>
          $<TARGET_FILE_DIR:weapon_test>
)

target_compile_options(weapon_test PUBLIC
  "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
