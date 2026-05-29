cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
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

add_custom_command(TARGET model_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:onnxruntime::onnxruntime>
          $<TARGET_FILE_DIR:model_test>
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
)

add_custom_command(TARGET weapon_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:onnxruntime::onnxruntime>
          $<TARGET_FILE_DIR:weapon_test>
)

target_compile_options(model_test PUBLIC
  "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
