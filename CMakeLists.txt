cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(model_test)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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
  $<$<BOOL:${WIN32}>:ws2_32>
)

add_custom_command(TARGET model_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:onnxruntime::onnxruntime>
          $<TARGET_FILE_DIR:model_test>
)
# 拷贝 OpenCV DLL (Debug 和 Release 自动选)
if(EXISTS "${OPENCV_DLL_DEBUG}")
  add_custom_command(TARGET model_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL_DEBUG}"
            $<TARGET_FILE_DIR:model_test>
  )
endif()
if(EXISTS "${OPENCV_DLL_RELEASE}")
  add_custom_command(TARGET model_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL_RELEASE}"
            $<TARGET_FILE_DIR:model_test>
  )
endif()

# 自动拷贝 OpenCV DLL 到输出目录
set(OPENCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
set(OPENCV_DLL_BASE "opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
set(OPENCV_DLL_DEBUG "${OPENCV_BIN_DIR}/${OPENCV_DLL_BASE}d.dll")
set(OPENCV_DLL_RELEASE "${OPENCV_BIN_DIR}/${OPENCV_DLL_BASE}.dll")

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
  # 便于在windows平台上使用asio库
  $<$<BOOL:${WIN32}>:ws2_32>   
)

add_custom_command(TARGET weapon_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:onnxruntime::onnxruntime>
          $<TARGET_FILE_DIR:weapon_test>
)
# 拷贝 OpenCV DLL (Debug 和 Release 自动选)
if(EXISTS "${OPENCV_DLL_DEBUG}")
  add_custom_command(TARGET weapon_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL_DEBUG}"
            $<TARGET_FILE_DIR:weapon_test>
  )
endif()
if(EXISTS "${OPENCV_DLL_RELEASE}")
  add_custom_command(TARGET weapon_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL_RELEASE}"
            $<TARGET_FILE_DIR:weapon_test>
  )
endif()

target_compile_options(weapon_test PUBLIC
  "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
