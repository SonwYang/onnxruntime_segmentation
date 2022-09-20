cmake_minimum_required(VERSION 3.0.0)
project(seg VERSION 0.1.0)

Set(ONNXRUNTIME_INCLUDE_DIR /home/yp/onnx/onnxruntime-segmentation/include/onnxruntime)
Set(ONNXRUNTIME_LIB_DIR /home/yp/onnx/onnxruntime-segmentation/lib/onnxruntime)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")


include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)

add_executable(seg ${PROJECT_SOURCE_DIR}/main.cpp)


target_link_libraries(seg ${OpenCV_LIBS})
target_include_directories(seg PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_include_directories(seg PRIVATE ./include)
target_link_libraries(seg "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
