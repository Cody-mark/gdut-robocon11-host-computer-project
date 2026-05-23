#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "block_face_pipeline.hpp"

int main() {
  std::cout << "Hello world!\n";
  BlockFacePipeline pipeline{"1.onnx", {}, "2.onnx", {}, "3.onnx", {}};
}
