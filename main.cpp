#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "block_face_pipeline.hpp"
#include "transfer_protocol.hpp"

int main() {
  try {
    BlockFacePipeline pipeline{
        "model/block_detect.onnx", {"red", "blue", "empty"},
        "model/block_seg.onnx",    {"red_face", "blue_face"},
        "model/block_cls.onnx",    {"falsekfs", "r1kfs", "r2kfs", "unknown"}};

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open camera\n";
      return -1;
    }

    while (true) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty())
        break;

      auto blocks = pipeline.detectBlocks(frame);
      auto detect_frame = frame.clone();
      pipeline.getDetector().drawDetections(detect_frame, blocks); // 在原图上画检测框
      for (const auto &block : blocks) {
        try {
          auto cropped_block = pipeline.cropBlock(
              frame,
              block.box); // 获取裁剪图（可选，分类器也可以直接用全图+坐标）
          auto faces = pipeline.extractInstances(cropped_block, block);
          auto drawn_block = cropped_block.clone();
          pipeline.getSegmentor().drawInstances(drawn_block,
                                                faces); // 在原图上画分割结果
          cv::imshow("Cropped Block", drawn_block); // 显示裁剪图（调试用）
          for (const auto &face : faces) {
            try {
              auto result = pipeline.classifyFace(cropped_block, face, face.bbox);
              // 这里可以继续调用分类器对 face.warped_image 进行分类
              std::cout << "Detected face: class=" << result.class_name
                        << ", confidence=" << result.confidence << "\n";
            } catch (const std::exception &ex) {
              std::cerr << "Classification error: " << ex.what() << "\n";
            }
          }
        } catch (const std::exception &ex) {
          std::cerr << "Segmentation error: " << ex.what() << "\n";
        }
      }

      // 显示原图（可选）
      cv::imshow("Camera", detect_frame);
      if (cv::waitKey(1) == 27) // 按下 ESC 键退出
        break;
    }
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return -1;
  }
}
