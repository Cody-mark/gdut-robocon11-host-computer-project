#pragma once

#include "geometry_utils.hpp"
#include "yolo_model_classifier.hpp"
#include "yolo_model_detector.hpp"
#include "yolo_model_segmentor.hpp"
#include <opencv2/core/types.hpp>

class BlockFacePipeline {
public:
  // 分类的结果：这张面属于哪个类别
  struct FaceResult {
    cv::Mat warped_face; // 透视矫正后的正方形面图像
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    cv::Rect source_bbox; // 在原图中检测框的位置（用于索引）
    int face_index = 0;   // 该检测框内的第几个面
  };

  // 构造函数：传入三个模型的路径和对应的类别名
  BlockFacePipeline(const std::filesystem::path &detect_model_path,
                    const std::vector<std::string> &detect_classes,
                    const std::filesystem::path &segment_model_path,
                    const std::vector<std::string>
                        &seg_classes, // 分割模型的类别名（可能有背景类）
                    const std::filesystem::path &classify_model_path,
                    const std::vector<std::string> &face_classes,
                    int warp_size = 512, float detect_conf = 0.5f,
                    float seg_conf = 0.3f, float min_face_area = 500)
      : detector_(detect_model_path, detect_classes),
        segmentor_(segment_model_path, seg_classes), // 实例分割，需要类别名
        classifier_(classify_model_path, face_classes), warp_size_(warp_size),
        detect_conf_(detect_conf), seg_conf_(seg_conf),
        min_face_area_(min_face_area) {}

  // 处理一张图像，返回所有检测到的面的分类结果
  std::vector<FaceResult> process(const cv::Mat &image) {
    std::vector<FaceResult> results;

    // 1. 检测方块
    auto detections = detector_.process(image, detect_conf_);
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto &det = detections[i];
      // 稍微扩大裁剪区域
      int pad = 10;
      cv::Rect roi = det.box;
      roi.x -= pad;
      roi.y -= pad;
      roi.width += 2 * pad;
      roi.height += 2 * pad;
      roi = roi & cv::Rect(0, 0, image.cols, image.rows); // 限制在图像内
      if (roi.width <= 0 || roi.height <= 0)
        continue;

      cv::Mat cropped = image(roi).clone();

      // 2. 实例分割面（在裁剪图上运行）
      auto instances = segmentor_.process(cropped, seg_conf_);
      for (size_t j = 0; j < instances.size(); ++j) {
        auto &inst = instances[j];
        if (inst.polygon.size() < 3)
          continue;

        std::vector<cv::Point> contour;
        contour.resize(inst.polygon.size());
        std::transform(inst.polygon.begin(), inst.polygon.end(),
                       contour.begin(), [](cv::Point2f p) {
                         return cv::Point{static_cast<int>(p.x),
                                          static_cast<int>(p.y)};
                       });

        // 检测面积大小
        if (cv::contourArea(inst.polygon) < min_face_area_)
          continue;

        // 近似为四边形
        auto quad = approxToQuad(contour);
        if (quad.size() != 4)
          continue;

        // 3. 透视矫正
        cv::Mat warped = warpPolygonToSquare(cropped, quad, warp_size_);

        // 4. 分类
        auto cls = classifier_.process(warped);
        if (cls.classId < 0)
          continue;

        FaceResult fr;
        fr.warped_face = warped;
        fr.class_id = cls.classId;
        fr.class_name = cls.className;
        fr.confidence = cls.confidence;
        fr.source_bbox = det.box; // 原始检测框（全图坐标）
        fr.face_index = static_cast<int>(j);
        results.push_back(fr);
      }
    }
    return results;
  }

private:
  YoloOnnxDetector detector_;
  YoloOnnxSegmentor segmentor_; // 实例分割
  YoloOnnxClassifier classifier_;
  int warp_size_;
  float detect_conf_;
  float seg_conf_;
  float min_face_area_;
};
