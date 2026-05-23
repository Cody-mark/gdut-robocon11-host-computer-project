#pragma once

#include "yolo_model_interface.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class YoloOnnxSegmentor : public YoloOnnxProcessor<YoloOnnxSegmentor> {
  friend class YoloOnnxProcessor<YoloOnnxSegmentor>;

public:
  struct InstanceMask {
    int classId = -1;
    float score = 0.0f;
    cv::Rect box;                     // 原图坐标系下的边界框
    std::vector<cv::Point2f> polygon; // 原图坐标系下的多边形轮廓
    cv::Mat mask;                     // 原图坐标系下的二值掩码 (CV_8UC1, 0/255)
  };

  YoloOnnxSegmentor(const std::string &modelPath,
                    const std::vector<std::string> &classNames,
                    int numMaskCoeffs = 32)
      : YoloOnnxProcessor<YoloOnnxSegmentor>(modelPath),
        classNames_(classNames), numMaskCoeffs_(numMaskCoeffs) {}

  /** 在原图上绘制所有实例的轮廓和标签 */
  void drawInstances(cv::Mat &image,
                     const std::vector<InstanceMask> &instances) {
    for (const auto &inst : instances) {
      // 绘制多边形轮廓
      if (inst.polygon.size() >= 3) {
        std::vector<cv::Point> pts;
        for (auto &p : inst.polygon)
          pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        cv::polylines(image, pts, true, cv::Scalar(0, 255, 0), 2);
      }
      // 绘制类别标签
      std::string label =
          inst.classId >= 0 &&
                  inst.classId < static_cast<int>(classNames_.size())
              ? classNames_[inst.classId]
              : "?";
      label += ": " + cv::format("%.2f", inst.score);
      cv::putText(image, label, inst.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 255, 0), 2);
    }
  }

private:
  /**
   * 核心处理函数：单帧图像 → 实例分割结果
   */
  std::vector<InstanceMask> processImpl(const cv::Mat &frame, float confThres) {
    // 1. 预处理 (letterbox + BGR->RGB + normalize + HWC->CHW)
    float ratio, dw, dh;
    auto inputTensor = preprocess(frame, ratio, dw, dh);

    // 2. 推理（双输出）
    //    注意：如果你的模型输出名称不同，请修改下面两个字符串
    std::vector<std::string> outputNames = {"output0", "output1"};
    auto outputs = runInference(inputTensor, outputNames);
    if (outputs.size() < 2)
      return {}; // 必须有两个输出

    // 3. 解析 output0 : 检测结果 [1, N, 6 + numMaskCoeffs]
    auto detShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (detShape.size() != 3 || detShape[0] != 1)
      return {};
    size_t numDet = static_cast<size_t>(detShape[1]);
    size_t detLen = static_cast<size_t>(detShape[2]);
    if (detLen < 6 + numMaskCoeffs_)
      return {};

    const float *detData = outputs[0].GetTensorData<float>();

    // 4. 解析 output1 : Proto 特征图 [1, numMaskCoeffs, H, W]
    auto protoShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (protoShape.size() != 4 || protoShape[0] != 1 ||
        protoShape[1] != numMaskCoeffs_)
      return {};
    int protoH = static_cast<int>(protoShape[2]);
    int protoW = static_cast<int>(protoShape[3]);
    const float *protoData = outputs[1].GetTensorData<float>();

    // 5. 遍历每个检测，生成实例 mask
    std::vector<InstanceMask> results;
    int numClasses = static_cast<int>(classNames_.size());

    for (size_t i = 0; i < numDet; ++i) {
      const float *row = detData + i * detLen;

      float score = row[4];
      if (score < confThres)
        continue;

      int cls = static_cast<int>(std::round(row[5]));
      cls = std::max(0, std::min(cls, numClasses - 1));

      // 坐标反变换 (letterbox → 原图)
      float x1 = (row[0] - dw) / ratio;
      float y1 = (row[1] - dh) / ratio;
      float x2 = (row[2] - dw) / ratio;
      float y2 = (row[3] - dh) / ratio;

      int ix1 = std::max(
          0, std::min(static_cast<int>(std::round(x1)), frame.cols - 1));
      int iy1 = std::max(
          0, std::min(static_cast<int>(std::round(y1)), frame.rows - 1));
      int ix2 = std::max(
          0, std::min(static_cast<int>(std::round(x2)), frame.cols - 1));
      int iy2 = std::max(
          0, std::min(static_cast<int>(std::round(y2)), frame.rows - 1));

      if (ix2 <= ix1 || iy2 <= iy1)
        continue;

      // 提取 mask 系数
      std::vector<float> maskCoeffs(numMaskCoeffs_);
      for (int k = 0; k < numMaskCoeffs_; ++k)
        maskCoeffs[k] = row[6 + k];

      // 从 proto 特征图解码出整个图像的 mask，然后裁剪到检测框
      cv::Mat fullMask = decodeMask(protoData, protoH, protoW, maskCoeffs,
                                    cv::Size(frame.cols, frame.rows));

      // 裁剪检测框区域，获取该实例的局部掩码
      cv::Rect roi(ix1, iy1, ix2 - ix1, iy2 - iy1);
      cv::Mat roiMask = fullMask(roi).clone();

      // 提取轮廓（在局部掩码中）
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(roiMask, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);
      if (contours.empty())
        continue;

      // 选择面积最大的轮廓（通常就是该实例的 mask）
      auto biggest = std::max_element(
          contours.begin(), contours.end(), [](const auto &a, const auto &b) {
            return cv::contourArea(a) < cv::contourArea(b);
          });

      // 轮廓点坐标转换回全图坐标
      std::vector<cv::Point2f> polygon;
      for (auto &p : *biggest)
        polygon.emplace_back(p.x + ix1, p.y + iy1);

      // 构建结果
      InstanceMask inst;
      inst.classId = cls;
      inst.score = score;
      inst.box = cv::Rect(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
      inst.polygon = polygon;
      inst.mask = fullMask; // 全图掩码（可用于后续可视化）
      results.push_back(inst);
    }

    return results;
  }

  /**
   * 从 Proto 特征图和 mask 系数重建实例的二值掩码（全图尺寸）
   */
  cv::Mat decodeMask(const float *protoData, int protoH, int protoW,
                     const std::vector<float> &coeffs, cv::Size outputSize) {
    // 加权求和 proto 通道
    cv::Mat maskProto(protoH, protoW, CV_32F, cv::Scalar(0));
    for (int c = 0; c < numMaskCoeffs_; ++c) {
      const float *channel = protoData + c * protoH * protoW;
      for (int y = 0; y < protoH; ++y) {
        float *dst = maskProto.ptr<float>(y);
        const float *src = channel + y * protoW;
        for (int x = 0; x < protoW; ++x) {
          dst[x] += src[x] * coeffs[c];
        }
      }
    }

    // sigmoid 激活
    cv::Mat maskFloat;
    cv::exp(-maskProto, maskProto);
    maskFloat = 1.0f / (1.0f + maskProto);

    // 缩放到输出尺寸（通常为原图大小）
    cv::Mat resized;
    cv::resize(maskFloat, resized, outputSize, 0, 0, cv::INTER_LINEAR);

    // 二值化
    cv::Mat binary = resized > 0.5f;
    binary.convertTo(binary, CV_8U, 255);
    return binary;
  }

  std::vector<std::string> classNames_;
  int numMaskCoeffs_;
};
