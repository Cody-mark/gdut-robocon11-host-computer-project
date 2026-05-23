#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>


// 封装 ONNXRuntime + YOLO 前后处理
template <typename Derived> class YoloOnnxProcessor {
public:
  // letterbox 结果：
  // image: 缩放+填充后的图
  // ratio: 缩放比例（原图 -> 网络输入图）
  // dw/dh: 左右与上下方向的填充偏移（用于把框映射回原图）
  struct LetterboxResult {
    cv::Mat image;
    float ratio;
    float dw;
    float dh;
  };

  int inputH() const { return inputH_; }
  int inputW() const { return inputW_; }
  const std::string &inputName() const { return inputName_; }

  // 对单帧图像做检测
  auto process(const cv::Mat &frame, float confThres = 0.25f) {
    return static_cast<Derived *>(this)->processImpl(frame, confThres);
  }

protected:
  // 构造函数：加载模型、读取输入输出信息
  explicit YoloOnnxProcessor(const std::filesystem::path &modelPath)
      : env_(ORT_LOGGING_LEVEL_WARNING, "yolo-onnx-cpp"), sessionOptions_(),
        session_(nullptr), memoryInfo_(Ort::MemoryInfo::CreateCpu(
                               OrtArenaAllocator, OrtMemTypeDefault)) {
    // 打开图优化，可减少推理耗时
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 线程数可按设备调整；嵌入式常见做法是先设 1，再按性能调参
    sessionOptions_.SetIntraOpNumThreads(1);

    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAllocated = session_.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session_.GetOutputNameAllocated(0, allocator);
    inputName_ = inputNameAllocated.get();
    outputName_ = outputNameAllocated.get();

    // 一般是 [1, 3, H, W]
    auto inputInfo = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> inputShape = inputInfo.GetShape();
    inputH_ = (inputShape.size() >= 4 && inputShape[2] > 0)
                  ? static_cast<int>(inputShape[2])
                  : 320;
    inputW_ = (inputShape.size() >= 4 && inputShape[3] > 0)
                  ? static_cast<int>(inputShape[3])
                  : 320;
  }

  ~YoloOnnxProcessor() = default;

  // letterbox：保持长宽比缩放后，再用灰边填充到固定尺寸
  // 这样能减少几何变形，通常比直接拉伸更稳
  static LetterboxResult
  letterbox(const cv::Mat &src, int newH, int newW,
            const cv::Scalar &color = cv::Scalar(114, 114, 114)) {
    const int h = src.rows;
    const int w = src.cols;
    const float r = std::min(static_cast<float>(newH) / static_cast<float>(h),
                             static_cast<float>(newW) / static_cast<float>(w));

    const int nw = static_cast<int>(std::round(w * r));
    const int nh = static_cast<int>(std::round(h * r));

    const float dw = (newW - nw) / 2.0f;
    const float dh = (newH - nh) / 2.0f;

    cv::Mat resized;
    if (w != nw || h != nh) {
      cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    } else {
      resized = src.clone();
    }

    const int top = static_cast<int>(std::round(dh - 0.1f));
    const int bottom = static_cast<int>(std::round(dh + 0.1f));
    const int left = static_cast<int>(std::round(dw - 0.1f));
    const int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, color);
    return {out, r, dw, dh};
  }

  // 预处理细节：
  // 1) letterbox 到模型输入尺寸
  // 2) BGR -> RGB
  // 3) uint8 -> float32，并归一化到 [0,1]
  // 4) HWC -> CHW
  std::vector<float> preprocess(const cv::Mat &bgr, float &ratio, float &dw,
                                float &dh) const {
    const LetterboxResult lb = letterbox(bgr, inputH_, inputW_);
    ratio = lb.ratio;
    dw = lb.dw;
    dh = lb.dh;

    cv::Mat rgb;
    cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat f32;
    rgb.convertTo(f32, CV_32F, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(f32, channels);

    std::vector<float> inputTensorValues(1ULL * 3ULL *
                                         static_cast<size_t>(inputH_) *
                                         static_cast<size_t>(inputW_));
    const size_t channelSize =
        static_cast<size_t>(inputH_) * static_cast<size_t>(inputW_);
    for (int c = 0; c < 3; ++c) {
      std::memcpy(inputTensorValues.data() + c * channelSize, channels[c].data,
                  channelSize * sizeof(float));
    }

    return inputTensorValues;
  }

  std::vector<Ort::Value> runInference(const std::vector<float> &inputData) {
    std::array<int64_t, 4> tensorShape = {1, 3, inputH_, inputW_};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float *>(inputData.data()), inputData.size(),
        tensorShape.data(), tensorShape.size());

    std::array<const char *, 1> inputNames = {inputName_.c_str()};
    std::array<const char *, 1> outputNames = {outputName_.c_str()};

    return session_.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                        &inputTensor, 1, outputNames.data(), 1);
  }

  // 多输出推理（用于分割等需要多个输出张量的模型）
  std::vector<Ort::Value>
  runInference(const std::vector<float> &inputData,
               const std::vector<std::string> &outputNames) {
    std::array<int64_t, 4> tensorShape = {1, 3, inputH_, inputW_};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float *>(inputData.data()), inputData.size(),
        tensorShape.data(), tensorShape.size());

    std::vector<const char *> namePtrs;
    for (auto &name : outputNames)
      namePtrs.push_back(name.c_str());

    std::array<const char *, 1> inputNames = {inputName_.c_str()};

    return session_.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                        &inputTensor, 1, namePtrs.data(), namePtrs.size());
  }

  // ONNX Runtime 关键对象
  Ort::Env env_;
  Ort::SessionOptions sessionOptions_;
  Ort::Session session_;
  Ort::MemoryInfo memoryInfo_;

  // 输入输出信息
  std::string inputName_;
  std::string outputName_;
  int inputH_ = 320;
  int inputW_ = 320;
};
