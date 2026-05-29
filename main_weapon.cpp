#include "transfer_protocol.hpp"
#include "yolo_model_detector.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  try {
    // ---- 参数 ----
    const char *env_model = std::getenv("WEAPON_MODEL_PATH");
    std::string modelPath =
        env_model ? env_model : "model/weapon_pickup.onnx";

    const char *env_port = std::getenv("SERIAL_PORT");
    std::string serialPort = env_port ? env_port : "/dev/ttyUSB0";

    int cameraIndex = 0;
    float confThres = 0.25f;
    int weaponClassId = 0;
    int itfClassId = 1;
    bool showWindow = true;

    for (int i = 1; i < argc; ++i) {
      std::string arg(argv[i]);
      if (arg == "--model" && i + 1 < argc)
        modelPath = argv[++i];
      else if (arg == "--camera" && i + 1 < argc)
        cameraIndex = std::stoi(argv[++i]);
      else if (arg == "--conf" && i + 1 < argc)
        confThres = std::stof(argv[++i]);
      else if (arg == "--port" && i + 1 < argc)
        serialPort = argv[++i];
      else if (arg == "--no-window")
        showWindow = false;
      else {
        std::cerr
            << "用法: " << argv[0]
            << " [--model PATH] [--camera N] [--conf F] [--port PATH] [--no-window]\n";
        return 1;
      }
    }

    if (!std::filesystem::exists(modelPath)) {
      std::cerr << "模型文件不存在: " << modelPath << "\n";
      return 1;
    }

    // ---- 传输协议 ----
    using namespace gdut;
    using packet_t = data_packet<crc16_algorithm>;

    packet_manager<crc16_algorithm> manager;
    std::ofstream serialOut;

    if (!serialPort.empty()) {
      serialOut.open(serialPort, std::ios::binary);
      if (serialOut.is_open()) {
        manager.set_send_function(
            [&serialOut](const uint8_t *begin, const uint8_t *end) {
              serialOut.write(reinterpret_cast<const char *>(begin),
                              end - begin);
              serialOut.flush();
            });
        std::cout << "串口输出: " << serialPort << "\n";
      } else {
        std::cerr << "无法打开串口: " << serialPort
                  << " (仅显示模式运行)\n";
      }
    }

    constexpr uint16_t weapon_code = 0x0001;

    // ---- 模型 ----
    std::cout << "加载模型: " << modelPath << "\n";
    YoloOnnxDetector weapon(modelPath, {"weapon", "itf"});
    std::cout << "模型输入: " << weapon.inputW() << "x" << weapon.inputH()
              << "\n";

    // ---- 摄像头 ----
    std::cout << "打开摄像头 " << cameraIndex << "...\n";
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
      std::cerr << "无法打开摄像头\n";
      return -1;
    }

    std::cout << "开始检测，code=" << weapon_code << "，按 ESC 退出\n\n";

    // ---- 主循环 ----
    cv::Mat frame;
    int frameCount = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (true) {
      cap >> frame;
      if (frame.empty())
        continue;
      frameCount++;

      const cv::Point imgCenter(frame.cols / 2, frame.rows / 2);

      // YOLO 推理
      const auto allDets = weapon.process(frame, confThres);

      // weapon → ITF 级联检测
      const auto weapons =
          YoloOnnxDetector::getByClass(allDets, weaponClassId);
      const auto itfs = YoloOnnxDetector::getByClass(allDets, itfClassId);

      const YoloOnnxDetector::Detection *targetItf = nullptr;
      const YoloOnnxDetector::Detection *centerWeapon = nullptr;

      if (!weapons.empty()) {
        centerWeapon =
            YoloOnnxDetector::nearestToCenter(weapons, frame.size());
        if (centerWeapon) {
          for (const auto &itf : itfs) {
            if (centerWeapon->box.contains(itf.center())) {
              if (!targetItf || itf.score > targetItf->score)
                targetItf = &itf;
            }
          }
        }
      }

      // 发送欧式距离（右正左负）
      if (targetItf) {
        float distance = static_cast<float>(
            cv::norm(imgCenter - targetItf->center()));
        if (targetItf->center().x < imgCenter.x)
          distance = -distance;

        std::vector<uint8_t> body(sizeof(float));
        std::memcpy(body.data(), &distance, sizeof(float));

        packet_t packet(weapon_code, body.begin(), body.end(),
                        build_packet);
        if (packet) {
          manager.send(packet);
        }
      }

      // 终端输出
      if (targetItf) {
        cv::Point c = targetItf->center();
        float distance = static_cast<float>(
            cv::norm(imgCenter - c));
        std::cout << "ITF: (" << c.x << ", " << c.y << ")"
                  << "  dist=" << distance
                  << "  score=" << targetItf->score
                  << "  weapon=(" << centerWeapon->center().x << ","
                  << centerWeapon->center().y << ")"
                  << "  detections=" << allDets.size() << "\n";
      } else if (centerWeapon) {
        std::cout << "无 ITF  weapon=(" << centerWeapon->center().x
                  << "," << centerWeapon->center().y << ")"
                  << "  detections=" << allDets.size() << "\n";
      }

      // 可视化
      if (showWindow) {
        cv::Mat display = frame.clone();
        YoloOnnxDetector::drawDetections(display, allDets);

        cv::drawMarker(display, imgCenter, cv::Scalar(0, 255, 0),
                       cv::MARKER_CROSS, 20, 2);

        if (centerWeapon) {
          cv::drawMarker(display, centerWeapon->center(),
                         cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 30, 2);
        }
        if (targetItf) {
          cv::drawMarker(display, targetItf->center(),
                         cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 30, 2);
          cv::line(display, imgCenter, targetItf->center(),
                   cv::Scalar(0, 255, 255), 2);
          float dist = static_cast<float>(
              cv::norm(imgCenter - targetItf->center()));
          cv::putText(display,
                      "dist=" + std::to_string(static_cast<int>(dist)),
                      cv::Point(imgCenter.x + 15, imgCenter.y - 15),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      cv::Scalar(0, 255, 255), 2);
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration<double>(t1 - t0).count();
        if (elapsed >= 1.0) {
          double fps = frameCount / elapsed;
          cv::putText(display, cv::format("FPS: %.1f", fps),
                      cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                      cv::Scalar(0, 255, 255), 2);
          frameCount = 0;
          t0 = t1;
        }

        cv::imshow("Weapon Detection", display);
        if (cv::waitKey(1) == 27)
          break;
      }
    }

    cv::destroyAllWindows();
    std::cout << "退出\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return -1;
  }
}
