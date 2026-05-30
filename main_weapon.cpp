#include "transfer_protocol.hpp"
#include "weapon_pipeline.hpp"
#include "serial_connector.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char *argv[]) {
  #ifdef _WIN32
  std::system("chcp 65001 > nul"); // Windows 下启用 UTF-8 输出
  #endif
  try {
    // ---- 参数 ----
    const char *env_model = std::getenv("WEAPON_MODEL_PATH");
    std::string modelPath =
        env_model ? env_model : "model/weapon_pickup.onnx";

    const char *env_port = std::getenv("SERIAL_PORT");
  #ifdef _WIN32
    std::string serialPort = env_port ? env_port : "COM9";
  #else
    std::string serialPort = env_port ? env_port : "/dev/ttyUSB0";
  #endif

    int cameraIndex = 1;
    float confThres = 0.25f;
    bool showWindow = true;
    bool useSerial = true;

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
      else if (arg == "--no-serial")
        useSerial = false;
      else if (arg == "--no-window")
        showWindow = false;
      else {
        std::cerr
            << "用法: " << argv[0]
            << " [--model PATH] [--camera N] [--conf F] [--port PATH] [--no-serial] [--no-window]\n";
        return 1;
      }
    }

    if (!std::filesystem::exists(modelPath)) {
      std::cerr << "模型文件不存在: " << modelPath << "\n";
      return 1;
    }

    // ---- 串口异步通信 ----
    using namespace gdut;
    using packet_t = data_packet<crc16_algorithm>;

    asio::io_context ioContext;
    auto workGuard = asio::make_work_guard(ioContext);
    std::thread ioThread([&ioContext]() { ioContext.run(); });

    std::unique_ptr<SerialConnector> serialConnector;

    if (useSerial && !serialPort.empty()) {
      try {
        serialConnector = std::make_unique<SerialConnector>(serialPort, ioContext);
        std::cout << "串口已连接: " << serialPort << "\n";
      } catch (const std::exception &ex) {
        std::cerr << "串口初始化失败: " << ex.what() << " (仅显示模式运行)\n";
      }
    } else {
      std::cout << "串口已禁用 (--no-serial)\n";
    }

    std::function<void()> startAsyncReceive;
    startAsyncReceive = [&]() {
      if (!serialConnector)
        return;

      serialConnector->asyncReceive(
          [&](std::error_code ec, SerialConnector::packet_t packet) {
            if (ec) {
              std::cerr << "串口接收错误: " << ec.message() << "\n";
              return;
            }

            std::cout << "收到串口包: code=0x" << std::hex << packet.code()
                      << std::dec << " size=" << packet.body_size() << "\n";

            if (serialConnector) {
              startAsyncReceive();
            }
          });
    };

    if (serialConnector) {
      startAsyncReceive();
    }

    constexpr uint16_t weapon_code = 0x0001;

    // ---- 管线 ----
    std::cout << "加载模型: " << modelPath << "\n";
    WeaponPipeline pipeline(modelPath, {"weapon", "itf"}, 0, 1, confThres);
    std::cout << "模型输入: " << pipeline.getDetector().inputW() << "x"
              << pipeline.getDetector().inputH() << "\n";

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

      // Stage 1 + 2: 检测 + 匹配
      auto target = pipeline.process(frame);

      // 发送 X 轴距离（右正左负）
      if (target.found) {
        std::vector<uint8_t> body(sizeof(int16_t));
        int16_t number = static_cast<int16_t>(std::floor(target.distance));
        std::memcpy(body.data(), &number, sizeof(int16_t));

        auto packet = std::make_shared<packet_t>(weapon_code, body.begin(), body.end(), build_packet);
        if (*packet && serialConnector) {
          serialConnector->asyncSend(
              *packet,
              [packet](std::error_code ec, std::size_t bytesTransferred) {
                if (ec) {
                  std::cerr << "串口发送错误: " << ec.message() << "\n";
                  return;
                }
                (void)bytesTransferred;
              });
        }
      }

      // 终端输出
      if (target.found) {
        std::cout << "ITF: (" << target.itf_center.x << ", "
                  << target.itf_center.y << ")"
                  << "  dist=" << target.distance
                  << "  score=" << target.itf_score
                  << "  weapon=(" << target.weapon_center.x << ","
                  << target.weapon_center.y << ")\n";
      } else if (target.weapon_center != cv::Point(0, 0)) {
        std::cout << "无 ITF  weapon=(" << target.weapon_center.x
                  << "," << target.weapon_center.y << ")\n";
      }

      // 可视化
      if (showWindow) {
        cv::Mat display = frame.clone();
        pipeline.drawTarget(display, target);

        // FPS
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

    serialConnector.reset();
    workGuard.reset();
    ioContext.stop();
    if (ioThread.joinable()) {
      ioThread.join();
    }

    cv::destroyAllWindows();
    std::cout << "退出\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return -1;
  }
}
