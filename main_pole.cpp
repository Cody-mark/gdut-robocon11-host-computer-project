#include "transfer_protocol.hpp"
#include "serial_connector.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

int hmin=0,smin=104,vmin=97;
int hmax=11,smax=255,vmax=255;

cv::Mat frame_pole;

cv::Point getContours(cv::Mat imgDil)
{
    std::vector<cv::Point> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(imgDil,contours,hierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> conPoly;
    cv::Rect boundRect;
    cv::Point pole_center(0,0);

    int area = cv::contourArea(contours);
    if (area > 500)
    {
        float peri = cv::arcLength(contours, true);
        cv::approxPolyDP(contours, conPoly, 0.02 * peri, true);
        boundRect = cv::boundingRect(conPoly);
        cv::rectangle(frame_pole, boundRect.tl(), boundRect.br(), cv::Scalar(0, 255, 0), 2);
        pole_center.x = boundRect.x + boundRect.width / 2;
        pole_center.y = boundRect.y + boundRect.height / 2;
    }
    return pole_center;
}

float getDistance(cv::Point pole_center)
{
    cv::Point frame_center(frame_pole.cols / 2, frame_pole.rows / 2);
    float distance = cv::norm(frame_center - pole_center);
    if(pole_center.x < frame_center.x)
    {
        distance = -distance;
    }
    return distance;
}

void drawOnFrame(cv::Mat &frame, cv::Point pole_center)
{
    // 画面中心十字
    cv::drawMarker(frame_pole, pole_center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);

    // 两点连线
    cv::line(frame_pole, cv::Point(frame_pole.cols / 2,frame_pole.rows / 2), pole_center,
             cv::Scalar(0, 255, 255), 2);
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
    std::system("chcp 65001 > nul");
#endif
    try {
        // ---- 参数 ----
        const char *env_port = std::getenv("SERIAL_PORT");
#ifdef _WIN32
        std::string serialPort = env_port ? env_port : "COM9";
#else
        std::string serialPort = env_port ? env_port : "/dev/ttyUSB0";
#endif

        int cameraIndex = 1;
        bool showWindow = true;
        bool useSerial = true;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--camera" && i + 1 < argc)
                cameraIndex = std::stoi(argv[++i]);
            else if (arg == "--port" && i + 1 < argc)
                serialPort = argv[++i];
            else if (arg == "--no-serial")
                useSerial = false;
            else if (arg == "--no-window")
                showWindow = false;
            else {
                std::cerr
                    << "用法: " << argv[0]
                    << " [--camera N] [--port PATH] [--no-serial] [--no-window]\n";
                return 1;
            }
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

        constexpr uint16_t pole_code = 0x0002;

        // ---- 摄像头 ----
        std::cout << "打开摄像头 " << cameraIndex << "...\n";
        cv::VideoCapture cap(cameraIndex);
        if (!cap.isOpened()) {
            std::cerr << "Error: 无法正确打开摄像头！" << std::endl;
            return -1;
        }

        std::cout << "开始检测，code=" << pole_code << "，按 ESC 退出\n\n";

        // ---- 主循环 ----
        cv::Mat HSV_pole,pole_mask;
        int frameCount = 0;
        auto t0 = std::chrono::steady_clock::now();

        while (true) {
            cap >> frame_pole;
            if (frame_pole.empty()) {
                std::cout << "Error: 无法读取摄像头帧！" << std::endl;
                break;
            }
            frameCount++;

            cv::cvtColor(frame_pole,HSV_pole,cv::COLOR_BGR2HSV);
            cv::Scalar lower(hmin,smin,vmin);
            cv::Scalar upper(hmax,smax,vmax);

            cv::inRange(HSV_pole,lower,upper,pole_mask);

            cv::Point pole_pole_center_point = getContours(pole_mask);

            // 左负右正
            float distance = getDistance(pole_pole_center_point);

            drawOnFrame(frame_pole, pole_pole_center_point);

            // 发送 X 轴距离（右正左负）
            if (pole_pole_center_point != cv::Point(0,0)) {
                std::vector<uint8_t> body(sizeof(int16_t));
                int16_t number = static_cast<int16_t>(std::floor(distance));
                std::memcpy(body.data(), &number, sizeof(int16_t));

                auto packet = std::make_shared<packet_t>(pole_code, body.begin(), body.end(), build_packet);
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
            std::cout << " Distance from pole_center to frame_center: " << distance << std::endl;

            // 可视化
            if (showWindow) {
                cv::Mat display = frame_pole.clone();

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

                cv::imshow("frame_pole", display);
                if (cv::waitKey(1) == 27) // 按下 ESC 键退出
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

    return 0;
}
