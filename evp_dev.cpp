#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <GL/freeglut.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <tuple>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstdio>

// Global variables
cv::Mat latestFrame;
bool frameAvailable = false;
bool isPaused = false;
bool isLooping = false;
bool showUI = true;
int currentVideoIndex = 0;
std::vector<std::string> playlist;

// Dual windows: video (stream only) + control panel
int videoWindowId = 0;
int controlWindowId = 0;
int videoWinX = 100, videoWinY = 100;
int videoWinW = 800, videoWinH = 600;
int controlWinW = 960, controlWinH = 720;
int controlWinX = 920, controlWinY = 80;
bool videoGeomFromArgs = false;
bool controlGeomFromArgs = false;
bool videoFullscreen = false;
bool controlFullscreen = false;
// Saved when entering control fullscreen
int controlSavedX = 920, controlSavedY = 80, controlSavedW = 960, controlSavedH = 720;

// Legacy aliases used by control-window UI coordinate math
int windowWidth = 960, windowHeight = 720;
float uiScale = 1.0f;
int activeSlider = -1; // -1 means no slider is active
bool isSeeking = false;
double videoDuration = 0;
double currentVideoTime = 0;
double lastFrameTime = 0;
bool mouseLeftDown = false;
float fastForwardSpeed = 1.0f;
const float MAX_FF_SPEED = 16.0f;

// Video-window drag placement
enum class VideoDragMode { None, Move, ResizeW, ResizeE, ResizeN, ResizeS, ResizeNW, ResizeNE, ResizeSW, ResizeSE };
VideoDragMode videoDragMode = VideoDragMode::None;
int dragStartScreenX = 0, dragStartScreenY = 0;
int dragGeomX = 0, dragGeomY = 0, dragGeomW = 0, dragGeomH = 0;
const int VIDEO_DRAG_EDGE = 14;

//pi menu vars

bool showPieMenu = false;
float pieMenuCenterX = 0, pieMenuCenterY = 0;
int selectedFilterIndex = -1;
int pieHoverIndex = -1;
const float PIE_MENU_RADIUS = 0.40f;
const float PIE_MENU_INNER = 0.10f;

// Video content placement within the video window (1.0 = fit, offsets are normalized -1..1)
float videoScale = 1.0f;
float videoOffsetX = 0.0f;
float videoOffsetY = 0.0f;
int activeViewSlider = -1; // 0=scale, 1=posX, 2=posY
const float VIDEO_SCALE_MIN = 0.2f;
const float VIDEO_SCALE_MAX = 2.0f;

// Window geometry placement sliders (normalized against screen size)
int activePlaceSlider = -1; // 0=x 1=y 2=w 3=h
const int VIDEO_MIN_W = 160;
const int VIDEO_MIN_H = 120;

// Theater calibration guides (drawn on video window)
bool guideCrosshair = false;
bool guideEdgeBorder = false;
bool guideThirds = false;
bool guideGrid = false;
bool guideActionSafe = false;
bool guideTitleSafe = false;
bool guideAspect169 = false;
bool guideAspect43 = false;
bool guideAspect239 = false;

// Control-panel layout: guides/placement sit in a middle column so Active Effects
// can use the full left column for filter sliders.
const float GUIDE_PANEL_X = -0.45f;
const float GUIDE_PANEL_Y = 0.58f;
const float PLACE_PANEL_X = -0.45f;
const float PLACE_PANEL_Y = 0.10f;

// Source picker menus (file browser / webcam list)
enum class SourceMenuMode { None, Files, Cameras };
SourceMenuMode sourceMenuMode = SourceMenuMode::None;
std::string browserPath;
std::vector<std::pair<std::string, bool>> browserEntries; // name, isDirectory
int browserScroll = 0;
struct CameraDevice {
    int index;
    std::string label;
};
std::vector<CameraDevice> cameraDevices;
int cameraScroll = 0;
const int SOURCE_MENU_ROWS = 12;
namespace fs = std::filesystem;

// Scrollable filter toggle list (right side of control panel)
int filterListScroll = 0;
const float FILT_PANEL_X = 0.48f;
const float FILT_PANEL_W = 0.48f;
const float FILT_PANEL_TOP = 0.82f;
const float FILT_PANEL_BOTTOM = -0.68f;
const float FILT_ROW_H = 0.072f;
const float FILT_BTN_H = 0.055f;

// Colors
struct Color {
    float r, g, b;
};
Color buttonColor = {0.2f, 0.2f, 0.2f};
Color buttonHoverColor = {0.3f, 0.3f, 0.3f};
Color buttonActiveColor = {0.1f, 0.5f, 0.1f};
Color textColor = {1.0f, 1.0f, 1.0f};
Color sliderTrackColor = {0.3f, 0.3f, 0.3f};
Color sliderThumbColor = {0.5f, 0.5f, 0.5f};
Color bgColor = {0.1f, 0.1f, 0.1f};
Color progressBarColor = {0.8f, 0.2f, 0.2f};
Color progressBgColor = {0.3f, 0.3f, 0.3f};

class VideoFilter {
public:
    virtual cv::Mat apply(const cv::Mat& frame) = 0;
    virtual std::string name() const = 0;
    virtual bool hasStrength() const { return true; }
    virtual bool hasParam1() const { return false; }
    virtual bool hasParam2() const { return false; }
    virtual std::string param1Name() const { return ""; }
    virtual std::string param2Name() const { return ""; }
    
    bool enabled = false;
    float strength = 1.0f;
    float param1 = 0.5f;
    float param2 = 0.5f;
    virtual ~VideoFilter() = default;
};

// Filter implementations
class Anaglyph3DFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        int shift = static_cast<int>(10 * strength);
        if (frame.cols <= shift) return frame;

        cv::Mat left = frame(cv::Rect(0, 0, frame.cols - shift, frame.rows));
        cv::Mat right = frame(cv::Rect(shift, 0, frame.cols - shift, frame.rows));

        cv::Mat output(frame.rows, frame.cols - shift, CV_8UC3);
        for (int y = 0; y < output.rows; ++y) {
            for (int x = 0; x < output.cols; ++x) {
                cv::Vec3b lpx = left.at<cv::Vec3b>(y, x);
                cv::Vec3b rpx = right.at<cv::Vec3b>(y, x);
                output.at<cv::Vec3b>(y, x) = { 
                    static_cast<uchar>(rpx[0] * param1), 
                    static_cast<uchar>(rpx[1] * param1), 
                    static_cast<uchar>(lpx[2] * param2) 
                };
            }
        }
        cv::copyMakeBorder(output, output, 0, 0, 0, shift, cv::BORDER_CONSTANT);
        return output;
    }
    std::string name() const override { return "Anaglyph 3D"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Right Eye"; }
    std::string param2Name() const override { return "Left Eye"; }
};

// Temporal anaglyph: alternate full-frame red vs blue/cyan views each frame
// so only one eye channel is visible at a time (field-sequential style).
class AlternatingAnaglyphFilter : public VideoFilter {
    int frameTick = 0;
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;

        int shift = static_cast<int>(10 * strength);
        if (frame.cols <= shift) return frame;

        cv::Mat left = frame(cv::Rect(0, 0, frame.cols - shift, frame.rows));
        cv::Mat right = frame(cv::Rect(shift, 0, frame.cols - shift, frame.rows));

        // Even ticks: red (left). Odd ticks: blue/cyan (right).
        // param1 > 0.5 swaps which eye appears first.
        bool showRed = ((frameTick & 1) == 0);
        if (param1 > 0.5f) showRed = !showRed;
        ++frameTick;

        const float gain = std::max(0.0f, std::min(1.0f, param2));
        cv::Mat output(frame.rows, frame.cols - shift, CV_8UC3);
        for (int y = 0; y < output.rows; ++y) {
            for (int x = 0; x < output.cols; ++x) {
                cv::Vec3b lpx = left.at<cv::Vec3b>(y, x);
                cv::Vec3b rpx = right.at<cv::Vec3b>(y, x);
                if (showRed) {
                    // Left eye only — red channel (BGR: index 2)
                    output.at<cv::Vec3b>(y, x) = {
                        0,
                        0,
                        static_cast<uchar>(lpx[2] * gain)
                    };
                } else {
                    // Right eye only — blue + green (cyan) channels
                    output.at<cv::Vec3b>(y, x) = {
                        static_cast<uchar>(rpx[0] * gain),
                        static_cast<uchar>(rpx[1] * gain),
                        0
                    };
                }
            }
        }
        cv::copyMakeBorder(output, output, 0, 0, 0, shift, cv::BORDER_CONSTANT);
        return output;
    }

    std::string name() const override { return "Anaglyph Seq"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Swap Phase"; }
    std::string param2Name() const override { return "Eye Gain"; }
};

class GrayscaleFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 1.0f - strength, gray, strength, 0, gray);
        return gray;
    }
    std::string name() const override { return "Grayscale"; }
};

class EdgeDetectFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50 * strength, 150 * strength);
        cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 1.0f - strength, edges, strength, 0, edges);
        return edges;
    }
    std::string name() const override { return "Edge Detection"; }
};

class SepiaFilter : public VideoFilter {
public:
    // Assuming these are member variables
    bool enabled = true;
    float strength = 1.0f;  // Range should be [0, 1]

    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat sepia;
        cv::transform(frame, sepia, cv::Matx33f(
            0.393f + 0.607f * (1 - strength), 0.769f * strength, 0.189f * strength,
            0.349f * strength, 0.686f + 0.314f * (1 - strength), 0.168f * strength,
            0.272f * strength, 0.534f * strength, 0.131f + 0.869f * (1 - strength)
        ));
        
        cv::addWeighted(frame, 1.0f - strength, sepia, strength, 0, sepia);
        return sepia;
    }

    std::string name() const override { return "Sepia"; }
};

class PixelateFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        int blockSize = static_cast<int>(1 + 30 * strength);
        if (blockSize <= 1) return frame;
        
        cv::Mat pixelated;
        cv::resize(frame, pixelated, cv::Size(frame.cols / blockSize, frame.rows / blockSize), 0, 0, cv::INTER_NEAREST);
        cv::resize(pixelated, pixelated, frame.size(), 0, 0, cv::INTER_NEAREST);
        cv::addWeighted(frame, 1.0f - strength, pixelated, strength, 0, pixelated);
        return pixelated;
    }
    std::string name() const override { return "Pixelate"; }
};

class InvertFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        cv::Mat inverted;
        cv::bitwise_not(frame, inverted);
        cv::addWeighted(frame, 1.0f - strength, inverted, strength, 0, inverted);
        return inverted;
    }
    std::string name() const override { return "Invert"; }
};

class BlurFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        cv::Mat blurred;
        int ksize = static_cast<int>(1 + 30 * strength);
        if (ksize % 2 == 0) ksize++; // Ensure odd kernel size
        cv::GaussianBlur(frame, blurred, cv::Size(ksize, ksize), 0);
        cv::addWeighted(frame, 1.0f - strength, blurred, strength, 0, blurred);
        return blurred;
    }
    std::string name() const override { return "Blur"; }
};

class VignetteFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat mask(frame.size(), CV_32F);
        float radius = 1.0f - 0.8f * strength;
        float softness = 0.3f * param1;
        
        cv::Point center(frame.cols / 2, frame.rows / 2);
        float maxDist = sqrt(center.x * center.x + center.y * center.y) * radius;
        
        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                float dist = sqrt((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
                float intensity = 1.0f - smoothstep(maxDist * (1.0f - softness), maxDist, dist);
                mask.at<float>(y, x) = intensity;
            }
        }
        
        cv::Mat result;
        cv::Mat mask3ch;
        cv::cvtColor(mask, mask3ch, cv::COLOR_GRAY2BGR);
        frame.convertTo(result, CV_32F);
        result = result.mul(mask3ch);
        result.convertTo(result, frame.type());
        
        cv::addWeighted(frame, 1.0f - strength, result, strength, 0, result);
        return result;
    }
    
    float smoothstep(float edge0, float edge1, float x) {
        x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return x * x * (3 - 2 * x);
    }
    
    std::string name() const override { return "Vignette"; }
    bool hasParam1() const override { return true; }
    std::string param1Name() const override { return "Softness"; }
};

class NoiseFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat noise(frame.size(), frame.type());
        cv::randn(noise, 0, 255 * strength * param1);
        
        cv::Mat result;
        cv::addWeighted(frame, 1.0f - strength, noise, strength, 0, result);
        
        if (param2 > 0.1f) { // Color noise
            cv::Mat colorNoise(frame.size(), frame.type());
            cv::randn(colorNoise, 0, 255 * strength * param2);
            cv::addWeighted(result, 0.7f, colorNoise, 0.3f, 0, result);
        }
        
        return result;
    }
    
    std::string name() const override { return "Noise"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Intensity"; }
    std::string param2Name() const override { return "Color Noise"; }
};

class ColorBalanceFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        
        // Red channel
        channels[2] = channels[2] * (1.0f + (param1 - 0.5f) * 2.0f * strength);
        // Green channel
        channels[1] = channels[1] * (1.0f + (param2 - 0.5f) * 2.0f * strength);
        // Blue channel
        channels[0] = channels[0] * (1.0f + (0.5f - (param1 + param2)/2) * 2.0f * strength);
        
        cv::Mat result;
        cv::merge(channels, result);
        cv::addWeighted(frame, 1.0f - strength, result, strength, 0, result);
        return result;
    }
    
    std::string name() const override { return "Color Balance"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Warmth"; }
    std::string param2Name() const override { return "Tint"; }
};

class ScanlinesFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat scanlines(frame.size(), frame.type(), cv::Scalar::all(0));
        int lineHeight = static_cast<int>(1 + 4 * (1.0f - param1));
        float intensity = 0.7f * strength;
        
        for (int y = 0; y < frame.rows; y += lineHeight + 1) {
            cv::Rect roi(0, y, frame.cols, 1);
            frame(roi).copyTo(scanlines(roi));
        }
        
        // Add some glow between lines
        if (param2 > 0.1f) {
            cv::Mat glow;
            cv::GaussianBlur(scanlines, glow, cv::Size(0, 0), 3.0f * param2);
            cv::addWeighted(scanlines, 0.8f, glow, 0.2f, 0, scanlines);
        }
        
        cv::addWeighted(frame, 1.0f - strength, scanlines, strength, 0, scanlines);
        return scanlines;
    }
    
    std::string name() const override { return "Scanlines"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Density"; }
    std::string param2Name() const override { return "Glow"; }
};

class CRTFilter : public VideoFilter {
private:
    cv::Mat lastFrame;
    bool firstFrame = true;
    
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat result;
        
        // Apply curvature
        if (strength > 0.1f) {
            result = applyBarrelDistortion(frame, strength * 0.3f);
        } else {
            frame.copyTo(result);
        }
        
        // Apply scanlines
        if (param1 > 0.1f) {
            cv::Mat scanlines(result.size(), result.type(), cv::Scalar::all(0));
            int lineHeight = static_cast<int>(1 + 2 * (1.0f - param1));
            
            for (int y = 0; y < result.rows; y += lineHeight + 1) {
                cv::Rect roi(0, y, result.cols, 1);
                result(roi).copyTo(scanlines(roi));
            }
            
            cv::addWeighted(result, 1.0f - param1 * 0.7f, scanlines, param1 * 0.7f, 0, result);
        }
        
        // Apply phosphor persistence (temporal effect)
        if (param2 > 0.1f && !firstFrame) {
            cv::addWeighted(result, 1.0f - param2 * 0.5f, lastFrame, param2 * 0.5f, 0, result);
        }
        
        // Add slight blur to simulate CRT softness
        cv::GaussianBlur(result, result, cv::Size(3, 3), 0.5f);
        
        // Store for next frame
        result.copyTo(lastFrame);
        firstFrame = false;
        
        return result;
    }
    
    cv::Mat applyBarrelDistortion(const cv::Mat& frame, float strength) {
        cv::Mat map_x, map_y;
        map_x.create(frame.size(), CV_32FC1);
        map_y.create(frame.size(), CV_32FC1);
        
        int w = frame.cols;
        int h = frame.rows;
        float half_w = w * 0.5f;
        float half_h = h * 0.5f;
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float nx = (x - half_w) / half_w;
                float ny = (y - half_h) / half_h;
                
                float r = sqrt(nx * nx + ny * ny);
                float theta = 1.0f - strength * r * r;
                
                float src_x = half_w + theta * nx * half_w;
                float src_y = half_h + theta * ny * half_h;
                
                map_x.at<float>(y, x) = src_x;
                map_y.at<float>(y, x) = src_y;
            }
        }
        
        cv::Mat result;
        cv::remap(frame, result, map_x, map_y, cv::INTER_LINEAR);
        return result;
    }
    
    std::string name() const override { return "CRT Effect"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    bool isTemporal() const { return true; }
    std::string param1Name() const override { return "Scanlines"; }
    std::string param2Name() const override { return "Persistence"; }
};

class GlitchFilter : public VideoFilter {
private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist{0, 100};
    
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat result;
        frame.copyTo(result);
        
        int glitchAmount = static_cast<int>(strength * 20);
        if (glitchAmount < 1) return frame;
        
        // Random line shifts
        for (int i = 0; i < glitchAmount; i++) {
            int y = dist(generator) % frame.rows;
            int shift = (dist(generator) % 20) - 10;
            shift *= strength;
            
            if (shift == 0) continue;
            
            cv::Mat line = frame.row(y).clone();
            if (shift > 0) {
                cv::Mat shiftedPart = line.colRange(0, line.cols - shift);
                shiftedPart.copyTo(result.row(y).colRange(shift, line.cols));
            } else {
                cv::Mat shiftedPart = line.colRange(-shift, line.cols);
                shiftedPart.copyTo(result.row(y).colRange(0, line.cols + shift));
            }
        }
        
        // Color channel offset
        if (param1 > 0.1f) {
            int offset = static_cast<int>(5 * param1);
            std::vector<cv::Mat> channels;
            cv::split(result, channels);
            
            cv::Mat shiftedR = channels[2].clone();
            cv::Mat shiftedB = channels[0].clone();
            
            cv::Rect roiR(offset, 0, channels[2].cols - offset, channels[2].rows);
            cv::Rect roiB(0, 0, channels[0].cols - offset, channels[0].rows);
            
            channels[2](roiR).copyTo(shiftedR(cv::Rect(0, 0, channels[2].cols - offset, channels[2].rows)));
            channels[0](roiB).copyTo(shiftedB(cv::Rect(offset, 0, channels[0].cols - offset, channels[0].rows)));
            
            channels[2] = shiftedR;
            channels[0] = shiftedB;
            cv::merge(channels, result);
        }
        
        // Noise
        if (param2 > 0.1f) {
            cv::Mat noise(result.size(), result.type());
            cv::randn(noise, 0, 25 * param2);
            cv::add(result, noise, result);
        }
        
        cv::addWeighted(frame, 1.0f - strength, result, strength, 0, result);
        return result;
    }
    
    std::string name() const override { return "Glitch"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Ch. Offset"; }
    std::string param2Name() const override { return "Noise"; }
};

class KaleidoscopeFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        int segments = static_cast<int>(3 + 9 * strength);
        if (segments < 3) return frame;
        
        cv::Point center(frame.cols / 2, frame.rows / 2);
        int radius = std::min(frame.cols, frame.rows) / 2;
        
        cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(0));
        cv::circle(mask, center, radius, cv::Scalar(255), -1);
        
        cv::Mat polar;
        cv::warpPolar(frame, polar, cv::Size(radius, radius * 2), center, radius, 
                     cv::WARP_POLAR_LINEAR + cv::INTER_LINEAR);
        
        cv::Mat segment = polar(cv::Rect(0, 0, radius, polar.rows / segments));
        cv::Mat repeated;
        cv::repeat(segment, 1, segments, repeated);
        
        cv::Mat result;
        cv::warpPolar(repeated, result, frame.size(), center, radius, 
                      cv::WARP_POLAR_LINEAR + cv::WARP_INVERSE_MAP + cv::INTER_LINEAR);
        
        cv::Mat final;
        result.copyTo(final, mask);
        cv::addWeighted(frame, 1.0f - strength, final, strength, 0, final);
        return final;
    }
    
    std::string name() const override { return "Kaleidoscope"; }
};

class NightVisionFilter : public VideoFilter {
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat green;
        cv::cvtColor(frame, green, cv::COLOR_BGR2GRAY);
        cv::cvtColor(green, green, cv::COLOR_GRAY2BGR);
        
        // Boost green channel
        std::vector<cv::Mat> channels;
        cv::split(green, channels);
        channels[1] = channels[1] * (1.0f + strength);
        cv::merge(channels, green);
        
        // Add noise
        if (param1 > 0.1f) {
            cv::Mat noise(green.size(), green.type());
            cv::randn(noise, 0, 25 * param1);
            cv::add(green, noise, green);
        }
        
        // Add vignette
        if (param2 > 0.1f) {
            cv::Mat mask(green.size(), CV_32F);
            cv::Point center(green.cols / 2, green.rows / 2);
            float maxDist = sqrt(center.x * center.x + center.y * center.y) * 0.7f;
            
            for (int y = 0; y < green.rows; y++) {
                for (int x = 0; x < green.cols; x++) {
                    float dist = sqrt((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
                    float intensity = 1.0f - (dist / maxDist) * param2;
                    mask.at<float>(y, x) = intensity;
                }
            }
            
            cv::Mat result;
            cv::Mat mask3ch;
            cv::cvtColor(mask, mask3ch, cv::COLOR_GRAY2BGR);
            green.convertTo(result, CV_32F);
            result = result.mul(mask3ch);
            result.convertTo(result, green.type());
            green = result;
        }
        
        cv::addWeighted(frame, 1.0f - strength, green, strength, 0, green);
        return green;
    }
    
    std::string name() const override { return "Night Vision"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Noise"; }
    std::string param2Name() const override { return "Vignette"; }
};

class ColorizeFilter : public VideoFilter {
private:
    cv::Vec3b targetColor = cv::Vec3b(255, 0, 0); // Default red
public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;
        
        cv::Mat result;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
        
        // Calculate target color based on params
        cv::Vec3b color(
            static_cast<uchar>(param1 * 255),
            static_cast<uchar>(param2 * 255),
            static_cast<uchar>((1.0f - (param1 + param2)/2) * 255)
        );
        
        // Colorize based on grayscale
        for (int y = 0; y < gray.rows; y++) {
            for (int x = 0; x < gray.cols; x++) {
                cv::Vec3b pixel = gray.at<cv::Vec3b>(y, x);
                float intensity = pixel[0] / 255.0f;
                gray.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    color[0] * intensity,
                    color[1] * intensity,
                    color[2] * intensity
                );
            }
        }
        
        cv::addWeighted(frame, 1.0f - strength, gray, strength, 0, result);
        return result;
    }
    
    std::string name() const override { return "Colorize"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Hue"; }
    std::string param2Name() const override { return "Saturation"; }
};

std::vector<std::unique_ptr<VideoFilter>> filters;
cv::VideoCapture cap;

// Open a playlist entry: numeric strings are camera indices (not GStreamer pipelines).
bool openVideoSource(const std::string& source) {
    bool isCamera = !source.empty() &&
        std::all_of(source.begin(), source.end(),
                    [](unsigned char c) { return std::isdigit(c); });
    if (isCamera) {
        int index = std::stoi(source);
        // Prefer V4L2 on Linux so "0" is not parsed as a GStreamer bin.
        if (cap.open(index, cv::CAP_V4L2)) return true;
        return cap.open(index, cv::CAP_ANY);
    }
    return cap.open(source);
}

bool isVideoExtension(const std::string& ext) {
    static const std::unordered_set<std::string> kVideoExt = {
        ".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v",
        ".wmv", ".flv", ".mpg", ".mpeg", ".m2ts", ".ts", ".3gp"
    };
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return kVideoExt.count(e) > 0;
}

std::string truncateLabel(const std::string& s, size_t maxLen) {
    if (s.size() <= maxLen) return s;
    if (maxLen <= 3) return s.substr(0, maxLen);
    return s.substr(0, maxLen - 3) + "...";
}

void refreshBrowserEntries() {
    browserEntries.clear();
    std::error_code ec;
    if (browserPath.empty()) {
        const char* home = std::getenv("HOME");
        browserPath = home ? home : ".";
    }
    fs::path path(browserPath);
    if (!fs::exists(path, ec) || !fs::is_directory(path, ec)) {
        const char* home = std::getenv("HOME");
        browserPath = home ? home : ".";
        path = browserPath;
    }
    browserPath = fs::weakly_canonical(path, ec).string();
    if (ec) browserPath = path.string();

    std::vector<std::pair<std::string, bool>> dirs;
    std::vector<std::pair<std::string, bool>> files;
    for (fs::directory_iterator it(path, fs::directory_options::skip_permission_denied, ec), end;
         it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        const auto& entry = *it;
        std::string name = entry.path().filename().string();
        if (name.empty() || name[0] == '.') continue;
        std::error_code sec;
        if (entry.is_directory(sec)) {
            dirs.push_back({name, true});
        } else if (entry.is_regular_file(sec) && isVideoExtension(entry.path().extension().string())) {
            files.push_back({name, false});
        }
    }
    std::sort(dirs.begin(), dirs.end());
    std::sort(files.begin(), files.end());
    browserEntries = std::move(dirs);
    browserEntries.insert(browserEntries.end(), files.begin(), files.end());
    browserScroll = std::max(0, std::min(browserScroll,
        std::max(0, static_cast<int>(browserEntries.size()) - SOURCE_MENU_ROWS)));
}

void scanCameras() {
    cameraDevices.clear();

    // Prefer sysfs enumeration so we don't interrupt the active capture device.
    bool foundSysfs = false;
    for (int i = 0; i < 10; ++i) {
        std::string sysDir = "/sys/class/video4linux/video" + std::to_string(i);
        if (!fs::exists(sysDir)) continue;
        foundSysfs = true;

        std::string label = "Camera " + std::to_string(i);
        std::ifstream nameFile(sysDir + "/name");
        if (nameFile) {
            std::string deviceName;
            std::getline(nameFile, deviceName);
            if (!deviceName.empty())
                label += ": " + deviceName;
        }
        cameraDevices.push_back({i, label});
    }

    // Fallback: probe indices if sysfs is unavailable
    if (!foundSysfs) {
        std::string currentSource;
        bool restore = cap.isOpened() && !playlist.empty();
        if (restore) {
            currentSource = playlist[currentVideoIndex];
            cap.release();
        }
        for (int i = 0; i < 10; ++i) {
            cv::VideoCapture test;
            bool ok = test.open(i, cv::CAP_V4L2) || test.open(i, cv::CAP_ANY);
            if (!ok) continue;
            test.release();
            cameraDevices.push_back({i, "Camera " + std::to_string(i)});
        }
        if (restore)
            openVideoSource(currentSource);
    }

    cameraScroll = std::max(0, std::min(cameraScroll,
        std::max(0, static_cast<int>(cameraDevices.size()) - SOURCE_MENU_ROWS)));
}

bool switchToSource(const std::string& source) {
    cap.release();
    auto it = std::find(playlist.begin(), playlist.end(), source);
    if (it != playlist.end()) {
        currentVideoIndex = static_cast<int>(it - playlist.begin());
    } else {
        playlist.push_back(source);
        currentVideoIndex = static_cast<int>(playlist.size()) - 1;
    }

    if (!openVideoSource(source)) {
        std::cerr << "Failed to open source: " << source << std::endl;
        return false;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (fps > 1e-3 && frames > 0)
        videoDuration = frames / fps;
    else
        videoDuration = 0.0;
    currentVideoTime = 0.0;
    lastFrameTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    latestFrame.release();
    isPaused = false;
    frameAvailable = true;
    return true;
}

void openFileBrowserMenu() {
    showPieMenu = false;
    pieHoverIndex = -1;
    sourceMenuMode = SourceMenuMode::Files;
    if (browserPath.empty()) {
        const char* home = std::getenv("HOME");
        browserPath = home ? home : ".";
    }
    browserScroll = 0;
    refreshBrowserEntries();
}

void openCameraMenu() {
    showPieMenu = false;
    pieHoverIndex = -1;
    sourceMenuMode = SourceMenuMode::Cameras;
    cameraScroll = 0;
    scanCameras();
}

void closeSourceMenu() {
    sourceMenuMode = SourceMenuMode::None;
}

void drawText(float x, float y, const std::string& text, void* font = GLUT_BITMAP_HELVETICA_12);
void drawButton(float x, float y, float w, float h, const std::string& label, bool hover = false, bool active = false);
bool isInside(float mx, float my, float x, float y, float w, float h);

void drawSourceMenu() {
    if (sourceMenuMode == SourceMenuMode::None) return;

    const float x = -0.55f;
    const float y = -0.55f;
    const float w = 1.10f;
    const float h = 1.15f;
    const float rowH = 0.07f;
    const float listTop = y + h - 0.18f;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.0f, 0.0f, 0.0f, 0.45f);
    glBegin(GL_QUADS);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f( 1.0f, -1.0f);
    glVertex2f( 1.0f,  1.0f);
    glVertex2f(-1.0f,  1.0f);
    glEnd();

    glColor4f(0.10f, 0.10f, 0.14f, 0.96f);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();
    glColor4f(0.40f, 0.40f, 0.50f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();
    glDisable(GL_BLEND);

    if (sourceMenuMode == SourceMenuMode::Files) {
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(x + 0.03f, y + h - 0.05f, "Open Video File", GLUT_BITMAP_HELVETICA_18);
        drawText(x + 0.03f, y + h - 0.11f,
                 truncateLabel(browserPath, 70), GLUT_BITMAP_HELVETICA_12);

        drawButton(x + 0.03f, y + 0.03f, 0.16f, 0.06f, "Up");
        drawButton(x + 0.22f, y + 0.03f, 0.16f, 0.06f, "Home");
        drawButton(x + 0.41f, y + 0.03f, 0.16f, 0.06f, "Refresh");
        drawButton(x + w - 0.35f, y + 0.03f, 0.14f, 0.06f, "Prev");
        drawButton(x + w - 0.19f, y + 0.03f, 0.14f, 0.06f, "Next");
        drawButton(x + w - 0.19f, y + h - 0.08f, 0.14f, 0.06f, "Close");

        int total = static_cast<int>(browserEntries.size());
        if (total == 0) {
            glColor3f(0.7f, 0.7f, 0.75f);
            drawText(x + 0.05f, listTop, "No folders or video files here.");
        }
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = browserScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            bool isDir = browserEntries[idx].second;
            glColor3f(isDir ? 0.18f : 0.16f, isDir ? 0.22f : 0.16f, isDir ? 0.30f : 0.20f);
            glBegin(GL_QUADS);
            glVertex2f(x + 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY + 0.045f);
            glVertex2f(x + 0.03f, rowY + 0.045f);
            glEnd();
            glColor3f(isDir ? 0.75f : 0.95f, isDir ? 0.85f : 0.95f, 1.0f);
            std::string prefix = isDir ? "[DIR] " : "      ";
            drawText(x + 0.05f, rowY, truncateLabel(prefix + browserEntries[idx].first, 60));
        }
    } else {
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(x + 0.03f, y + h - 0.05f, "Select Webcam", GLUT_BITMAP_HELVETICA_18);
        drawText(x + 0.03f, y + h - 0.11f,
                 std::to_string(cameraDevices.size()) + " device(s) found",
                 GLUT_BITMAP_HELVETICA_12);

        drawButton(x + 0.03f, y + 0.03f, 0.20f, 0.06f, "Refresh");
        drawButton(x + w - 0.35f, y + 0.03f, 0.14f, 0.06f, "Prev");
        drawButton(x + w - 0.19f, y + 0.03f, 0.14f, 0.06f, "Next");
        drawButton(x + w - 0.19f, y + h - 0.08f, 0.14f, 0.06f, "Close");

        int total = static_cast<int>(cameraDevices.size());
        if (total == 0) {
            glColor3f(0.7f, 0.7f, 0.75f);
            drawText(x + 0.05f, listTop, "No webcams detected.");
        }
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = cameraScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            glColor3f(0.16f, 0.22f, 0.18f);
            glBegin(GL_QUADS);
            glVertex2f(x + 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY + 0.045f);
            glVertex2f(x + 0.03f, rowY + 0.045f);
            glEnd();
            glColor3f(0.9f, 1.0f, 0.9f);
            drawText(x + 0.05f, rowY, truncateLabel(cameraDevices[idx].label, 60));
        }
    }
}

bool hitTestSourceMenu(float fx, float fy) {
    if (sourceMenuMode == SourceMenuMode::None) return false;

    const float x = -0.55f;
    const float y = -0.55f;
    const float w = 1.10f;
    const float h = 1.15f;
    const float rowH = 0.07f;
    const float listTop = y + h - 0.18f;

    // Absorb clicks on the dimmed backdrop; only Close dismisses from chrome
    if (!(fx >= x && fx <= x + w && fy >= y && fy <= y + h)) {
        closeSourceMenu();
        return true;
    }

    if (isInside(fx, fy, x + w - 0.19f, y + h - 0.08f, 0.14f, 0.06f)) {
        closeSourceMenu();
        return true;
    }

    if (sourceMenuMode == SourceMenuMode::Files) {
        if (isInside(fx, fy, x + 0.03f, y + 0.03f, 0.16f, 0.06f)) {
            fs::path parent = fs::path(browserPath).parent_path();
            if (!parent.empty()) {
                browserPath = parent.string();
                browserScroll = 0;
                refreshBrowserEntries();
            }
            return true;
        }
        if (isInside(fx, fy, x + 0.22f, y + 0.03f, 0.16f, 0.06f)) {
            const char* home = std::getenv("HOME");
            browserPath = home ? home : ".";
            browserScroll = 0;
            refreshBrowserEntries();
            return true;
        }
        if (isInside(fx, fy, x + 0.41f, y + 0.03f, 0.16f, 0.06f)) {
            refreshBrowserEntries();
            return true;
        }
        if (isInside(fx, fy, x + w - 0.35f, y + 0.03f, 0.14f, 0.06f)) {
            browserScroll = std::max(0, browserScroll - SOURCE_MENU_ROWS);
            return true;
        }
        if (isInside(fx, fy, x + w - 0.19f, y + 0.03f, 0.14f, 0.06f)) {
            int maxScroll = std::max(0, static_cast<int>(browserEntries.size()) - SOURCE_MENU_ROWS);
            browserScroll = std::min(maxScroll, browserScroll + SOURCE_MENU_ROWS);
            return true;
        }

        int total = static_cast<int>(browserEntries.size());
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = browserScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            if (isInside(fx, fy, x + 0.03f, rowY - 0.015f, w - 0.06f, 0.06f)) {
                const auto& entry = browserEntries[idx];
                if (entry.second) {
                    browserPath = (fs::path(browserPath) / entry.first).string();
                    browserScroll = 0;
                    refreshBrowserEntries();
                } else {
                    std::string full = (fs::path(browserPath) / entry.first).string();
                    if (switchToSource(full))
                        closeSourceMenu();
                }
                return true;
            }
        }
    } else {
        if (isInside(fx, fy, x + 0.03f, y + 0.03f, 0.20f, 0.06f)) {
            scanCameras();
            return true;
        }
        if (isInside(fx, fy, x + w - 0.35f, y + 0.03f, 0.14f, 0.06f)) {
            cameraScroll = std::max(0, cameraScroll - SOURCE_MENU_ROWS);
            return true;
        }
        if (isInside(fx, fy, x + w - 0.19f, y + 0.03f, 0.14f, 0.06f)) {
            int maxScroll = std::max(0, static_cast<int>(cameraDevices.size()) - SOURCE_MENU_ROWS);
            cameraScroll = std::min(maxScroll, cameraScroll + SOURCE_MENU_ROWS);
            return true;
        }

        int total = static_cast<int>(cameraDevices.size());
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = cameraScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            if (isInside(fx, fy, x + 0.03f, rowY - 0.015f, w - 0.06f, 0.06f)) {
                if (switchToSource(std::to_string(cameraDevices[idx].index)))
                    closeSourceMenu();
                return true;
            }
        }
    }
    return true; // consume clicks inside panel
}

// Utility functions
void drawText(float x, float y, const std::string& text, void* font) {
    glRasterPos2f(x, y);
    for (char c : text)
        glutBitmapCharacter(font, c);
}

void drawButton(float x, float y, float w, float h, const std::string& label, bool hover, bool active) {
    if (active) {
        glColor3f(buttonActiveColor.r, buttonActiveColor.g, buttonActiveColor.b);
    } else if (hover) {
        glColor3f(buttonHoverColor.r, buttonHoverColor.g, buttonHoverColor.b);
    } else {
        glColor3f(buttonColor.r, buttonColor.g, buttonColor.b);
    }
    
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();

    glColor3f(textColor.r, textColor.g, textColor.b);
    float textX = x + w/2 - (label.length() * 0.01f * uiScale);
    float textY = y + h/2 - 0.015f * uiScale;
    drawText(textX, textY, label);
}

void drawSlider(float x, float y, float w, float h, float value, const std::string& label);

int filterListVisibleRows() {
    const float usable = (FILT_PANEL_TOP - 0.10f) - (FILT_PANEL_BOTTOM + 0.08f);
    return std::max(1, static_cast<int>(usable / FILT_ROW_H));
}

void clampFilterListScroll() {
    int visible = filterListVisibleRows();
    int maxScroll = std::max(0, static_cast<int>(filters.size()) - visible);
    filterListScroll = std::max(0, std::min(filterListScroll, maxScroll));
}

void drawFilterListPanel() {
    clampFilterListScroll();
    const int visible = filterListVisibleRows();
    const int total = static_cast<int>(filters.size());

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.85f);
    glBegin(GL_QUADS);
    glVertex2f(FILT_PANEL_X, FILT_PANEL_BOTTOM);
    glVertex2f(FILT_PANEL_X + FILT_PANEL_W, FILT_PANEL_BOTTOM);
    glVertex2f(FILT_PANEL_X + FILT_PANEL_W, FILT_PANEL_TOP);
    glVertex2f(FILT_PANEL_X, FILT_PANEL_TOP);
    glEnd();
    glColor4f(0.35f, 0.35f, 0.45f, 0.95f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(FILT_PANEL_X, FILT_PANEL_BOTTOM);
    glVertex2f(FILT_PANEL_X + FILT_PANEL_W, FILT_PANEL_BOTTOM);
    glVertex2f(FILT_PANEL_X + FILT_PANEL_W, FILT_PANEL_TOP);
    glVertex2f(FILT_PANEL_X, FILT_PANEL_TOP);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1.0f, 1.0f, 1.0f);
    std::ostringstream title;
    title << "Filters (" << total << ")";
    drawText(FILT_PANEL_X + 0.02f, FILT_PANEL_TOP - 0.04f, title.str(), GLUT_BITMAP_HELVETICA_12);

    drawButton(FILT_PANEL_X + 0.02f, FILT_PANEL_BOTTOM + 0.015f, 0.13f, 0.05f, "Up");
    drawButton(FILT_PANEL_X + 0.17f, FILT_PANEL_BOTTOM + 0.015f, 0.13f, 0.05f, "Down");
    {
        std::ostringstream page;
        int end = std::min(total, filterListScroll + visible);
        page << (total == 0 ? 0 : filterListScroll + 1) << "-" << end;
        glColor3f(0.75f, 0.75f, 0.85f);
        drawText(FILT_PANEL_X + 0.32f, FILT_PANEL_BOTTOM + 0.03f, page.str());
    }

    float y = FILT_PANEL_TOP - 0.10f;
    for (int row = 0; row < visible; ++row) {
        int idx = filterListScroll + row;
        if (idx >= total) break;
        std::string label = std::to_string(idx + 1) + ". " + filters[idx]->name();
        if (filters[idx]->enabled) label += " ON";
        drawButton(FILT_PANEL_X + 0.02f, y - FILT_BTN_H, FILT_PANEL_W - 0.04f, FILT_BTN_H,
                   label, false, filters[idx]->enabled);
        y -= FILT_ROW_H;
    }
}

bool hitTestFilterListPanel(float fx, float fy) {
    if (fx < FILT_PANEL_X || fx > FILT_PANEL_X + FILT_PANEL_W ||
        fy < FILT_PANEL_BOTTOM || fy > FILT_PANEL_TOP) {
        return false;
    }

    clampFilterListScroll();
    const int visible = filterListVisibleRows();
    const int total = static_cast<int>(filters.size());

    if (isInside(fx, fy, FILT_PANEL_X + 0.02f, FILT_PANEL_BOTTOM + 0.015f, 0.13f, 0.05f)) {
        filterListScroll = std::max(0, filterListScroll - visible);
        return true;
    }
    if (isInside(fx, fy, FILT_PANEL_X + 0.17f, FILT_PANEL_BOTTOM + 0.015f, 0.13f, 0.05f)) {
        int maxScroll = std::max(0, total - visible);
        filterListScroll = std::min(maxScroll, filterListScroll + visible);
        return true;
    }

    float y = FILT_PANEL_TOP - 0.10f;
    for (int row = 0; row < visible; ++row) {
        int idx = filterListScroll + row;
        if (idx >= total) break;
        if (isInside(fx, fy, FILT_PANEL_X + 0.02f, y - FILT_BTN_H, FILT_PANEL_W - 0.04f, FILT_BTN_H)) {
            filters[idx]->enabled = !filters[idx]->enabled;
            if (filters[idx]->enabled)
                activeSlider = idx * 3;
            return true;
        }
        y -= FILT_ROW_H;
    }
    return true; // absorb clicks inside panel chrome
}

// Active-effects panel on the left — tall enough for multiple filter sliders.
const float AE_PANEL_X = -0.95f;
const float AE_PANEL_W = 0.48f;
const float AE_CONTENT_X = -0.93f;
const float AE_SLIDER_W = 0.42f;
const float AE_SLIDER_H = 0.025f;
const float AE_TOP_Y = 0.82f;
const float AE_TITLE_STEP = 0.06f;
const float AE_NAME_STEP = 0.05f;
const float AE_SLIDER_STEP = 0.07f;
const float AE_FILTER_GAP = 0.035f;
const float AE_BOTTOM_LIMIT = -0.58f;

bool anyFilterEnabled() {
    for (const auto& filter : filters) {
        if (filter->enabled) return true;
    }
    return false;
}

float activeEffectsContentHeight() {
    float h = AE_TITLE_STEP;
    for (const auto& filter : filters) {
        if (!filter->enabled) continue;
        h += AE_NAME_STEP;
        if (filter->hasStrength()) h += AE_SLIDER_STEP;
        if (filter->hasParam1()) h += AE_SLIDER_STEP;
        if (filter->hasParam2()) h += AE_SLIDER_STEP;
        h += AE_FILTER_GAP;
    }
    return h;
}

void drawActiveEffectsPanel() {
    if (!anyFilterEnabled()) return;

    const float contentH = activeEffectsContentHeight();
    const float listY = std::max(AE_BOTTOM_LIMIT, AE_TOP_Y - contentH);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.82f);
    glBegin(GL_QUADS);
    glVertex2f(AE_PANEL_X, listY);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, listY);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, AE_TOP_Y);
    glVertex2f(AE_PANEL_X, AE_TOP_Y);
    glEnd();

    glColor4f(0.35f, 0.35f, 0.45f, 0.95f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(AE_PANEL_X, listY);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, listY);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, AE_TOP_Y);
    glVertex2f(AE_PANEL_X, AE_TOP_Y);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1.0f, 1.0f, 1.0f);
    drawText(AE_CONTENT_X, AE_TOP_Y - 0.04f, "Active Effects", GLUT_BITMAP_HELVETICA_18);

    float currentY = AE_TOP_Y - AE_TITLE_STEP;
    for (size_t i = 0; i < filters.size(); ++i) {
        if (!filters[i]->enabled) continue;
        if (currentY < listY + 0.04f) break;

        glColor3f(0.85f, 0.85f, 1.0f);
        drawText(AE_CONTENT_X, currentY, filters[i]->name(), GLUT_BITMAP_HELVETICA_12);
        currentY -= AE_NAME_STEP;

        if (filters[i]->hasStrength()) {
            drawSlider(AE_CONTENT_X, currentY - AE_SLIDER_H, AE_SLIDER_W, AE_SLIDER_H,
                       filters[i]->strength, "Strength");
            currentY -= AE_SLIDER_STEP;
        }
        if (filters[i]->hasParam1()) {
            drawSlider(AE_CONTENT_X, currentY - AE_SLIDER_H, AE_SLIDER_W, AE_SLIDER_H,
                       filters[i]->param1, filters[i]->param1Name());
            currentY -= AE_SLIDER_STEP;
        }
        if (filters[i]->hasParam2()) {
            drawSlider(AE_CONTENT_X, currentY - AE_SLIDER_H, AE_SLIDER_W, AE_SLIDER_H,
                       filters[i]->param2, filters[i]->param2Name());
            currentY -= AE_SLIDER_STEP;
        }
        currentY -= AE_FILTER_GAP;
    }
}

// Returns true if a slider was hit; sets activeSlider and updates the value.
bool hitTestActiveEffectsSliders(float fx, float fy) {
    if (!anyFilterEnabled()) return false;

    const float contentH = activeEffectsContentHeight();
    const float listY = std::max(AE_BOTTOM_LIMIT, AE_TOP_Y - contentH);
    if (fx < AE_PANEL_X || fx > AE_PANEL_X + AE_PANEL_W ||
        fy < listY || fy > AE_TOP_Y) {
        return false;
    }

    float currentY = AE_TOP_Y - AE_TITLE_STEP;
    for (size_t i = 0; i < filters.size(); ++i) {
        if (!filters[i]->enabled) continue;

        currentY -= AE_NAME_STEP;

        auto trySlider = [&](int paramType, float& value) -> bool {
            float sy = currentY - AE_SLIDER_H;
            if (isInside(fx, fy, AE_CONTENT_X, sy, AE_SLIDER_W, AE_SLIDER_H)) {
                activeSlider = static_cast<int>(i) * 3 + paramType;
                value = std::max(0.0f, std::min(1.0f, (fx - AE_CONTENT_X) / AE_SLIDER_W));
                return true;
            }
            currentY -= AE_SLIDER_STEP;
            return false;
        };

        if (filters[i]->hasStrength() && trySlider(0, filters[i]->strength)) return true;
        if (filters[i]->hasParam1() && trySlider(1, filters[i]->param1)) return true;
        if (filters[i]->hasParam2() && trySlider(2, filters[i]->param2)) return true;

        currentY -= AE_FILTER_GAP;
    }
    return false;
}

void drawSlider(float x, float y, float w, float h, float value, const std::string& label) {
    // Label (moved above the slider)
    glColor3f(textColor.r, textColor.g, textColor.b);
    drawText(x, y + h + 0.02f * uiScale, label + ": " + std::to_string(value).substr(0, 4));

    // Track
    glColor3f(sliderTrackColor.r, sliderTrackColor.g, sliderTrackColor.b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();

    // Thumb
    float thumbPos = x + (w - h) * value;
    glColor3f(sliderThumbColor.r, sliderThumbColor.g, sliderThumbColor.b);
    glBegin(GL_QUADS);
    glVertex2f(thumbPos, y);
    glVertex2f(thumbPos + h, y);
    glVertex2f(thumbPos + h, y + h);
    glVertex2f(thumbPos, y + h);
    glEnd();
}


void drawProgressBar(float x, float y, float w, float h, float progress) {
    // Background
    glColor3f(progressBgColor.r, progressBgColor.g, progressBgColor.b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();

    // Progress
    glColor3f(progressBarColor.r, progressBarColor.g, progressBarColor.b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w * progress, y);
    glVertex2f(x + w * progress, y + h);
    glVertex2f(x, y + h);
    glEnd();
}

bool isInside(float mx, float my, float x, float y, float w, float h) {
    return mx >= x && mx <= x + w && my >= y && my <= y + h;
}

void seekVideo(double position) {
    if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_POS_MSEC, position * 1000);
        currentVideoTime = position;
        lastFrameTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    }
}

void updatePieHover(float fx, float fy) {
    pieHoverIndex = -1;
    if (!showPieMenu || filters.empty()) return;

    float dx = fx - pieMenuCenterX;
    float dy = fy - pieMenuCenterY;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > PIE_MENU_RADIUS || dist < PIE_MENU_INNER) return;

    float angle = std::atan2(dy, dx);
    if (angle < 0.0f) angle += 2.0f * M_PI;
    int idx = static_cast<int>((angle / (2.0f * M_PI)) * filters.size());
    if (idx >= 0 && idx < static_cast<int>(filters.size()))
        pieHoverIndex = idx;
}

void drawPieMenu() {
    if (!showPieMenu || filters.empty()) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const size_t n = filters.size();
    const float sliceAngle = 2.0f * M_PI / static_cast<float>(n);
    const float outerR = PIE_MENU_RADIUS;
    const float innerR = PIE_MENU_INNER;
    const float labelR = (innerR + outerR) * 0.55f;
    const float gap = sliceAngle * 0.04f;
    const int arcSteps = std::max(4, static_cast<int>(48 / n) + 2);

    // Dim the scene behind the menu
    glColor4f(0.0f, 0.0f, 0.0f, 0.35f);
    glBegin(GL_QUADS);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f( 1.0f, -1.0f);
    glVertex2f( 1.0f,  1.0f);
    glVertex2f(-1.0f,  1.0f);
    glEnd();

    // Soft drop shadow
    glColor4f(0.0f, 0.0f, 0.0f, 0.45f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pieMenuCenterX + 0.012f, pieMenuCenterY - 0.018f);
    for (int i = 0; i <= 48; ++i) {
        float a = i * 2.0f * M_PI / 48.0f;
        glVertex2f(pieMenuCenterX + 0.012f + outerR * cos(a),
                   pieMenuCenterY - 0.018f + outerR * sin(a));
    }
    glEnd();

    // Outer ring background
    glColor4f(0.12f, 0.12f, 0.16f, 0.92f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pieMenuCenterX, pieMenuCenterY);
    for (int i = 0; i <= 48; ++i) {
        float a = i * 2.0f * M_PI / 48.0f;
        glVertex2f(pieMenuCenterX + outerR * cos(a),
                   pieMenuCenterY + outerR * sin(a));
    }
    glEnd();

    for (size_t i = 0; i < n; ++i) {
        float a0 = i * sliceAngle + gap * 0.5f;
        float a1 = (i + 1) * sliceAngle - gap * 0.5f;
        float mid = 0.5f * (a0 + a1);
        bool hovered = (static_cast<int>(i) == pieHoverIndex);
        bool enabled = filters[i]->enabled;

        if (hovered)
            glColor4f(0.35f, 0.55f, 0.85f, 0.95f);
        else if (enabled)
            glColor4f(0.22f, 0.55f, 0.30f, 0.90f);
        else
            glColor4f(0.28f, 0.28f, 0.34f, 0.88f);

        glBegin(GL_TRIANGLE_STRIP);
        for (int s = 0; s <= arcSteps; ++s) {
            float t = static_cast<float>(s) / arcSteps;
            float a = a0 + (a1 - a0) * t;
            glVertex2f(pieMenuCenterX + outerR * cos(a), pieMenuCenterY + outerR * sin(a));
            glVertex2f(pieMenuCenterX + innerR * cos(a), pieMenuCenterY + innerR * sin(a));
        }
        glEnd();

        glColor4f(0.55f, 0.55f, 0.62f, 0.55f);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        glVertex2f(pieMenuCenterX + innerR * cos(a0), pieMenuCenterY + innerR * sin(a0));
        glVertex2f(pieMenuCenterX + outerR * cos(a0), pieMenuCenterY + outerR * sin(a0));
        glEnd();

        float tx = pieMenuCenterX + labelR * cos(mid);
        float ty = pieMenuCenterY + labelR * sin(mid);
        std::string num = std::to_string(i + 1);
        glColor3f(0.0f, 0.0f, 0.0f);
        drawText(tx - 0.012f, ty - 0.012f, num, GLUT_BITMAP_HELVETICA_18);
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(tx - 0.01f, ty - 0.01f, num, GLUT_BITMAP_HELVETICA_18);
    }

    // Center hub
    glColor4f(0.18f, 0.18f, 0.22f, 0.98f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pieMenuCenterX, pieMenuCenterY);
    for (int i = 0; i <= 32; ++i) {
        float a = i * 2.0f * M_PI / 32.0f;
        glVertex2f(pieMenuCenterX + innerR * cos(a), pieMenuCenterY + innerR * sin(a));
    }
    glEnd();
    glColor4f(0.55f, 0.55f, 0.65f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i <= 32; ++i) {
        float a = i * 2.0f * M_PI / 32.0f;
        glVertex2f(pieMenuCenterX + innerR * cos(a), pieMenuCenterY + innerR * sin(a));
    }
    glEnd();

    std::string hubText = (pieHoverIndex >= 0 && pieHoverIndex < static_cast<int>(n))
        ? filters[pieHoverIndex]->name()
        : "Close";
    if (pieHoverIndex >= 0 && pieHoverIndex < static_cast<int>(n) && filters[pieHoverIndex]->enabled)
        hubText += " *";
    float hubX = pieMenuCenterX - hubText.length() * 0.0065f;
    glColor3f(0.95f, 0.95f, 0.98f);
    drawText(hubX, pieMenuCenterY - 0.01f, hubText, GLUT_BITMAP_HELVETICA_12);

    glDisable(GL_BLEND);
}

// View transform panel (scale / position)
const float VIEW_PANEL_X = -0.48f;
const float VIEW_PANEL_Y = -0.72f;
const float VIEW_SLIDER_W = 0.22f;
const float VIEW_SLIDER_H = 0.025f;
const float VIEW_SLIDER_GAP = 0.28f;

void drawViewControls() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.75f);
    glBegin(GL_QUADS);
    glVertex2f(VIEW_PANEL_X - 0.02f, VIEW_PANEL_Y - 0.02f);
    glVertex2f(VIEW_PANEL_X + 0.92f, VIEW_PANEL_Y - 0.02f);
    glVertex2f(VIEW_PANEL_X + 0.92f, VIEW_PANEL_Y + 0.10f);
    glVertex2f(VIEW_PANEL_X - 0.02f, VIEW_PANEL_Y + 0.10f);
    glEnd();
    glDisable(GL_BLEND);

    float scaleNorm = (videoScale - VIDEO_SCALE_MIN) / (VIDEO_SCALE_MAX - VIDEO_SCALE_MIN);
    float xNorm = (videoOffsetX + 1.0f) * 0.5f;
    float yNorm = (videoOffsetY + 1.0f) * 0.5f;

    drawSlider(VIEW_PANEL_X, VIEW_PANEL_Y, VIEW_SLIDER_W, VIEW_SLIDER_H, scaleNorm, "Scale");
    drawSlider(VIEW_PANEL_X + VIEW_SLIDER_GAP, VIEW_PANEL_Y, VIEW_SLIDER_W, VIEW_SLIDER_H, xNorm, "Pos X");
    drawSlider(VIEW_PANEL_X + VIEW_SLIDER_GAP * 2.0f, VIEW_PANEL_Y, VIEW_SLIDER_W, VIEW_SLIDER_H, yNorm, "Pos Y");
    drawButton(VIEW_PANEL_X + VIEW_SLIDER_GAP * 3.0f, VIEW_PANEL_Y - 0.01f, 0.10f, 0.05f, "Reset");
}

bool hitTestViewControls(float fx, float fy) {
    auto setFromNorm = [&](int which, float norm) {
        norm = std::max(0.0f, std::min(1.0f, norm));
        activeViewSlider = which;
        if (which == 0)
            videoScale = VIDEO_SCALE_MIN + norm * (VIDEO_SCALE_MAX - VIDEO_SCALE_MIN);
        else if (which == 1)
            videoOffsetX = norm * 2.0f - 1.0f;
        else if (which == 2)
            videoOffsetY = norm * 2.0f - 1.0f;
    };

    if (isInside(fx, fy, VIEW_PANEL_X, VIEW_PANEL_Y, VIEW_SLIDER_W, VIEW_SLIDER_H)) {
        setFromNorm(0, (fx - VIEW_PANEL_X) / VIEW_SLIDER_W);
        return true;
    }
    if (isInside(fx, fy, VIEW_PANEL_X + VIEW_SLIDER_GAP, VIEW_PANEL_Y, VIEW_SLIDER_W, VIEW_SLIDER_H)) {
        setFromNorm(1, (fx - (VIEW_PANEL_X + VIEW_SLIDER_GAP)) / VIEW_SLIDER_W);
        return true;
    }
    if (isInside(fx, fy, VIEW_PANEL_X + VIEW_SLIDER_GAP * 2.0f, VIEW_PANEL_Y, VIEW_SLIDER_W, VIEW_SLIDER_H)) {
        setFromNorm(2, (fx - (VIEW_PANEL_X + VIEW_SLIDER_GAP * 2.0f)) / VIEW_SLIDER_W);
        return true;
    }
    if (isInside(fx, fy, VIEW_PANEL_X + VIEW_SLIDER_GAP * 3.0f, VIEW_PANEL_Y - 0.01f, 0.10f, 0.05f)) {
        videoScale = 1.0f;
        videoOffsetX = 0.0f;
        videoOffsetY = 0.0f;
        activeViewSlider = -1;
        return true;
    }
    return false;
}

bool hitTestViewControls(float fx, float fy);

void postRedisplayBoth();
void setVideoFullscreen(bool enable);
void setControlFullscreen(bool enable);
void applyVideoGeometry();
bool hitTestPlacementPanel(float fx, float fy);
bool hitTestGuidesPanel(float fx, float fy);
bool hitTestFilterListPanel(float fx, float fy);

void mouse(int button, int state, int x, int y) {
    if (!showUI || windowWidth < 1 || windowHeight < 1) return;

    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;

    // Right-click toggles pie menu (also closes if already open)
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        if (sourceMenuMode != SourceMenuMode::None) {
            closeSourceMenu();
            return;
        }
        if (showPieMenu) {
            showPieMenu = false;
            pieHoverIndex = -1;
        } else {
            showPieMenu = true;
            pieMenuCenterX = fx;
            pieMenuCenterY = fy;
            updatePieHover(fx, fy);
        }
        return;
    }

    // Mouse wheel over filter list (FreeGLUT: button 3=up, 4=down)
    if ((button == 3 || button == 4) && state == GLUT_DOWN) {
        if (fx >= FILT_PANEL_X && fx <= FILT_PANEL_X + FILT_PANEL_W &&
            fy >= FILT_PANEL_BOTTOM && fy <= FILT_PANEL_TOP) {
            clampFilterListScroll();
            int visible = filterListVisibleRows();
            int maxScroll = std::max(0, static_cast<int>(filters.size()) - visible);
            if (button == 3)
                filterListScroll = std::max(0, filterListScroll - 1);
            else
                filterListScroll = std::min(maxScroll, filterListScroll + 1);
            return;
        }
    }

    if (button != GLUT_LEFT_BUTTON) return;

    if (button == GLUT_LEFT_BUTTON) {
        mouseLeftDown = (state == GLUT_DOWN);

        if (state == GLUT_DOWN) {
            // Source menus capture input while open
            if (sourceMenuMode != SourceMenuMode::None) {
                hitTestSourceMenu(fx, fy);
                return;
            }

            // Check progress bar for seeking
            if (isInside(fx, fy, -0.95f, -0.95f, 1.9f, 0.03f * uiScale)) {
                isSeeking = true;
                double seekPos = (fx + 0.95f) / 1.9f;
                seekVideo(seekPos * videoDuration);
                return;
            }

            // Check main control buttons (updated y positions to match display())
            if (isInside(fx, fy, -0.95f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                isPaused = !isPaused;
                return;
            }
            else if (isInside(fx, fy, -0.75f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                seekVideo(std::max(0.0, currentVideoTime - 5.0)); // Rewind 5 seconds
                return;
            }
            else if (isInside(fx, fy, -0.55f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                seekVideo(std::min(videoDuration, currentVideoTime + 5.0)); // Fast forward 5 seconds
                return;
            }
            else if (isInside(fx, fy, -0.35f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                isLooping = !isLooping;
                return;
            }
            else if (isInside(fx, fy, -0.15f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                currentVideoIndex = (currentVideoIndex + 1) % playlist.size();
                openVideoSource(playlist[currentVideoIndex]);
                if (cap.isOpened()) {
                    videoDuration = cap.get(cv::CAP_PROP_FRAME_COUNT) / cap.get(cv::CAP_PROP_FPS);
                }
                return;
            }
            else if (isInside(fx, fy, 0.05f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                openFileBrowserMenu();
                return;
            }
            else if (isInside(fx, fy, 0.25f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                openCameraMenu();
                return;
            }
            else if (isInside(fx, fy, 0.45f, -0.85f, 0.14f * uiScale, 0.08f * uiScale)) {
                setVideoFullscreen(!videoFullscreen);
                return;
            }
            else if (isInside(fx, fy, 0.62f, -0.85f, 0.14f * uiScale, 0.08f * uiScale)) {
                setControlFullscreen(!controlFullscreen);
                return;
            }

            // Scrollable filter list (right panel)
            if (hitTestFilterListPanel(fx, fy)) {
                return;
            }

            // Active Effects panel sliders (left side)
            if (hitTestActiveEffectsSliders(fx, fy)) {
                return;
            }

            if (hitTestPlacementPanel(fx, fy)) {
                return;
            }
            if (hitTestGuidesPanel(fx, fy)) {
                return;
            }

            // When pie is open, handle it before other empty-area actions
            if (showPieMenu) {
                float dx = fx - pieMenuCenterX;
                float dy = fy - pieMenuCenterY;
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist <= PIE_MENU_INNER) {
                    showPieMenu = false;
                    pieHoverIndex = -1;
                } else if (dist <= PIE_MENU_RADIUS) {
                    float angle = std::atan2(dy, dx);
                    if (angle < 0.0f) angle += 2.0f * M_PI;
                    int selected = static_cast<int>((angle / (2.0f * M_PI)) * filters.size());
                    if (selected >= 0 && selected < static_cast<int>(filters.size())) {
                        filters[selected]->enabled = !filters[selected]->enabled;
                        activeSlider = selected * 3;
                    }
                } else {
                    showPieMenu = false;
                    pieHoverIndex = -1;
                }
                return;
            }

            // View scale / position controls
            if (hitTestViewControls(fx, fy)) {
                return;
            }

            // Don't open the pie menu over the Active Effects / View panels
            if (anyFilterEnabled()) {
                const float listY = std::max(AE_BOTTOM_LIMIT, AE_TOP_Y - activeEffectsContentHeight());
                if (fx >= AE_PANEL_X && fx <= AE_PANEL_X + AE_PANEL_W &&
                    fy >= listY && fy <= AE_TOP_Y) {
                    return;
                }
            }
            if (fx >= VIEW_PANEL_X - 0.02f && fx <= VIEW_PANEL_X + 0.92f &&
                fy >= VIEW_PANEL_Y - 0.02f && fy <= VIEW_PANEL_Y + 0.10f) {
                return;
            }

            // Left-click empty area opens pie menu
            if (fx < 0.55f) {
                showPieMenu = true;
                pieMenuCenterX = fx;
                pieMenuCenterY = fy;
                updatePieHover(fx, fy);
                return;
            }

        } else if (state == GLUT_UP) {
            activeSlider = -1;
            activeViewSlider = -1;
            activePlaceSlider = -1;
            isSeeking = false;
        }
    }
}

void mouseMotion(int x, int y) {
    if (windowWidth < 1 || windowHeight < 1) return;
    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;

    updatePieHover(fx, fy);

    if (isSeeking && mouseLeftDown) {
        double seekPos = std::max(0.0, std::min(1.0, ((double)fx + 0.95f) / 1.9f));
        seekVideo(seekPos * videoDuration);
        return;
    }

    if (!showUI) return;

    if (activePlaceSlider >= 0 && mouseLeftDown) {
        int sw = std::max(1, glutGet(GLUT_SCREEN_WIDTH));
        int sh = std::max(1, glutGet(GLUT_SCREEN_HEIGHT));
        float norm = 0.0f;
        if (activePlaceSlider == 0 || activePlaceSlider == 2)
            norm = (fx - (PLACE_PANEL_X + 0.02f)) / 0.22f;
        else
            norm = (fx - (PLACE_PANEL_X + 0.28f)) / 0.22f;
        // Y and H use the right column; detect by slider id
        if (activePlaceSlider == 1 || activePlaceSlider == 3)
            norm = (fx - (PLACE_PANEL_X + 0.28f)) / 0.22f;
        if (activePlaceSlider == 0 || activePlaceSlider == 2)
            norm = (fx - (PLACE_PANEL_X + 0.02f)) / 0.22f;
        norm = std::max(0.0f, std::min(1.0f, norm));
        if (activePlaceSlider == 0) videoWinX = int(norm * std::max(1, sw - videoWinW));
        else if (activePlaceSlider == 1) videoWinY = int(norm * std::max(1, sh - videoWinH));
        else if (activePlaceSlider == 2) videoWinW = VIDEO_MIN_W + int(norm * std::max(1, sw - VIDEO_MIN_W));
        else if (activePlaceSlider == 3) videoWinH = VIDEO_MIN_H + int(norm * std::max(1, sh - VIDEO_MIN_H));
        postRedisplayBoth();
        return;
    }

    if (activeViewSlider >= 0 && mouseLeftDown) {
        float norm = 0.0f;
        if (activeViewSlider == 0)
            norm = (fx - VIEW_PANEL_X) / VIEW_SLIDER_W;
        else if (activeViewSlider == 1)
            norm = (fx - (VIEW_PANEL_X + VIEW_SLIDER_GAP)) / VIEW_SLIDER_W;
        else
            norm = (fx - (VIEW_PANEL_X + VIEW_SLIDER_GAP * 2.0f)) / VIEW_SLIDER_W;
        norm = std::max(0.0f, std::min(1.0f, norm));
        if (activeViewSlider == 0)
            videoScale = VIDEO_SCALE_MIN + norm * (VIDEO_SCALE_MAX - VIDEO_SCALE_MIN);
        else if (activeViewSlider == 1)
            videoOffsetX = norm * 2.0f - 1.0f;
        else
            videoOffsetY = norm * 2.0f - 1.0f;
        glutPostRedisplay(); // control window only is fine for slider drag
        return;
    }

    if (activeSlider == -1) return;

    int filterIdx = activeSlider / 3;
    int paramType = activeSlider % 3;
    if (filterIdx < 0 || filterIdx >= static_cast<int>(filters.size())) return;

    float normalizedValue = std::max(0.0f, std::min(1.0f, (fx - AE_CONTENT_X) / AE_SLIDER_W));
    switch (paramType) {
        case 0: filters[filterIdx]->strength = normalizedValue; break;
        case 1: filters[filterIdx]->param1 = normalizedValue; break;
        case 2: filters[filterIdx]->param2 = normalizedValue; break;
    }

    glutPostRedisplay();
}

void passiveMouseMotion(int x, int y) {
    if (windowWidth < 1 || windowHeight < 1) return;
    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;
    int prev = pieHoverIndex;
    updatePieHover(fx, fy);
    if (prev != pieHoverIndex)
        glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // ESC
            if (sourceMenuMode != SourceMenuMode::None) {
                closeSourceMenu();
            } else if (showPieMenu) {
                showPieMenu = false;
            } else if (videoFullscreen) {
                setVideoFullscreen(false);
            } else if (controlFullscreen) {
                setControlFullscreen(false);
            } else {
                exit(0);
            }
            break;
        case ' ':
            isPaused = !isPaused;
            break;
        case 'l':
        case 'L':
            isLooping = !isLooping;
            break;
        case 'n':
        case 'N':
            currentVideoIndex = (currentVideoIndex + 1) % playlist.size();
            openVideoSource(playlist[currentVideoIndex]);
            if (cap.isOpened()) {
                videoDuration = cap.get(cv::CAP_PROP_FRAME_COUNT) / cap.get(cv::CAP_PROP_FPS);
            }
            break;
        case 'f':
        case 'F':
            setVideoFullscreen(!videoFullscreen);
            break;
        case 'h':
        case 'H':
            setControlFullscreen(!controlFullscreen);
            break;
        case ';':
            clampFilterListScroll();
            filterListScroll = std::max(0, filterListScroll - 1);
            break;
        case '\'':
            {
                clampFilterListScroll();
                int visible = filterListVisibleRows();
                int maxScroll = std::max(0, static_cast<int>(filters.size()) - visible);
                filterListScroll = std::min(maxScroll, filterListScroll + 1);
            }
            break;
        case 'u':
        case 'U':
            showUI = !showUI;
            break;
        case 'v':
        case 'V':
            openFileBrowserMenu();
            break;
        case 'c':
        case 'C':
            openCameraMenu();
            break;
        case 'r':
        case 'R':
            if (sourceMenuMode == SourceMenuMode::Files) {
                refreshBrowserEntries();
            } else if (sourceMenuMode == SourceMenuMode::Cameras) {
                scanCameras();
            } else {
                videoScale = 1.0f;
                videoOffsetX = 0.0f;
                videoOffsetY = 0.0f;
            }
            break;
        case 'o':
        case 'O':
            videoScale = std::max(VIDEO_SCALE_MIN, videoScale - 0.05f);
            break;
        case 'p':
        case 'P':
            videoScale = std::min(VIDEO_SCALE_MAX, videoScale + 0.05f);
            break;
        case 'a':
        case 'A':
            videoOffsetX = std::max(-1.0f, videoOffsetX - 0.05f);
            break;
        case 'd':
        case 'D':
            videoOffsetX = std::min(1.0f, videoOffsetX + 0.05f);
            break;
        case 'w':
        case 'W':
            videoOffsetY = std::min(1.0f, videoOffsetY + 0.05f);
            break;
        case 's':
        case 'S':
            videoOffsetY = std::max(-1.0f, videoOffsetY - 0.05f);
            break;
        case '-':
            if (activeSlider >= 0) {
                int filterIdx = activeSlider / 3;
                int paramType = activeSlider % 3;
                float step = 0.05f;
                switch (paramType) {
                    case 0: filters[filterIdx]->strength = std::max(0.0f, filters[filterIdx]->strength - step); break;
                    case 1: filters[filterIdx]->param1 = std::max(0.0f, filters[filterIdx]->param1 - step); break;
                    case 2: filters[filterIdx]->param2 = std::max(0.0f, filters[filterIdx]->param2 - step); break;
                }
            } else {
                videoScale = std::max(VIDEO_SCALE_MIN, videoScale - 0.05f);
            }
            break;
        case '=':
        case '+':
            if (activeSlider >= 0) {
                int filterIdx = activeSlider / 3;
                int paramType = activeSlider % 3;
                float step = 0.05f;
                switch (paramType) {
                    case 0: filters[filterIdx]->strength = std::min(1.0f, filters[filterIdx]->strength + step); break;
                    case 1: filters[filterIdx]->param1 = std::min(1.0f, filters[filterIdx]->param1 + step); break;
                    case 2: filters[filterIdx]->param2 = std::min(1.0f, filters[filterIdx]->param2 + step); break;
                }
            } else {
                videoScale = std::min(VIDEO_SCALE_MAX, videoScale + 0.05f);
            }
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            if ((key - '1') < filters.size()) {
                filters[key - '1']->enabled = !filters[key - '1']->enabled;
                activeSlider = (key - '1') * 3; // Activate strength slider
            }
            break;
        case '[':
            fastForwardSpeed = std::max(1.0f, fastForwardSpeed / 2.0f);
            break;
        case ']':
            fastForwardSpeed = std::min(MAX_FF_SPEED, fastForwardSpeed * 2.0f);
            break;
        case ',':
        case '<':
            seekVideo(std::max(0.0, currentVideoTime - 5.0)); // Rewind 5 seconds
            break;
        case '.':
        case '>':
            seekVideo(std::min(videoDuration, currentVideoTime + 5.0)); // Fast forward 5 seconds
            break;
    }
}

void displayVideoFrame(const cv::Mat& frame) {
    if (frame.empty() || videoWinW < 1 || videoWinH < 1) return;
    if (frame.cols < 1 || frame.rows < 1) return;

    cv::Mat rgb;
    try {
        if (frame.channels() == 1)
            cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
        else if (frame.channels() == 4)
            cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
        else
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } catch (const cv::Exception&) {
        return;
    }
    if (rgb.empty() || rgb.cols < 1 || rgb.rows < 1) return;

    float frameAspect = static_cast<float>(rgb.cols) / static_cast<float>(rgb.rows);
    float windowAspect = static_cast<float>(videoWinW) / static_cast<float>(videoWinH);

    // Fit-contain size, then apply user scale
    float fitW, fitH;
    if (frameAspect > windowAspect) {
        fitW = static_cast<float>(videoWinW);
        fitH = fitW / frameAspect;
    } else {
        fitH = static_cast<float>(videoWinH);
        fitW = fitH * frameAspect;
    }

    int drawWidth = std::max(1, static_cast<int>(fitW * videoScale + 0.5f));
    int drawHeight = std::max(1, static_cast<int>(fitH * videoScale + 0.5f));
    // Cap allocation size to avoid huge temporaries while still allowing zoom via ROI clip
    const int maxDim = std::max(videoWinW, videoWinH) * 4;
    drawWidth = std::min(drawWidth, std::max(1, maxDim));
    drawHeight = std::min(drawHeight, std::max(1, maxDim));

    int offsetX = (videoWinW - drawWidth) / 2 + static_cast<int>(videoOffsetX * videoWinW * 0.5f);
    int offsetY = (videoWinH - drawHeight) / 2 + static_cast<int>(videoOffsetY * videoWinH * 0.5f);

    cv::Mat resized;
    try {
        cv::resize(rgb, resized, cv::Size(drawWidth, drawHeight), 0, 0, cv::INTER_LINEAR);
    } catch (const cv::Exception&) {
        return;
    }
    if (resized.empty()) return;
    if (!resized.isContinuous())
        resized = resized.clone();

    // Clip to the window so glDrawPixels never writes outside the framebuffer
    int srcX = 0, srcY = 0;
    int dstX = offsetX, dstY = offsetY;
    int dstW = drawWidth, dstH = drawHeight;
    if (dstX < 0) { srcX = -dstX; dstW += dstX; dstX = 0; }
    if (dstY < 0) { srcY = -dstY; dstH += dstY; dstY = 0; }
    if (dstX + dstW > videoWinW) dstW = videoWinW - dstX;
    if (dstY + dstH > videoWinH) dstH = videoWinH - dstY;
    if (dstW <= 0 || dstH <= 0) return;
    if (srcX + dstW > resized.cols) dstW = resized.cols - srcX;
    if (srcY + dstH > resized.rows) dstH = resized.rows - srcY;
    if (dstW <= 0 || dstH <= 0) return;

    cv::Mat roi = resized(cv::Rect(srcX, srcY, dstW, dstH));
    if (!roi.isContinuous())
        roi = roi.clone();

    glViewport(0, 0, videoWinW, videoWinH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(videoWinW), 0.0, static_cast<double>(videoWinH));
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2i(dstX, dstY);
    GLboolean valid = GL_FALSE;
    glGetBooleanv(GL_CURRENT_RASTER_POSITION_VALID, &valid);
    if (valid) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glDrawPixels(dstW, dstH, GL_RGB, GL_UNSIGNED_BYTE, roi.data);
    }

    // Reset for overlays (normalized -1..1)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void postRedisplayBoth() {
    // Prefer window-specific post so we don't change the current window
    // (changing current after glutFullScreen can FS the wrong window).
    if (videoWindowId)
        glutPostWindowRedisplay(videoWindowId);
    if (controlWindowId)
        glutPostWindowRedisplay(controlWindowId);
}

void syncVideoGeomFromWindow() {
    if (!videoWindowId) return;
    int prev = glutGetWindow();
    glutSetWindow(videoWindowId);
    videoWinX = glutGet(GLUT_WINDOW_X);
    videoWinY = glutGet(GLUT_WINDOW_Y);
    videoWinW = std::max(1, glutGet(GLUT_WINDOW_WIDTH));
    videoWinH = std::max(1, glutGet(GLUT_WINDOW_HEIGHT));
    if (prev > 0 && prev != videoWindowId)
        glutSetWindow(prev);
}

void applyVideoGeometry() {
    if (!videoWindowId) return;
    // Always target the video window; do not restore previous current window
    // before geometry ops finish (FreeGLUT can apply them to the wrong window).
    glutSetWindow(videoWindowId);
    videoWinW = std::max(VIDEO_MIN_W, videoWinW);
    videoWinH = std::max(VIDEO_MIN_H, videoWinH);
    if (videoFullscreen) {
        glutFullScreen();
    } else {
        glutPositionWindow(videoWinX, videoWinY);
        glutReshapeWindow(videoWinW, videoWinH);
    }
}

void setVideoFullscreen(bool enable) {
    if (!videoWindowId) return;
    glutSetWindow(videoWindowId);
    if (enable) {
        if (!videoFullscreen) {
            videoWinX = glutGet(GLUT_WINDOW_X);
            videoWinY = glutGet(GLUT_WINDOW_Y);
            videoWinW = std::max(VIDEO_MIN_W, glutGet(GLUT_WINDOW_WIDTH));
            videoWinH = std::max(VIDEO_MIN_H, glutGet(GLUT_WINDOW_HEIGHT));
        }
        videoFullscreen = true;
        glutFullScreen();
    } else {
        videoFullscreen = false;
        glutReshapeWindow(std::max(VIDEO_MIN_W, videoWinW), std::max(VIDEO_MIN_H, videoWinH));
        glutPositionWindow(videoWinX, videoWinY);
    }
}

void setControlFullscreen(bool enable) {
    if (!controlWindowId) return;
    glutSetWindow(controlWindowId);
    if (enable) {
        if (!controlFullscreen) {
            controlSavedX = glutGet(GLUT_WINDOW_X);
            controlSavedY = glutGet(GLUT_WINDOW_Y);
            controlSavedW = std::max(1, glutGet(GLUT_WINDOW_WIDTH));
            controlSavedH = std::max(1, glutGet(GLUT_WINDOW_HEIGHT));
            controlWinX = controlSavedX;
            controlWinY = controlSavedY;
            controlWinW = controlSavedW;
            controlWinH = controlSavedH;
        }
        controlFullscreen = true;
        glutFullScreen();
    } else {
        controlFullscreen = false;
        glutReshapeWindow(std::max(1, controlSavedW), std::max(1, controlSavedH));
        glutPositionWindow(controlSavedX, controlSavedY);
        controlWinW = controlSavedW;
        controlWinH = controlSavedH;
        windowWidth = controlWinW;
        windowHeight = controlWinH;
    }
}

void placeVideoPreset(const std::string& name) {
    int sw = std::max(1, glutGet(GLUT_SCREEN_WIDTH));
    int sh = std::max(1, glutGet(GLUT_SCREEN_HEIGHT));
    videoFullscreen = false;
    if (name == "center") {
        videoWinW = std::min(sw * 2 / 3, 1280);
        videoWinH = std::min(sh * 2 / 3, 720);
        videoWinX = (sw - videoWinW) / 2;
        videoWinY = (sh - videoWinH) / 2;
    } else if (name == "topleft") {
        videoWinW = sw / 2; videoWinH = sh / 2;
        videoWinX = 0; videoWinY = 0;
    } else if (name == "topright") {
        videoWinW = sw / 2; videoWinH = sh / 2;
        videoWinX = sw / 2; videoWinY = 0;
    } else if (name == "bottomleft") {
        videoWinW = sw / 2; videoWinH = sh / 2;
        videoWinX = 0; videoWinY = sh / 2;
    } else if (name == "bottomright") {
        videoWinW = sw / 2; videoWinH = sh / 2;
        videoWinX = sw / 2; videoWinY = sh / 2;
    } else if (name == "lefthalf") {
        videoWinW = sw / 2; videoWinH = sh;
        videoWinX = 0; videoWinY = 0;
    } else if (name == "righthalf") {
        videoWinW = sw / 2; videoWinH = sh;
        videoWinX = sw / 2; videoWinY = 0;
    } else if (name == "fullscreen") {
        setVideoFullscreen(true);
        return;
    }
    applyVideoGeometry();
}

void drawGuideRect(float x0, float y0, float x1, float y1) {
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

void drawAspectGuide(float aspect) {
    // aspect = width/height of frame overlay inside window
    float winAspect = static_cast<float>(videoWinW) / static_cast<float>(videoWinH);
    float x0 = -1.0f, y0 = -1.0f, x1 = 1.0f, y1 = 1.0f;
    if (aspect > winAspect) {
        float h = winAspect / aspect;
        y0 = -h; y1 = h;
    } else {
        float w = aspect / winAspect;
        x0 = -w; x1 = w;
    }
    drawGuideRect(x0, y0, x1, y1);
}

void drawCalibrationGuides() {
    bool any = guideCrosshair || guideEdgeBorder || guideThirds || guideGrid ||
               guideActionSafe || guideTitleSafe || guideAspect169 ||
               guideAspect43 || guideAspect239;
    if (!any) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.0f);

    if (guideGrid) {
        glColor4f(1.0f, 1.0f, 1.0f, 0.18f);
        glBegin(GL_LINES);
        for (int i = 1; i < 8; ++i) {
            float t = -1.0f + i * (2.0f / 8.0f);
            glVertex2f(t, -1.0f); glVertex2f(t, 1.0f);
            glVertex2f(-1.0f, t); glVertex2f(1.0f, t);
        }
        glEnd();
    }

    if (guideThirds) {
        glColor4f(0.3f, 0.9f, 1.0f, 0.75f);
        glBegin(GL_LINES);
        glVertex2f(-1.0f / 3.0f, -1.0f); glVertex2f(-1.0f / 3.0f, 1.0f);
        glVertex2f( 1.0f / 3.0f, -1.0f); glVertex2f( 1.0f / 3.0f, 1.0f);
        glVertex2f(-1.0f, -1.0f / 3.0f); glVertex2f(1.0f, -1.0f / 3.0f);
        glVertex2f(-1.0f,  1.0f / 3.0f); glVertex2f(1.0f,  1.0f / 3.0f);
        glEnd();
    }

    if (guideActionSafe) {
        glColor4f(1.0f, 0.85f, 0.2f, 0.85f);
        drawGuideRect(-0.9f, -0.9f, 0.9f, 0.9f);
    }
    if (guideTitleSafe) {
        glColor4f(1.0f, 0.4f, 0.2f, 0.85f);
        drawGuideRect(-0.8f, -0.8f, 0.8f, 0.8f);
    }

    if (guideAspect169) {
        glColor4f(0.4f, 1.0f, 0.5f, 0.9f);
        drawAspectGuide(16.0f / 9.0f);
    }
    if (guideAspect43) {
        glColor4f(0.5f, 0.7f, 1.0f, 0.9f);
        drawAspectGuide(4.0f / 3.0f);
    }
    if (guideAspect239) {
        glColor4f(1.0f, 0.5f, 0.9f, 0.9f);
        drawAspectGuide(2.39f);
    }

    if (guideEdgeBorder) {
        glColor4f(1.0f, 1.0f, 1.0f, 0.95f);
        glLineWidth(2.0f);
        float inset = 2.0f / std::max(videoWinW, 1);
        float insetY = 2.0f / std::max(videoWinH, 1);
        drawGuideRect(-1.0f + inset, -1.0f + insetY, 1.0f - inset, 1.0f - insetY);
        glLineWidth(1.0f);
    }

    if (guideCrosshair) {
        glColor4f(1.0f, 0.2f, 0.2f, 0.95f);
        glLineWidth(1.5f);
        glBegin(GL_LINES);
        glVertex2f(-1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
        glVertex2f(0.0f, -1.0f); glVertex2f(0.0f, 1.0f);
        glEnd();
        glLineWidth(1.0f);
    }

    glDisable(GL_BLEND);
}

void processCaptureFrame() {
    if (isPaused) return;

    double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    double deltaTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;

    if (fastForwardSpeed > 1.0f && videoDuration > 0.0) {
        double targetTime = currentVideoTime + deltaTime * fastForwardSpeed;
        if (targetTime > videoDuration && isLooping)
            targetTime = fmod(targetTime, videoDuration);
        seekVideo(std::min(videoDuration, targetTime));
    }

    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) {
        currentVideoTime = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        for (auto& filter : filters)
            frame = filter->apply(frame);
        cv::flip(frame, frame, 0);
        latestFrame = frame;
    } else {
        if (isLooping) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        } else if (!playlist.empty()) {
            currentVideoIndex = (currentVideoIndex + 1) % playlist.size();
            openVideoSource(playlist[currentVideoIndex]);
            if (cap.isOpened()) {
                double fps = cap.get(cv::CAP_PROP_FPS);
                double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
                videoDuration = (fps > 1e-3 && frames > 0) ? frames / fps : 0.0;
            }
        }
    }
}

// Placement + guides panel (control window)
void drawPlacementPanel() {
    int sw = std::max(1, glutGet(GLUT_SCREEN_WIDTH));
    int sh = std::max(1, glutGet(GLUT_SCREEN_HEIGHT));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.82f);
    glBegin(GL_QUADS);
    glVertex2f(PLACE_PANEL_X, PLACE_PANEL_Y - 0.42f);
    glVertex2f(PLACE_PANEL_X + 0.52f, PLACE_PANEL_Y - 0.42f);
    glVertex2f(PLACE_PANEL_X + 0.52f, PLACE_PANEL_Y + 0.08f);
    glVertex2f(PLACE_PANEL_X, PLACE_PANEL_Y + 0.08f);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1, 1, 1);
    drawText(PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y + 0.04f, "Video Window Placement", GLUT_BITMAP_HELVETICA_12);

    float xN = std::max(0.0f, std::min(1.0f, videoWinX / float(std::max(1, sw - videoWinW))));
    float yN = std::max(0.0f, std::min(1.0f, videoWinY / float(std::max(1, sh - videoWinH))));
    float wN = std::max(0.0f, std::min(1.0f, (videoWinW - VIDEO_MIN_W) / float(std::max(1, sw - VIDEO_MIN_W))));
    float hN = std::max(0.0f, std::min(1.0f, (videoWinH - VIDEO_MIN_H) / float(std::max(1, sh - VIDEO_MIN_H))));

    drawSlider(PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.04f, 0.22f, 0.025f, xN, "Win X");
    drawSlider(PLACE_PANEL_X + 0.28f, PLACE_PANEL_Y - 0.04f, 0.22f, 0.025f, yN, "Win Y");
    drawSlider(PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.14f, 0.22f, 0.025f, wN, "Win W");
    drawSlider(PLACE_PANEL_X + 0.28f, PLACE_PANEL_Y - 0.14f, 0.22f, 0.025f, hN, "Win H");

    drawButton(PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f, "Apply");
    drawButton(PLACE_PANEL_X + 0.14f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f, "Center");
    drawButton(PLACE_PANEL_X + 0.26f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f, "L Half");
    drawButton(PLACE_PANEL_X + 0.38f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f, "R Half");
    drawButton(PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f, "TL");
    drawButton(PLACE_PANEL_X + 0.14f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f, "TR");
    drawButton(PLACE_PANEL_X + 0.26f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f, "BL");
    drawButton(PLACE_PANEL_X + 0.38f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f, "BR");
    drawButton(PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f, "Vid FS", false, videoFullscreen);
    drawButton(PLACE_PANEL_X + 0.145f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f, "Vid Win");
    drawButton(PLACE_PANEL_X + 0.27f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f, "Ctl FS", false, controlFullscreen);
    drawButton(PLACE_PANEL_X + 0.395f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f, "Ctl Win");
}

bool hitTestPlacementPanel(float fx, float fy) {
    int sw = std::max(1, glutGet(GLUT_SCREEN_WIDTH));
    int sh = std::max(1, glutGet(GLUT_SCREEN_HEIGHT));

    auto setPlace = [&](int which, float norm) {
        norm = std::max(0.0f, std::min(1.0f, norm));
        activePlaceSlider = which;
        if (which == 0) videoWinX = int(norm * std::max(1, sw - videoWinW));
        else if (which == 1) videoWinY = int(norm * std::max(1, sh - videoWinH));
        else if (which == 2) videoWinW = VIDEO_MIN_W + int(norm * std::max(1, sw - VIDEO_MIN_W));
        else if (which == 3) videoWinH = VIDEO_MIN_H + int(norm * std::max(1, sh - VIDEO_MIN_H));
    };

    if (isInside(fx, fy, PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.04f, 0.22f, 0.025f)) {
        setPlace(0, (fx - (PLACE_PANEL_X + 0.02f)) / 0.22f); return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.28f, PLACE_PANEL_Y - 0.04f, 0.22f, 0.025f)) {
        setPlace(1, (fx - (PLACE_PANEL_X + 0.28f)) / 0.22f); return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.14f, 0.22f, 0.025f)) {
        setPlace(2, (fx - (PLACE_PANEL_X + 0.02f)) / 0.22f); return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.28f, PLACE_PANEL_Y - 0.14f, 0.22f, 0.025f)) {
        setPlace(3, (fx - (PLACE_PANEL_X + 0.28f)) / 0.22f); return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f)) { applyVideoGeometry(); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.14f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f)) { placeVideoPreset("center"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.26f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f)) { placeVideoPreset("lefthalf"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.38f, PLACE_PANEL_Y - 0.24f, 0.10f, 0.05f)) { placeVideoPreset("righthalf"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f)) { placeVideoPreset("topleft"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.14f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f)) { placeVideoPreset("topright"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.26f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f)) { placeVideoPreset("bottomleft"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.38f, PLACE_PANEL_Y - 0.32f, 0.10f, 0.05f)) { placeVideoPreset("bottomright"); return true; }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.02f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f)) {
        setVideoFullscreen(true);
        return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.145f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f)) {
        setVideoFullscreen(false);
        return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.27f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f)) {
        setControlFullscreen(true);
        return true;
    }
    if (isInside(fx, fy, PLACE_PANEL_X + 0.395f, PLACE_PANEL_Y - 0.40f, 0.115f, 0.055f)) {
        setControlFullscreen(false);
        return true;
    }
    return false;
}

void drawGuidesPanel() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.82f);
    glBegin(GL_QUADS);
    glVertex2f(GUIDE_PANEL_X, GUIDE_PANEL_Y - 0.30f);
    glVertex2f(GUIDE_PANEL_X + 0.52f, GUIDE_PANEL_Y - 0.30f);
    glVertex2f(GUIDE_PANEL_X + 0.52f, GUIDE_PANEL_Y + 0.08f);
    glVertex2f(GUIDE_PANEL_X, GUIDE_PANEL_Y + 0.08f);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1, 1, 1);
    drawText(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y + 0.04f, "Calibration Guides", GLUT_BITMAP_HELVETICA_12);

    auto tog = [&](float x, float y, bool on, const char* label) {
        drawButton(x, y, 0.15f, 0.045f, label, false, on);
    };
    tog(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.04f, guideCrosshair, "Cross");
    tog(GUIDE_PANEL_X + 0.19f, GUIDE_PANEL_Y - 0.04f, guideEdgeBorder, "Edge");
    tog(GUIDE_PANEL_X + 0.36f, GUIDE_PANEL_Y - 0.04f, guideThirds, "Thirds");
    tog(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.11f, guideGrid, "Grid");
    tog(GUIDE_PANEL_X + 0.19f, GUIDE_PANEL_Y - 0.11f, guideActionSafe, "Action");
    tog(GUIDE_PANEL_X + 0.36f, GUIDE_PANEL_Y - 0.11f, guideTitleSafe, "Title");
    tog(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.18f, guideAspect169, "16:9");
    tog(GUIDE_PANEL_X + 0.19f, GUIDE_PANEL_Y - 0.18f, guideAspect43, "4:3");
    tog(GUIDE_PANEL_X + 0.36f, GUIDE_PANEL_Y - 0.18f, guideAspect239, "2.39");
    drawButton(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.27f, 0.22f, 0.05f, "Guides Off");
    drawButton(GUIDE_PANEL_X + 0.28f, GUIDE_PANEL_Y - 0.27f, 0.22f, 0.05f, "Guides All");
}

bool hitTestGuidesPanel(float fx, float fy) {
    auto hit = [&](float x, float y, bool& flag) {
        if (isInside(fx, fy, x, y, 0.15f, 0.045f)) { flag = !flag; return true; }
        return false;
    };
    if (hit(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.04f, guideCrosshair)) return true;
    if (hit(GUIDE_PANEL_X + 0.19f, GUIDE_PANEL_Y - 0.04f, guideEdgeBorder)) return true;
    if (hit(GUIDE_PANEL_X + 0.36f, GUIDE_PANEL_Y - 0.04f, guideThirds)) return true;
    if (hit(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.11f, guideGrid)) return true;
    if (hit(GUIDE_PANEL_X + 0.19f, GUIDE_PANEL_Y - 0.11f, guideActionSafe)) return true;
    if (hit(GUIDE_PANEL_X + 0.36f, GUIDE_PANEL_Y - 0.11f, guideTitleSafe)) return true;
    if (hit(GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.18f, guideAspect169)) return true;
    if (hit(GUIDE_PANEL_X + 0.19f, GUIDE_PANEL_Y - 0.18f, guideAspect43)) return true;
    if (hit(GUIDE_PANEL_X + 0.36f, GUIDE_PANEL_Y - 0.18f, guideAspect239)) return true;
    if (isInside(fx, fy, GUIDE_PANEL_X + 0.02f, GUIDE_PANEL_Y - 0.27f, 0.22f, 0.05f)) {
        guideCrosshair = guideEdgeBorder = guideThirds = guideGrid = false;
        guideActionSafe = guideTitleSafe = guideAspect169 = guideAspect43 = guideAspect239 = false;
        return true;
    }
    if (isInside(fx, fy, GUIDE_PANEL_X + 0.28f, GUIDE_PANEL_Y - 0.27f, 0.22f, 0.05f)) {
        guideCrosshair = guideEdgeBorder = guideThirds = guideGrid = true;
        guideActionSafe = guideTitleSafe = guideAspect169 = guideAspect43 = guideAspect239 = true;
        return true;
    }
    return false;
}

void displayVideoWindow() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!latestFrame.empty())
        displayVideoFrame(latestFrame);
    drawCalibrationGuides();
    glutSwapBuffers();
}

void displayControlWindow() {
    glClearColor(bgColor.r, bgColor.g, bgColor.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (showUI) {
        float progress = videoDuration > 0 ? currentVideoTime / videoDuration : 0;
        drawProgressBar(-0.95f, -0.95f, 1.9f, 0.03f * uiScale, progress);

        std::ostringstream timeText;
        timeText << std::fixed << std::setprecision(1) << currentVideoTime << " / " << videoDuration << "s";
        if (fastForwardSpeed > 1.0f)
            timeText << " (" << fastForwardSpeed << "x)";
        drawText(-0.95f, -0.9f, timeText.str());

        drawButton(-0.95f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, isPaused ? ">" : "||");
        drawButton(-0.75f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, "<<");
        drawButton(-0.55f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, ">>");
        drawButton(-0.35f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, isLooping ? "Loop" : "NoLoop");
        drawButton(-0.15f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, "Next");
        drawButton(0.05f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, "Open");
        drawButton(0.25f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, "Cam");
        drawButton(0.45f, -0.85f, 0.14f * uiScale, 0.08f * uiScale, "Vid FS", false, videoFullscreen);
        drawButton(0.62f, -0.85f, 0.14f * uiScale, 0.08f * uiScale, "Ctl FS", false, controlFullscreen);

        drawFilterListPanel();

        drawViewControls();
        drawPlacementPanel();
        drawGuidesPanel();
        drawActiveEffectsPanel();

        drawText(-0.95f, 0.95f, "F: Video fullscreen | H: Control fullscreen | V Open | C Cam");
        drawText(-0.95f, 0.90f, "Drag video window edges to resize, center to move | Guides + Placement on left");
    }

    drawPieMenu();
    drawSourceMenu();
    glutSwapBuffers();
}

void reshapeVideo(int w, int h) {
    videoWinW = std::max(1, w);
    videoWinH = std::max(1, h);
    if (!videoFullscreen) {
        videoWinX = glutGet(GLUT_WINDOW_X);
        videoWinY = glutGet(GLUT_WINDOW_Y);
    }
    glViewport(0, 0, videoWinW, videoWinH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void reshapeControl(int w, int h) {
    windowWidth = std::max(1, w);
    windowHeight = std::max(1, h);
    controlWinW = windowWidth;
    controlWinH = windowHeight;
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    uiScale = std::min(1.0f, std::max(0.7f, std::min(windowWidth / 800.0f, windowHeight / 600.0f)));
}

void idle() {
    processCaptureFrame();
    postRedisplayBoth();
}

VideoDragMode hitVideoDragMode(int x, int y) {
    bool left = x <= VIDEO_DRAG_EDGE;
    bool right = x >= videoWinW - VIDEO_DRAG_EDGE;
    bool bottom = y >= videoWinH - VIDEO_DRAG_EDGE; // GLUT y grows downward
    bool top = y <= VIDEO_DRAG_EDGE;
    if (top && left) return VideoDragMode::ResizeNW;
    if (top && right) return VideoDragMode::ResizeNE;
    if (bottom && left) return VideoDragMode::ResizeSW;
    if (bottom && right) return VideoDragMode::ResizeSE;
    if (left) return VideoDragMode::ResizeW;
    if (right) return VideoDragMode::ResizeE;
    if (top) return VideoDragMode::ResizeN;
    if (bottom) return VideoDragMode::ResizeS;
    return VideoDragMode::Move;
}

void mouseVideo(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            if (videoFullscreen) return;
            syncVideoGeomFromWindow();
            videoDragMode = hitVideoDragMode(x, y);
            dragStartScreenX = videoWinX + x;
            dragStartScreenY = videoWinY + y;
            dragGeomX = videoWinX;
            dragGeomY = videoWinY;
            dragGeomW = videoWinW;
            dragGeomH = videoWinH;
        } else {
            videoDragMode = VideoDragMode::None;
            syncVideoGeomFromWindow();
        }
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        // right-click video toggles crosshair quickly
        guideCrosshair = !guideCrosshair;
    }
}

void motionVideo(int x, int y) {
    if (videoDragMode == VideoDragMode::None || videoFullscreen) return;
    int screenX = glutGet(GLUT_WINDOW_X) + x;
    int screenY = glutGet(GLUT_WINDOW_Y) + y;
    int dx = screenX - dragStartScreenX;
    int dy = screenY - dragStartScreenY;

    int nx = dragGeomX, ny = dragGeomY, nw = dragGeomW, nh = dragGeomH;
    switch (videoDragMode) {
        case VideoDragMode::Move:
            nx = dragGeomX + dx; ny = dragGeomY + dy; break;
        case VideoDragMode::ResizeE:
            nw = std::max(VIDEO_MIN_W, dragGeomW + dx); break;
        case VideoDragMode::ResizeW:
            nw = std::max(VIDEO_MIN_W, dragGeomW - dx);
            nx = dragGeomX + dragGeomW - nw; break;
        case VideoDragMode::ResizeS:
            nh = std::max(VIDEO_MIN_H, dragGeomH + dy); break;
        case VideoDragMode::ResizeN:
            nh = std::max(VIDEO_MIN_H, dragGeomH - dy);
            ny = dragGeomY + dragGeomH - nh; break;
        case VideoDragMode::ResizeSE:
            nw = std::max(VIDEO_MIN_W, dragGeomW + dx);
            nh = std::max(VIDEO_MIN_H, dragGeomH + dy); break;
        case VideoDragMode::ResizeSW:
            nw = std::max(VIDEO_MIN_W, dragGeomW - dx);
            nx = dragGeomX + dragGeomW - nw;
            nh = std::max(VIDEO_MIN_H, dragGeomH + dy); break;
        case VideoDragMode::ResizeNE:
            nw = std::max(VIDEO_MIN_W, dragGeomW + dx);
            nh = std::max(VIDEO_MIN_H, dragGeomH - dy);
            ny = dragGeomY + dragGeomH - nh; break;
        case VideoDragMode::ResizeNW:
            nw = std::max(VIDEO_MIN_W, dragGeomW - dx);
            nx = dragGeomX + dragGeomW - nw;
            nh = std::max(VIDEO_MIN_H, dragGeomH - dy);
            ny = dragGeomY + dragGeomH - nh; break;
        default: break;
    }
    videoWinX = nx; videoWinY = ny; videoWinW = nw; videoWinH = nh;
    glutPositionWindow(videoWinX, videoWinY);
    glutReshapeWindow(videoWinW, videoWinH);
}

void keyboardVideo(unsigned char key, int, int) {
    switch (key) {
        case 27:
            if (videoFullscreen) setVideoFullscreen(false);
            else exit(0);
            break;
        case 'f': case 'F':
            setVideoFullscreen(!videoFullscreen);
            break;
        case ' ':
            isPaused = !isPaused;
            break;
        case 'g': case 'G':
            guideCrosshair = !guideCrosshair;
            guideEdgeBorder = guideCrosshair;
            break;
    }
}

void updateVideoTime() {
    if (cap.isOpened() && !isPaused) {
        currentVideoTime = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    }
}

int main(int argc, char** argv) {
    auto parseIntArg = [](const std::string& s, int& out) -> bool {
        try { out = std::stoi(s); return true; } catch (...) { return false; }
    };
    auto parseGeometry = [&](const std::string& g, int& w, int& h, int& x, int& y) -> bool {
        // WxH+X+Y or WxH-X-Y
        int W=0,H=0,X=0,Y=0;
        char c1=0,c2=0,c3=0;
        if (sscanf(g.c_str(), "%d%c%d%c%d%c%d", &W, &c1, &H, &c2, &X, &c3, &Y) == 7) {
            if ((c1=='x'||c1=='X') && (c2=='+'||c2=='-') && (c3=='+'||c3=='-')) {
                if (c2=='-') X = -X;
                if (c3=='-') Y = -Y;
                w=W; h=H; x=X; y=Y;
                return true;
            }
        }
        if (sscanf(g.c_str(), "%d%c%d", &W, &c1, &H) == 3 && (c1=='x'||c1=='X')) {
            w=W; h=H; return true;
        }
        return false;
    };
    auto enableGuidesList = [&](const std::string& list) {
        auto enableOne = [&](const std::string& g) {
            if (g=="cross"||g=="crosshair") guideCrosshair = true;
            else if (g=="edge"||g=="border") guideEdgeBorder = true;
            else if (g=="thirds") guideThirds = true;
            else if (g=="grid") guideGrid = true;
            else if (g=="action"||g=="actionsafe") guideActionSafe = true;
            else if (g=="title"||g=="titlesafe") guideTitleSafe = true;
            else if (g=="16:9"||g=="169") guideAspect169 = true;
            else if (g=="4:3"||g=="43") guideAspect43 = true;
            else if (g=="2.39"||g=="239") guideAspect239 = true;
            else if (g=="all") {
                guideCrosshair = guideEdgeBorder = guideThirds = guideGrid = true;
                guideActionSafe = guideTitleSafe = guideAspect169 = guideAspect43 = guideAspect239 = true;
            }
        };
        if (list.empty() || list == "all") { enableOne("all"); return; }
        std::stringstream ss(list);
        std::string item;
        while (std::getline(ss, item, ',')) {
            while (!item.empty() && item.front()==' ') item.erase(item.begin());
            while (!item.empty() && item.back()==' ') item.pop_back();
            enableOne(item);
        }
    };

    std::unordered_set<std::string> enabled;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needVal = [&](int& dst, bool& flag) {
            if (i + 1 < argc && parseIntArg(argv[i + 1], dst)) {
                ++i; flag = true; return true;
            }
            return false;
        };
        if (arg.rfind("--enable-", 0) == 0)
            enabled.insert(arg.substr(9));
        else if (arg == "--no-ui")
            showUI = false;
        else if (arg == "--fullscreen")
            videoFullscreen = true;
        else if (arg == "--control-fullscreen")
            controlFullscreen = true;
        else if (arg == "--video-x" && needVal(videoWinX, videoGeomFromArgs)) {}
        else if (arg == "--video-y" && needVal(videoWinY, videoGeomFromArgs)) {}
        else if (arg == "--video-w" && needVal(videoWinW, videoGeomFromArgs)) {}
        else if (arg == "--video-h" && needVal(videoWinH, videoGeomFromArgs)) {}
        else if (arg == "--control-x" && needVal(controlWinX, controlGeomFromArgs)) {}
        else if (arg == "--control-y" && needVal(controlWinY, controlGeomFromArgs)) {}
        else if (arg == "--control-w" && needVal(controlWinW, controlGeomFromArgs)) {}
        else if (arg == "--control-h" && needVal(controlWinH, controlGeomFromArgs)) {}
        else if (arg.rfind("--video-geometry=", 0) == 0) {
            if (parseGeometry(arg.substr(17), videoWinW, videoWinH, videoWinX, videoWinY))
                videoGeomFromArgs = true;
        } else if (arg == "--video-geometry" && i + 1 < argc) {
            if (parseGeometry(argv[++i], videoWinW, videoWinH, videoWinX, videoWinY))
                videoGeomFromArgs = true;
        } else if (arg.rfind("--control-geometry=", 0) == 0) {
            if (parseGeometry(arg.substr(19), controlWinW, controlWinH, controlWinX, controlWinY))
                controlGeomFromArgs = true;
        } else if (arg == "--control-geometry" && i + 1 < argc) {
            if (parseGeometry(argv[++i], controlWinW, controlWinH, controlWinX, controlWinY))
                controlGeomFromArgs = true;
        } else if (arg == "--guides") {
            enableGuidesList("all");
        } else if (arg.rfind("--guides=", 0) == 0) {
            enableGuidesList(arg.substr(9));
        } else if (arg == "--preset" && i + 1 < argc) {
            // applied after GLUT init
            playlist.push_back(std::string("__preset__") + argv[++i]);
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg << std::endl;
        } else
            playlist.push_back(arg);
    }

    // Extract delayed preset token if present
    std::string startupPreset;
    playlist.erase(std::remove_if(playlist.begin(), playlist.end(), [&](const std::string& s) {
        if (s.rfind("__preset__", 0) == 0) {
            startupPreset = s.substr(10);
            return true;
        }
        return false;
    }), playlist.end());

    if (playlist.empty()) playlist.push_back("0");

    // Initialize video capture
    if (!openVideoSource(playlist[0])) {
        std::cerr << "Failed to open source." << std::endl;
        return -1;
    }

    // Get video duration
    {
        double fps = cap.get(cv::CAP_PROP_FPS);
        double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        videoDuration = (fps > 1e-3 && frames > 0) ? frames / fps : 0.0;
    }

    // Initialize filters
    auto anaglyph = std::make_unique<Anaglyph3DFilter>();
    anaglyph->enabled = enabled.count("anaglyph");
    filters.push_back(std::move(anaglyph));

    auto anaglyphSeq = std::make_unique<AlternatingAnaglyphFilter>();
    anaglyphSeq->enabled = enabled.count("anaglyphseq") || enabled.count("anaglyph-seq");
    filters.push_back(std::move(anaglyphSeq));

    auto gray = std::make_unique<GrayscaleFilter>();
    gray->enabled = enabled.count("gray");
    filters.push_back(std::move(gray));

    auto edge = std::make_unique<EdgeDetectFilter>();
    edge->enabled = enabled.count("edge");
    filters.push_back(std::move(edge));

    auto sepia = std::make_unique<SepiaFilter>();
    sepia->enabled = enabled.count("sepia");
    filters.push_back(std::move(sepia));

    auto pixelate = std::make_unique<PixelateFilter>();
    pixelate->enabled = enabled.count("pixelate");
    filters.push_back(std::move(pixelate));

    auto invert = std::make_unique<InvertFilter>();
    invert->enabled = enabled.count("invert");
    filters.push_back(std::move(invert));

    auto blur = std::make_unique<BlurFilter>();
    blur->enabled = enabled.count("blur");
    filters.push_back(std::move(blur));

    auto vignette = std::make_unique<VignetteFilter>();
    vignette->enabled = enabled.count("vignette");
    filters.push_back(std::move(vignette));

    auto noise = std::make_unique<NoiseFilter>();
    noise->enabled = enabled.count("noise");
    filters.push_back(std::move(noise));

    auto colorBalance = std::make_unique<ColorBalanceFilter>();
    colorBalance->enabled = enabled.count("colorbalance");
    filters.push_back(std::move(colorBalance));

    auto scanlines = std::make_unique<ScanlinesFilter>();
    scanlines->enabled = enabled.count("scanlines");
    filters.push_back(std::move(scanlines));

    auto crt = std::make_unique<CRTFilter>();
    crt->enabled = enabled.count("crt");
    filters.push_back(std::move(crt));

    auto glitch = std::make_unique<GlitchFilter>();
    glitch->enabled = enabled.count("glitch");
    filters.push_back(std::move(glitch));

    auto kaleidoscope = std::make_unique<KaleidoscopeFilter>();
    kaleidoscope->enabled = enabled.count("kaleidoscope");
    filters.push_back(std::move(kaleidoscope));

    auto nightVision = std::make_unique<NightVisionFilter>();
    nightVision->enabled = enabled.count("nightvision");
    filters.push_back(std::move(nightVision));

    auto colorize = std::make_unique<ColorizeFilter>();
    colorize->enabled = enabled.count("colorize");
    filters.push_back(std::move(colorize));


    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // Video window (stream + guides only)
    glutInitWindowSize(std::max(VIDEO_MIN_W, videoWinW), std::max(VIDEO_MIN_H, videoWinH));
    glutInitWindowPosition(videoWinX, videoWinY);
    videoWindowId = glutCreateWindow("Enceladus Video");
    glutDisplayFunc(displayVideoWindow);
    glutReshapeFunc(reshapeVideo);
    glutMouseFunc(mouseVideo);
    glutMotionFunc(motionVideo);
    glutKeyboardFunc(keyboardVideo);

    // Control panel window
    if (showUI) {
        windowWidth = controlWinW;
        windowHeight = controlWinH;
        glutInitWindowSize(controlWinW, controlWinH);
        glutInitWindowPosition(controlWinX, controlWinY);
        controlWindowId = glutCreateWindow("Enceladus Control Panel");
        glutDisplayFunc(displayControlWindow);
        glutReshapeFunc(reshapeControl);
        glutMouseFunc(mouse);
        glutMotionFunc(mouseMotion);
        glutPassiveMotionFunc(passiveMouseMotion);
        glutKeyboardFunc(keyboard);
    }

    glutIdleFunc(idle);
    lastFrameTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;

    if (!startupPreset.empty())
        placeVideoPreset(startupPreset);
    else if (videoGeomFromArgs || !videoFullscreen)
        applyVideoGeometry();

    if (videoFullscreen)
        setVideoFullscreen(true);

    if (controlWindowId) {
        glutSetWindow(controlWindowId);
        glutPositionWindow(controlWinX, controlWinY);
        glutReshapeWindow(controlWinW, controlWinH);
        if (controlFullscreen)
            setControlFullscreen(true);
    }

    glutMainLoop();
    return 0;
}
