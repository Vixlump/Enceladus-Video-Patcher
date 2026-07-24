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

// Global variables
cv::Mat latestFrame;
bool frameAvailable = false;
bool isPaused = false;
bool isLooping = false;
bool showUI = true;
int currentVideoIndex = 0;
std::vector<std::string> playlist;
int windowWidth = 800, windowHeight = 600;
float uiScale = 1.0f;
bool fullscreen = false;
int activeSlider = -1; // -1 means no slider is active
bool isSeeking = false;
double videoDuration = 0;
double currentVideoTime = 0;
double lastFrameTime = 0;
bool mouseLeftDown = false;
float fastForwardSpeed = 1.0f;
const float MAX_FF_SPEED = 16.0f;

//pi menu vars

bool showPieMenu = false;
float pieMenuCenterX = 0, pieMenuCenterY = 0;
int selectedFilterIndex = -1;
int pieHoverIndex = -1;
const float PIE_MENU_RADIUS = 0.40f;
const float PIE_MENU_INNER = 0.10f;

// Video placement within the window (1.0 = fit, offsets are normalized -1..1)
float videoScale = 1.0f;
float videoOffsetX = 0.0f;
float videoOffsetY = 0.0f;
int activeViewSlider = -1; // 0=scale, 1=posX, 2=posY
const float VIDEO_SCALE_MIN = 0.2f;
const float VIDEO_SCALE_MAX = 2.0f;

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

// Utility functions
void drawText(float x, float y, const std::string& text, void* font = GLUT_BITMAP_HELVETICA_12) {
    glRasterPos2f(x, y);
    for (char c : text)
        glutBitmapCharacter(font, c);
}

void drawButton(float x, float y, float w, float h, const std::string& label, bool hover = false, bool active = false) {
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
bool isInside(float mx, float my, float x, float y, float w, float h);

// Active-effects panel sits on the left so sliders never overlap the filter buttons.
const float AE_PANEL_X = -0.95f;
const float AE_PANEL_W = 0.48f;
const float AE_CONTENT_X = -0.93f;
const float AE_SLIDER_W = 0.42f;
const float AE_SLIDER_H = 0.025f;
const float AE_TOP_Y = 0.78f;
const float AE_TITLE_STEP = 0.06f;
const float AE_NAME_STEP = 0.05f;
const float AE_SLIDER_STEP = 0.07f;
const float AE_FILTER_GAP = 0.035f;
const float AE_BOTTOM_LIMIT = -0.72f;

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

void mouse(int button, int state, int x, int y) {
    if (!showUI) return;

    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;

    // Right-click toggles pie menu (also closes if already open)
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
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

    if (button != GLUT_LEFT_BUTTON) return;

    if (button == GLUT_LEFT_BUTTON) {
        mouseLeftDown = (state == GLUT_DOWN);

        if (state == GLUT_DOWN) {
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
            else if (isInside(fx, fy, 0.8f, -0.85f, 0.15f * uiScale, 0.08f * uiScale)) {
                fullscreen = !fullscreen;
                if (fullscreen) {
                    glutFullScreen();
                } else {
                    glutReshapeWindow(800, 600);
                    glutPositionWindow(100, 100);
                }
                return;
            }

            // Check filter toggles (updated y positions)
            for (size_t i = 0; i < filters.size(); ++i) {
                float bx = 0.6f * uiScale;
                float by = -0.8f * uiScale + 0.15f * uiScale * i;
                if (isInside(fx, fy, bx, by, 0.35f * uiScale, 0.08f * uiScale)) {
                    filters[i]->enabled = !filters[i]->enabled;
                    return;
                }
            }

            // Active Effects panel sliders (left side)
            if (hitTestActiveEffectsSliders(fx, fy)) {
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
            isSeeking = false;
        }
    }
}

void mouseMotion(int x, int y) {
    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;

    updatePieHover(fx, fy);

    if (isSeeking && mouseLeftDown) {
        double seekPos = std::max(0.0, std::min(1.0, ((double)fx + 0.95f) / 1.9f));
        seekVideo(seekPos * videoDuration);
        return;
    }

    if (!showUI) return;

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
        glutPostRedisplay();
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
            if (showPieMenu) {
                showPieMenu = false;
            } else if (fullscreen) {
                fullscreen = false;
                glutReshapeWindow(800, 600);
                glutPositionWindow(100, 100);
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
            fullscreen = !fullscreen;
            if (fullscreen) {
                glutFullScreen();
            } else {
                glutReshapeWindow(800, 600);
                glutPositionWindow(100, 100);
            }
            break;
        case 'u':
        case 'U':
            showUI = !showUI;
            break;
        case 'r':
        case 'R':
            videoScale = 1.0f;
            videoOffsetX = 0.0f;
            videoOffsetY = 0.0f;
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
    if (frame.empty()) return;

    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    float frameAspect = (float)rgb.cols / (float)rgb.rows;
    float windowAspect = (float)windowWidth / (float)windowHeight;

    // Fit-contain size, then apply user scale
    float fitW, fitH;
    if (frameAspect > windowAspect) {
        fitW = (float)windowWidth;
        fitH = fitW / frameAspect;
    } else {
        fitH = (float)windowHeight;
        fitW = fitH * frameAspect;
    }

    int drawWidth = std::max(1, static_cast<int>(fitW * videoScale));
    int drawHeight = std::max(1, static_cast<int>(fitH * videoScale));
    int offsetX = (windowWidth - drawWidth) / 2 + static_cast<int>(videoOffsetX * windowWidth * 0.5f);
    int offsetY = (windowHeight - drawHeight) / 2 + static_cast<int>(videoOffsetY * windowHeight * 0.5f);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(drawWidth, drawHeight), 0, 0, cv::INTER_LINEAR);
    if (!resized.isContinuous())
        resized = resized.clone();

    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, windowWidth, 0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2i(offsetX, offsetY);
    // Avoid invalid raster positions when heavily offset
    GLboolean valid = GL_TRUE;
    glGetBooleanv(GL_CURRENT_RASTER_POSITION_VALID, &valid);
    if (valid) {
        glDrawPixels(resized.cols, resized.rows, GL_RGB, GL_UNSIGNED_BYTE, resized.data);
    }

    // Reset for UI
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void display() {
    glClearColor(bgColor.r, bgColor.g, bgColor.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw video frame if available
    if (frameAvailable) {
        if (!isPaused) {
            double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
            double deltaTime = currentTime - lastFrameTime;
            lastFrameTime = currentTime;

            // Apply fast forward speed
            if (fastForwardSpeed > 1.0f) {
                double targetTime = currentVideoTime + deltaTime * fastForwardSpeed;
                if (targetTime > videoDuration && isLooping) {
                    targetTime = fmod(targetTime, videoDuration);
                }
                seekVideo(std::min(videoDuration, targetTime));
            }

            cv::Mat frame;
            cap >> frame;
            if (!frame.empty()) {
                currentVideoTime = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
                
                for (auto& filter : filters)
                    frame = filter->apply(frame);

                cv::flip(frame, frame, 0); // Flip vertically for OpenGL
                latestFrame = frame;
                displayVideoFrame(frame);
            } else {
                if (isLooping) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                } else {
                    currentVideoIndex = (currentVideoIndex + 1) % playlist.size();
                    openVideoSource(playlist[currentVideoIndex]);
                    if (cap.isOpened()) {
                        videoDuration = cap.get(cv::CAP_PROP_FRAME_COUNT) / cap.get(cv::CAP_PROP_FPS);
                    }
                }
            }
        } else {
            displayVideoFrame(latestFrame);
        }
        frameAvailable = false;
    }

    // Draw UI
    if (showUI) {
        // Progress bar
        float progress = videoDuration > 0 ? currentVideoTime / videoDuration : 0;
        drawProgressBar(-0.95f, -0.95f, 1.9f, 0.03f * uiScale, progress);

        // Time display
        std::ostringstream timeText;
        timeText << std::fixed << std::setprecision(1) << currentVideoTime << " / " << videoDuration << "s";
        if (fastForwardSpeed > 1.0f) {
            timeText << " (" << fastForwardSpeed << "x)";
        }
        drawText(-0.95f, -0.9f, timeText.str());

        // Main controls
        drawButton(-0.95f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, isPaused ? ">" : "||");
        drawButton(-0.75f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, "<<");
        drawButton(-0.55f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, ">>");
        drawButton(-0.35f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, isLooping ? "Loop" : "NoLoop");
        drawButton(-0.15f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, "Next");
        drawButton(0.8f, -0.85f, 0.15f * uiScale, 0.08f * uiScale, fullscreen ? "FullScreen" : "Windowed");

        // Filter controls (toggles only — sliders live in the left Active Effects panel)
        for (size_t i = 0; i < filters.size(); ++i) {
            float bx = 0.6f * uiScale;
            float by = -0.8f * uiScale + 0.15f * uiScale * i;
            
            std::string label = std::to_string(i+1) + ". " + filters[i]->name() + (filters[i]->enabled ? " [ON]" : " [OFF]");
            drawButton(bx, by, 0.35f * uiScale, 0.08f * uiScale, label, false, filters[i]->enabled);
        }

        drawActiveEffectsPanel();
        drawViewControls();

        // Help text
        drawText(-0.95f, 0.95f, "Space: Play/Pause | L: Loop | N: Next | F: Fullscreen | U: Toggle UI | 1-9: Filters");
        drawText(-0.95f, 0.9f, "Right-click: Pie menu | WASD: Move video | O/P: Scale | R: Reset view | [/]: Speed");
    }

    drawPieMenu();

    glutSwapBuffers();
}

void reshape(int w, int h) {
    windowWidth = w;
    windowHeight = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Adjust UI scale based on window size
    uiScale = std::min(1.0f, std::max(0.7f, std::min(w / 800.0f, h / 600.0f)));
}

void idle() {
    frameAvailable = true;
    glutPostRedisplay();
}

void updateVideoTime() {
    if (cap.isOpened() && !isPaused) {
        currentVideoTime = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::unordered_set<std::string> enabled;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--enable-", 0) == 0)
            enabled.insert(arg.substr(9));
        else if (arg == "--no-ui")
            showUI = false;
        else if (arg == "--fullscreen")
            fullscreen = true;
        else
            playlist.push_back(arg);
    }
    if (playlist.empty()) playlist.push_back("0");

    // Initialize video capture
    if (!openVideoSource(playlist[0])) {
        std::cerr << "Failed to open source." << std::endl;
        return -1;
    }

    // Get video duration
    videoDuration = cap.get(cv::CAP_PROP_FRAME_COUNT) / cap.get(cv::CAP_PROP_FPS);
    lastFrameTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;

    // Initialize filters
    auto anaglyph = std::make_unique<Anaglyph3DFilter>();
    anaglyph->enabled = enabled.count("anaglyph");
    filters.push_back(std::move(anaglyph));

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
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Enceladus Video Patcher - Lumpology");
    
    if (fullscreen) {
        glutFullScreen();
    }

    // Set callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutPassiveMotionFunc(passiveMouseMotion);
    glutKeyboardFunc(keyboard);

    // Main loop
    glutMainLoop();

    return 0;
}