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
const float PIE_MENU_RADIUS = 0.3f;

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

void drawPieMenu() {
    if (!showPieMenu || filters.empty()) return;
    
    const float sliceAngle = 2.0f * M_PI / filters.size();
    const float iconSize = 0.05f * uiScale;
    const float innerRadius = PIE_MENU_RADIUS * 0.3f;
    const float textRadius = PIE_MENU_RADIUS * 0.7f;
    const float outerRadius = PIE_MENU_RADIUS;
    
    // Draw background circle with subtle shadow
    glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pieMenuCenterX, pieMenuCenterY - 0.02f);
    for (int i = 0; i <= 32; i++) {
        float angle = i * 2.0f * M_PI / 32;
        glVertex2f(pieMenuCenterX + outerRadius * cos(angle), 
                  pieMenuCenterY + outerRadius * sin(angle) - 0.02f);
    }
    glEnd();
    
    // Draw main pie menu background
    glColor4f(0.15f, 0.15f, 0.2f, 0.9f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pieMenuCenterX, pieMenuCenterY);
    for (int i = 0; i <= 32; i++) {
        float angle = i * 2.0f * M_PI / 32;
        glVertex2f(pieMenuCenterX + outerRadius * cos(angle), 
                  pieMenuCenterY + outerRadius * sin(angle));
    }
    glEnd();
    
    // Draw slices with highlight effect
    for (size_t i = 0; i < filters.size(); ++i) {
        float angle = i * sliceAngle;
        float midAngle = angle + sliceAngle/2;
        
        // Draw slice with highlight if enabled
        if (filters[i]->enabled) {
            glColor4f(0.3f, 0.6f, 0.3f, 0.7f);
        } else {
            glColor4f(0.3f, 0.3f, 0.4f, 0.7f);
        }
        
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(pieMenuCenterX, pieMenuCenterY);
        for (int j = 0; j <= 3; ++j) {
            float a = angle + (j * sliceAngle / 3);
            glVertex2f(pieMenuCenterX + outerRadius * cos(a), 
                      pieMenuCenterY + outerRadius * sin(a));
        }
        glEnd();
        
        // Draw slice border
        glColor4f(0.5f, 0.5f, 0.6f, 0.8f);
        glLineWidth(1.5f);
        glBegin(GL_LINE_STRIP);
        glVertex2f(pieMenuCenterX, pieMenuCenterY);
        glVertex2f(pieMenuCenterX + outerRadius * cos(angle), 
                  pieMenuCenterY + outerRadius * sin(angle));
        glVertex2f(pieMenuCenterX + outerRadius * cos(angle + sliceAngle), 
                  pieMenuCenterY + outerRadius * sin(angle + sliceAngle));
        glVertex2f(pieMenuCenterX, pieMenuCenterY);
        glEnd();
        
        // Calculate text position
        float textX = pieMenuCenterX + textRadius * cos(midAngle);
        float textY = pieMenuCenterY + textRadius * sin(midAngle);
        
        // Draw filter number with outline
        glColor3f(0.0f, 0.0f, 0.0f);
        for (float dx = -0.002f; dx <= 0.002f; dx += 0.002f) {
            for (float dy = -0.002f; dy <= 0.002f; dy += 0.002f) {
                drawText(textX + dx - iconSize/2, textY + dy - iconSize/2, 
                        std::to_string(i+1), GLUT_BITMAP_HELVETICA_18);
            }
        }
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(textX - iconSize/2, textY - iconSize/2, 
                std::to_string(i+1), GLUT_BITMAP_HELVETICA_18);
        
        // Draw filter name with outline
        float nameX = textX - (filters[i]->name().length() * 0.01f);
        float nameY = textY - iconSize/2 - 0.05f;
        
        glColor3f(0.0f, 0.0f, 0.0f);
        for (float dx = -0.0015f; dx <= 0.0015f; dx += 0.0015f) {
            for (float dy = -0.0015f; dy <= 0.0015f; dy += 0.0015f) {
                drawText(nameX + dx, nameY + dy, filters[i]->name());
            }
        }
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(nameX, nameY, filters[i]->name());
    }
    
    // Draw center circle with highlight
    glColor4f(0.4f, 0.4f, 0.5f, 1.0f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pieMenuCenterX, pieMenuCenterY);
    for (int i = 0; i <= 16; i++) {
        float angle = i * 2.0f * M_PI / 16;
        glVertex2f(pieMenuCenterX + innerRadius * cos(angle), 
                  pieMenuCenterY + innerRadius * sin(angle));
    }
    glEnd();
    
    // Draw active effects sliders list if any filters are enabled
    bool anyEnabled = false;
    for (const auto& filter : filters) {
        if (filter->enabled) {
            anyEnabled = true;
            break;
        }
    }
    
    if (anyEnabled) {
        const float sliderWidth = 0.3f;
        const float sliderHeight = 0.03f;
        const float sliderSpacing = 0.08f; // Increased spacing
        const float listX = pieMenuCenterX - 0.55f; // Move further left
        const float listY = pieMenuCenterY - 0.4f;
        const float listWidth = 0.5f; // Wider to accommodate text
        const float listHeight = 0.8f;
        
        // Draw list background with more transparency
        glColor4f(0.1f, 0.1f, 0.15f, 0.7f);
        glBegin(GL_QUADS);
        glVertex2f(listX, listY);
        glVertex2f(listX + listWidth, listY);
        glVertex2f(listX + listWidth, listY + listHeight);
        glVertex2f(listX, listY + listHeight);
        glEnd();
        
        // Draw list border
        glColor4f(0.3f, 0.3f, 0.4f, 0.9f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(listX, listY);
        glVertex2f(listX + listWidth, listY);
        glVertex2f(listX + listWidth, listY + listHeight);
        glVertex2f(listX, listY + listHeight);
        glEnd();
        
        // Draw title with larger font
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(listX + 0.02f, listY + listHeight - 0.04f, "Active Effects:", GLUT_BITMAP_HELVETICA_18);
        
        // Draw sliders for each enabled filter
        float currentY = listY + listHeight - 0.1f; // More initial space
        for (size_t i = 0; i < filters.size(); ++i) {
            if (!filters[i]->enabled) continue;
            
            // Filter name with larger font
            glColor3f(0.8f, 0.8f, 1.0f);
            drawText(listX + 0.02f, currentY, filters[i]->name(), GLUT_BITMAP_HELVETICA_18);
            currentY -= 0.06f;
            
            // Strength slider
            if (filters[i]->hasStrength()) {
                drawSlider(listX + 0.02f, currentY - 0.02f, sliderWidth, sliderHeight, 
                          filters[i]->strength, "Strength");
                currentY -= sliderSpacing;
            }
            
            // Param1 slider
            if (filters[i]->hasParam1()) {
                drawSlider(listX + 0.02f, currentY - 0.02f, sliderWidth, sliderHeight, 
                          filters[i]->param1, filters[i]->param1Name());
                currentY -= sliderSpacing;
            }
            
            // Param2 slider
            if (filters[i]->hasParam2()) {
                drawSlider(listX + 0.02f, currentY - 0.02f, sliderWidth, sliderHeight, 
                          filters[i]->param2, filters[i]->param2Name());
                currentY -= sliderSpacing;
            }
            
            currentY -= 0.04f; // Extra spacing between filters
        }
    }
}

void mouse(int button, int state, int x, int y) {
    if (button != GLUT_LEFT_BUTTON || !showUI) return;

    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;

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
                cap.open(playlist[currentVideoIndex]);
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

            // Check sliders (updated y positions)
            for (size_t i = 0; i < filters.size(); ++i) {
                if (!filters[i]->enabled) continue;
                
                float sy = -0.7f * uiScale + 0.15f * uiScale * i;
                
                if (filters[i]->hasStrength()) {
                    if (isInside(fx, fy, 0.6f * uiScale, sy, 0.35f * uiScale, 0.03f * uiScale)) {
                        activeSlider = i * 3; // Strength slider
                        filters[i]->strength = std::max(0.0f, std::min(1.0f, (fx - 0.6f * uiScale) / (0.35f * uiScale)));
                        return;
                    }
                    sy += 0.05f * uiScale;
                }
                
                if (filters[i]->hasParam1()) {
                    if (isInside(fx, fy, 0.6f * uiScale, sy, 0.35f * uiScale, 0.03f * uiScale)) {
                        activeSlider = i * 3 + 1; // Param1 slider
                        filters[i]->param1 = std::max(0.0f, std::min(1.0f, (fx - 0.6f * uiScale) / (0.35f * uiScale)));
                        return;
                    }
                    sy += 0.05f * uiScale;
                }
                
                if (filters[i]->hasParam2()) {
                    if (isInside(fx, fy, 0.6f * uiScale, sy, 0.35f * uiScale, 0.03f * uiScale)) {
                        activeSlider = i * 3 + 2; // Param2 slider
                        filters[i]->param2 = std::max(0.0f, std::min(1.0f, (fx - 0.6f * uiScale) / (0.35f * uiScale)));
                        return;
                    }
                }
            }
            if (fx < 0.5f && !showPieMenu) {
                showPieMenu = !showPieMenu;
                pieMenuCenterX = fx;
                pieMenuCenterY = fy;
                return;
            }

            if (showPieMenu) {
                // Check pie menu selection
                float dx = fx - pieMenuCenterX;
                float dy = fy - pieMenuCenterY;
                float dist = sqrt(dx*dx + dy*dy);
                
                // First check if we clicked on any slider in the active effects list
                bool anyEnabled = false;
                for (const auto& filter : filters) {
                    if (filter->enabled) {
                        anyEnabled = true;
                        break;
                    }
                }
                
                if (anyEnabled && fx < pieMenuCenterX - 0.1f) {
                    // Check if click is in the slider list area
                    const float listX = pieMenuCenterX - 0.5f;
                    const float listY = pieMenuCenterY - 0.4f;
                    const float listWidth = 0.4f;
                    const float listHeight = 0.8f;
                    
                    if (fx >= listX && fx <= listX + listWidth && 
                        fy >= listY && fy <= listY + listHeight) {
                        
                        // Find which slider was clicked
                        float currentY = listY + listHeight - 0.08f;
                        for (size_t i = 0; i < filters.size(); ++i) {
                            if (!filters[i]->enabled) continue;
                            
                            currentY -= 0.05f; // Skip name
                            
                            if (filters[i]->hasStrength()) {
                                if (fy >= currentY && fy <= currentY + 0.03f * uiScale) {
                                    activeSlider = i * 3;
                                    filters[i]->strength = std::max(0.0f, std::min(1.0f, 
                                        (fx - (listX + 0.02f)) / 0.3f));
                                    break;
                                }
                                currentY -= 0.05f;
                            }
                            
                            if (filters[i]->hasParam1()) {
                                if (fy >= currentY && fy <= currentY + 0.03f * uiScale) {
                                    activeSlider = i * 3 + 1;
                                    filters[i]->param1 = std::max(0.0f, std::min(1.0f, 
                                        (fx - (listX + 0.02f)) / 0.3f));
                                    break;
                                }
                                currentY -= 0.05f;
                            }
                            
                            if (filters[i]->hasParam2()) {
                                if (fy >= currentY && fy <= currentY + 0.03f * uiScale) {
                                    activeSlider = i * 3 + 2;
                                    filters[i]->param2 = std::max(0.0f, std::min(1.0f, 
                                        (fx - (listX + 0.02f)) / 0.3f));
                                    break;
                                }
                                currentY -= 0.05f;
                            }
                            
                            currentY -= 0.02f; // Extra spacing
                        }
                    }
                }
                else if (dist <= PIE_MENU_RADIUS) {
                    float angle = atan2(dy, dx);
                    if (angle < 0) angle += 2.0f * M_PI;
                    
                    int selected = static_cast<int>((angle / (2.0f * M_PI)) * filters.size());
                    if (selected >= 0 && selected < filters.size()) {
                        filters[selected]->enabled = !filters[selected]->enabled;
                        activeSlider = selected * 3; // Activate strength slider
                    }
                }
                return;
            }

        } else if (state == GLUT_UP) {
            //showPieMenu = false;
            activeSlider = -1;
            isSeeking = false;
        }
    }
}

void mouseMotion(int x, int y) {

    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    
    if (isSeeking && mouseLeftDown) {
        double seekPos = std::max(0.0, std::min(1.0, ((double)fx + 0.95f) / 1.9f));
        seekVideo(seekPos * videoDuration);
        return;
    }

    if (activeSlider == -1 || !showUI) return;

    //float fx = (float)x / windowWidth * 2.0f - 1.0f;
    int filterIdx = activeSlider / 3;
    int paramType = activeSlider % 3;
    
    float normalizedValue = std::max(0.0f, std::min(1.0f, (fx - 0.6f * uiScale) / (0.35f * uiScale)));
    
    switch (paramType) {
        case 0: filters[filterIdx]->strength = normalizedValue; break;
        case 1: filters[filterIdx]->param1 = normalizedValue; break;
        case 2: filters[filterIdx]->param2 = normalizedValue; break;
    }
    
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
            cap.open(playlist[currentVideoIndex]);
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

    float frameAspect = (float)rgb.cols / rgb.rows;
    float windowAspect = (float)windowWidth / windowHeight;

    int drawWidth, drawHeight;
    int offsetX = 0, offsetY = 0;

    // Calculate dimensions to maintain aspect ratio while fitting in window
    if (frameAspect > windowAspect) {
        // Video is wider than window - fit to width
        drawWidth = windowWidth;
        drawHeight = static_cast<int>(windowWidth / frameAspect);
        offsetY = (windowHeight - drawHeight) / 2;
    } else {
        // Video is taller than window - fit to height
        drawHeight = windowHeight;
        drawWidth = static_cast<int>(windowHeight * frameAspect);
        offsetX = (windowWidth - drawWidth) / 2;
    }

    // Set viewport to the calculated dimensions
    glViewport(offsetX, offsetY, drawWidth, drawHeight);

    // Draw the video frame
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, rgb.cols, 0, rgb.rows);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2i(0, 0);
    glDrawPixels(rgb.cols, rgb.rows, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);

    // Reset viewport for UI
    glViewport(0, 0, windowWidth, windowHeight);
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
                displayVideoFrame(frame);
            } else {
                if (isLooping) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                } else {
                    currentVideoIndex = (currentVideoIndex + 1) % playlist.size();
                    cap.open(playlist[currentVideoIndex]);
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

        // Filter controls
        for (size_t i = 0; i < filters.size(); ++i) {
            float bx = 0.6f * uiScale;
            float by = -0.8f * uiScale + 0.15f * uiScale * i;
            
            // Filter toggle button
            std::string label = std::to_string(i+1) + ". " + filters[i]->name() + (filters[i]->enabled ? " [ON]" : " [OFF]");
            drawButton(bx, by, 0.35f * uiScale, 0.08f * uiScale, label, false, filters[i]->enabled);

            // Sliders for active filters
            if (filters[i]->enabled) {
                float sy = by + 0.1f * uiScale;
                
                if (filters[i]->hasStrength()) {
                    drawSlider(bx, sy, 0.35f * uiScale, 0.03f * uiScale, filters[i]->strength, "Strength");
                    sy += 0.07f * uiScale;
                }
                
                if (filters[i]->hasParam1()) {
                    drawSlider(bx, sy, 0.35f * uiScale, 0.03f * uiScale, filters[i]->param1, filters[i]->param1Name());
                    sy += 0.07f * uiScale;
                }
                
                if (filters[i]->hasParam2()) {
                    drawSlider(bx, sy, 0.35f * uiScale, 0.03f * uiScale, filters[i]->param2, filters[i]->param2Name());
                }
            }
        }

        // Help text
        drawText(-0.95f, 0.95f, "Space: Play/Pause | L: Loop | N: Next | F: Fullscreen | U: Toggle UI | 1-9: Filters");
        drawText(-0.95f, 0.9f, "-/=: Adjust slider | </>: Seek 5s | [/]: Speed | Click progress bar to seek");
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
    cap.open(playlist[0]);
    if (!cap.isOpened()) {
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
    glutKeyboardFunc(keyboard);

    // Main loop
    glutMainLoop();

    return 0;
}