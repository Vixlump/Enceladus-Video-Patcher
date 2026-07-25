#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/ocl.hpp>
#include <GL/freeglut.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
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
#include <cstring>

// Global variables
cv::Mat latestFrame;
bool frameAvailable = false;
bool isPaused = false;
bool isLooping = false;
bool showUI = true;
int currentVideoIndex = 0;
std::vector<std::string> playlist;

// Performance: HW decode (VAAPI/etc via FFmpeg) + OpenGL textured display
bool useHwDecode = true;
bool useOpenCL = true;
std::string hwDecodeStatus = "n/a";
GLuint videoTexId = 0;
int videoTexW = 0;
int videoTexH = 0;

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
// Encodes activeSlider as filterIdx * AE_SLIDER_STRIDE + paramType (0=strength .. 17=param17)
const int AE_SLIDER_STRIDE = 18;
float aePanelScroll = 0.0f; // scroll offset for tall Active Effects lists
bool isSeeking = false;
double videoDuration = 0;
double currentVideoTime = 0;
double lastFrameTime = 0;
bool mouseLeftDown = false;
float fastForwardSpeed = 1.0f;
const float MAX_FF_SPEED = 16.0f;

// Video-window drag placement
enum class VideoDragMode { Idle, Move, ResizeW, ResizeE, ResizeN, ResizeS, ResizeNW, ResizeNE, ResizeSW, ResizeSE };
VideoDragMode videoDragMode = VideoDragMode::Idle;
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
// can use the full left column for filter sliders. Aspect formats sit under the
// Filters list on the right so they stay clear of transport / timeline.
const float GUIDE_PANEL_X = -0.45f;
const float GUIDE_PANEL_Y = 0.58f;
const float PLACE_PANEL_X = -0.45f;
const float PLACE_PANEL_Y = 0.10f;
const float FORMAT_PANEL_X = 0.48f;
const float FORMAT_PANEL_Y = 0.14f;
const float FORMAT_PANEL_H = 0.66f;
const float CAP_PANEL_X = -0.45f;
const float CAP_PANEL_Y = 0.92f;

// Cinema / delivery aspect presets (width / height)
struct AspectFormat {
    const char* name;
    float aspect; // w/h
};
static const AspectFormat kAspectFormats[] = {
    {"1:1 Square", 1.0f},
    {"5:4", 5.0f / 4.0f},
    {"4:3 Academy/SD", 4.0f / 3.0f},
    {"3:2 Classic", 3.0f / 2.0f},
    {"16:10", 16.0f / 10.0f},
    {"16:9 HD/UHD", 16.0f / 9.0f},
    {"1.85 Flat", 1.85f},
    {"1.90 IMAX Dig", 1.90f},
    {"2:1 Univisium", 2.0f},
    {"21:9 UltraWide", 21.0f / 9.0f},
    {"2.20 70mm", 2.20f},
    {"2.35 Scope", 2.35f},
    {"2.39 Anamorphic", 2.39f},
    {"2.40 Wide Scope", 2.40f},
    {"2.76 Ultra Pana", 2.76f},
    {"1.43 IMAX GT", 1.43f},
    {"4:5 Portrait", 4.0f / 5.0f},
    {"9:16 Stories", 9.0f / 16.0f},
    {"32:9 SuperUW", 32.0f / 9.0f},
};
static const int kAspectFormatCount = static_cast<int>(sizeof(kAspectFormats) / sizeof(kAspectFormats[0]));
int formatPresetIndex = -1; // -1 = native (off)
bool formatUseCustom = false; // true when Aspect slider overrides preset
int formatMode = 1;         // 0 = letterbox/pillarbox, 1 = center crop
bool formatShowBorder = true; // gold aspect guide on video window
int formatListScroll = 0;
const int FORMAT_LIST_ROWS = 4;
const float FORMAT_ASPECT_MIN = 0.40f;
const float FORMAT_ASPECT_MAX = 3.60f;
float formatAspectNorm = (16.0f / 9.0f - FORMAT_ASPECT_MIN) / (FORMAT_ASPECT_MAX - FORMAT_ASPECT_MIN);
int activeFormatSlider = -1; // 0=aspect, 1=forceW, 2=forceH

// Forced output resolution (-1 = source/default, -2 = custom W/H, else preset index)
struct ForceResPreset {
    const char* name;
    int w;
    int h;
};
static const ForceResPreset kForceResPresets[] = {
    {"480p", 854, 480},
    {"720p", 1280, 720},
    {"1080p", 1920, 1080},
    {"1440p", 2560, 1440},
    {"4K", 3840, 2160},
    {"1K sq", 1080, 1080},
};
static const int kForceResCount = static_cast<int>(sizeof(kForceResPresets) / sizeof(kForceResPresets[0]));
int forceResPreset = -1; // -1 default, -2 custom, >=0 preset
float forceResWNorm = (1920.0f - 320.0f) / (3840.0f - 320.0f);
float forceResHNorm = (1080.0f - 240.0f) / (2160.0f - 240.0f);

// Window-capture tuning (normalized crop fractions + performance)
float capCropL = 0.0f;
float capCropR = 0.0f;
float capCropT = 0.0f;
float capCropB = 0.0f;
float capDownscale = 1.0f; // 0.25..1.0 of captured resolution before filters
float capFpsNorm = 0.55f;  // maps to ~8–60 fps
int activeCapSlider = -1;
double lastCapGrabTime = 0.0;

// Source picker menus (file browser / webcam list / window capture / queue)
enum class SourceMenuMode { Closed, Files, Cameras, Windows, Queue };
SourceMenuMode sourceMenuMode = SourceMenuMode::Closed;
std::string appRootPath; // directory containing the running executable
std::string browserPath;
std::vector<std::pair<std::string, bool>> browserEntries; // name, isDirectory
int browserScroll = 0;
bool browserQueueMode = true; // true = add to queue (stay open); false = play now
int queueScroll = 0;
struct CameraDevice {
    int index;
    std::string label;
};
std::vector<CameraDevice> cameraDevices;
int cameraScroll = 0;
struct CaptureWindow {
    unsigned long xid = 0;
    std::string title;
    int width = 0;
    int height = 0;
    bool isScreen = false;
};
std::vector<CaptureWindow> captureWindows;
int windowScroll = 0;
Display* xDisplay = nullptr;
Window captureXid = 0;
bool windowCaptureActive = false;
std::string captureWindowTitle;
const int SOURCE_MENU_ROWS = 12;
namespace fs = std::filesystem;

// Scrollable filter toggle list (right side of control panel)
int filterListScroll = 0;
const float FILT_PANEL_X = 0.48f;
const float FILT_PANEL_W = 0.48f;
const float FILT_PANEL_TOP = 0.82f;
// Leave room under Filters for Aspect Formats (above view/transport).
const float FILT_PANEL_BOTTOM = 0.20f;
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
    virtual bool hasParam3() const { return false; }
    virtual bool hasParam4() const { return false; }
    virtual bool hasParam5() const { return false; }
    virtual bool hasParam6() const { return false; }
    virtual bool hasParam7() const { return false; }
    virtual bool hasParam8() const { return false; }
    virtual bool hasParam9() const { return false; }
    virtual bool hasParam10() const { return false; }
    virtual bool hasParam11() const { return false; }
    virtual bool hasParam12() const { return false; }
    virtual bool hasParam13() const { return false; }
    virtual bool hasParam14() const { return false; }
    virtual bool hasParam15() const { return false; }
    virtual std::string param1Name() const { return ""; }
    virtual std::string param2Name() const { return ""; }
    virtual std::string param3Name() const { return ""; }
    virtual std::string param4Name() const { return ""; }
    virtual std::string param5Name() const { return ""; }
    virtual std::string param6Name() const { return ""; }
    virtual std::string param7Name() const { return ""; }
    virtual std::string param8Name() const { return ""; }
    virtual std::string param9Name() const { return ""; }
    virtual std::string param10Name() const { return ""; }
    virtual std::string param11Name() const { return ""; }
    virtual std::string param12Name() const { return ""; }
    virtual std::string param13Name() const { return ""; }
    virtual std::string param14Name() const { return ""; }
    virtual std::string param15Name() const { return ""; }
    virtual bool hasParam16() const { return false; }
    virtual bool hasParam17() const { return false; }
    virtual std::string param16Name() const { return ""; }
    virtual std::string param17Name() const { return ""; }
    virtual bool hasReset() const { return false; }
    virtual void reset() {}
    // Optional calibration / tool buttons drawn under a filter's sliders
    virtual int toolButtonCount() const { return 0; }
    virtual std::string toolButtonName(int /*i*/) const { return ""; }
    virtual bool toolButtonActive(int /*i*/) const { return false; }
    virtual void toolButtonClick(int /*i*/) {}
    virtual std::string statusLine() const { return ""; }

    // Option pages keep dense filters (EnceladusVision) from dumping every slider at once
    int optionPage = 0;
    virtual int optionPageCount() const { return 1; }
    virtual std::string optionPageName(int /*i*/) const { return "Params"; }
    virtual bool showStrengthOnPage() const { return hasStrength(); }
    virtual bool showParamOnPage(int /*paramIndex1to17*/) const { return true; }
    
    bool enabled = false;
    float strength = 1.0f;
    float param1 = 0.5f;
    float param2 = 0.5f;
    float param3 = 0.5f;
    float param4 = 0.5f;
    float param5 = 0.0f;
    float param6 = 0.5f;
    float param7 = 0.5f;
    float param8 = 0.5f;
    float param9 = 0.5f;
    float param10 = 0.5f;
    float param11 = 0.35f;
    float param12 = 0.0f;
    float param13 = 0.5f;
    float param14 = 0.45f;
    float param15 = 0.35f;
    float param16 = 0.55f;
    float param17 = 0.45f;
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

// Detect large, persistent movers; enlarge only those; anaglyph with extra parallax.
class MotionPopAnaglyphFilter : public VideoFilter {
    struct Track {
        cv::Rect box;
        cv::Point2f center;
        double area = 0.0;
        int age = 0;      // consecutive frames matched
        int missed = 0;   // frames since last match
    };

    cv::Ptr<cv::BackgroundSubtractor> subtractor;
    std::vector<Track> tracks;

    static float rectIoU(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
        int uni = a.area() + b.area() - inter;
        return uni > 0 ? static_cast<float>(inter) / uni : 0.0f;
    }

    static void pasteScaledRoi(cv::Mat& dst, const cv::Mat& srcFrame, const cv::Rect& roi,
                               float scale, cv::Mat* maskOut) {
        if (roi.width < 4 || roi.height < 4) return;
        cv::Rect safe = roi & cv::Rect(0, 0, srcFrame.cols, srcFrame.rows);
        if (safe.empty()) return;

        cv::Mat patch = srcFrame(safe).clone();
        cv::Mat big;
        cv::resize(patch, big, cv::Size(), scale, scale, cv::INTER_LINEAR);

        int cx = safe.x + safe.width / 2;
        int cy = safe.y + safe.height / 2;
        int x0 = cx - big.cols / 2;
        int y0 = cy - big.rows / 2;

        for (int y = 0; y < big.rows; ++y) {
            int dy = y0 + y;
            if (dy < 0 || dy >= dst.rows) continue;
            for (int x = 0; x < big.cols; ++x) {
                int dx = x0 + x;
                if (dx < 0 || dx >= dst.cols) continue;
                dst.at<cv::Vec3b>(dy, dx) = big.at<cv::Vec3b>(y, x);
                if (maskOut)
                    maskOut->at<uchar>(dy, dx) = 255;
            }
        }
    }

public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;

        if (!subtractor)
            subtractor = cv::createBackgroundSubtractorMOG2(300, 20.0, true);

        cv::Mat motion;
        double learn = 0.03 + param1 * 0.06;
        subtractor->apply(frame, motion, learn);

        // Stricter cleanup so flicker / noise rarely becomes a candidate
        int thresh = static_cast<int>(80 + param1 * 100.0f);
        cv::threshold(motion, motion, thresh, 255, cv::THRESH_BINARY);
        cv::morphologyEx(motion, motion, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
        cv::morphologyEx(motion, motion, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {11, 11}));

        // Only large blobs are candidates (fraction of frame)
        const double frameArea = static_cast<double>(frame.rows) * frame.cols;
        const double minArea = frameArea * (0.012 + param1 * 0.028); // ~1.2%–4.0%
        const int minW = std::max(24, frame.cols / 16);
        const int minH = std::max(24, frame.rows / 16);
        // Persist long enough before popping (about 0.5–1.5s at 30fps)
        const int minAge = 18 + static_cast<int>(param1 * 30.0f); // 18–48 frames
        const int maxMissed = 10;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motion, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        struct Detection { cv::Rect box; cv::Point2f center; double area; };
        std::vector<Detection> detections;
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < minArea) continue;
            cv::Rect box = cv::boundingRect(c);
            if (box.width < minW || box.height < minH) continue;
            // Reject very thin noise strips
            float aspect = box.width / static_cast<float>(std::max(1, box.height));
            if (aspect > 8.0f || aspect < 1.0f / 8.0f) continue;
            detections.push_back({
                box,
                cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f),
                area
            });
        }

        // Match detections to tracks (greedy by IoU / center distance)
        std::vector<char> detUsed(detections.size(), 0);
        for (auto& tr : tracks) {
            int best = -1;
            float bestScore = 0.0f;
            for (size_t i = 0; i < detections.size(); ++i) {
                if (detUsed[i]) continue;
                float iou = rectIoU(tr.box, detections[i].box);
                float dist = cv::norm(tr.center - detections[i].center);
                float maxDist = 0.12f * std::max(frame.cols, frame.rows);
                if (iou < 0.15f && dist > maxDist) continue;
                float score = iou * 2.0f + (1.0f - std::min(1.0f, dist / maxDist));
                if (score > bestScore) {
                    bestScore = score;
                    best = static_cast<int>(i);
                }
            }
            if (best >= 0) {
                const auto& d = detections[best];
                tr.box = d.box;
                tr.center = d.center;
                tr.area = d.area;
                tr.age += 1;
                tr.missed = 0;
                detUsed[best] = 1;
            } else {
                tr.missed += 1;
            }
        }

        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [&](const Track& t) { return t.missed > maxMissed; }),
                     tracks.end());

        for (size_t i = 0; i < detections.size(); ++i) {
            if (detUsed[i]) continue;
            Track t;
            t.box = detections[i].box;
            t.center = detections[i].center;
            t.area = detections[i].area;
            t.age = 1;
            t.missed = 0;
            tracks.push_back(t);
        }

        // Build pop mask ONLY from mature, large tracks
        cv::Mat scene = frame.clone();
        cv::Mat popMask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        float popScale = 1.0f + param2 * 1.75f;

        for (const auto& tr : tracks) {
            if (tr.age < minAge) continue;
            if (tr.area < minArea) continue;

            cv::Rect box = tr.box;
            int pad = std::max(6, static_cast<int>(0.05 * std::max(box.width, box.height)));
            box.x = std::max(0, box.x - pad);
            box.y = std::max(0, box.y - pad);
            box.width = std::min(frame.cols - box.x, box.width + 2 * pad);
            box.height = std::min(frame.rows - box.y, box.height + 2 * pad);

            if (popScale > 1.05f)
                pasteScaledRoi(scene, frame, box, popScale, &popMask);
            else
                cv::rectangle(popMask, box, cv::Scalar(255), cv::FILLED);
        }

        // Soft disparity only where confirmed pops exist
        float bgShift = 0.5f + strength * 2.0f;
        float fgShift = 5.0f + strength * 16.0f;
        cv::Mat disp(frame.rows, frame.cols, CV_32FC1, cv::Scalar(bgShift));
        if (cv::countNonZero(popMask) > 0) {
            disp.setTo(fgShift, popMask);
            cv::GaussianBlur(disp, disp, cv::Size(31, 31), 0);
        }

        cv::Mat mapLx(frame.rows, frame.cols, CV_32FC1);
        cv::Mat mapLy(frame.rows, frame.cols, CV_32FC1);
        cv::Mat mapRx(frame.rows, frame.cols, CV_32FC1);
        cv::Mat mapRy(frame.rows, frame.cols, CV_32FC1);
        for (int y = 0; y < frame.rows; ++y) {
            const float* drow = disp.ptr<float>(y);
            float* lx = mapLx.ptr<float>(y);
            float* ly = mapLy.ptr<float>(y);
            float* rx = mapRx.ptr<float>(y);
            float* ry = mapRy.ptr<float>(y);
            for (int x = 0; x < frame.cols; ++x) {
                float d = drow[x];
                lx[x] = x + 0.5f * d;
                rx[x] = x - 0.5f * d;
                ly[x] = static_cast<float>(y);
                ry[x] = static_cast<float>(y);
            }
        }

        cv::Mat leftView, rightView;
        cv::remap(scene, leftView, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::remap(scene, rightView, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::Mat output;
        std::vector<cv::Mat> lch, rch;
        cv::split(leftView, lch);
        cv::split(rightView, rch);
        std::vector<cv::Mat> outCh = {rch[0], rch[1], lch[2]};
        cv::merge(outCh, output);

        if (strength < 0.999f)
            cv::addWeighted(frame, 1.0f - strength, output, strength, 0, output);

        return output;
    }

    std::string name() const override { return "Motion Pop 3D"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    std::string param1Name() const override { return "Strictness"; }
    std::string param2Name() const override { return "Pop Scale"; }
};

// EnceladusVision: soft wave extrusion of tracked regions on an otherwise flat,
// in-frame video plane. Orbit sliders tip the view to inspect 3D mounds.
class EnceladusVisionFilter : public VideoFilter {
    struct Track {
        cv::Rect box;
        cv::Point2f center;
        cv::Point2f vel{0.f, 0.f}; // smoothed px/frame
        double area = 0.0;
        float halfW = 40.0f;
        float halfH = 40.0f;
        float sizeNorm = 0.5f; // 0..1 relative to frame
        float blend = 0.0f;    // smoothed wave presence 0..1
        int age = 0;
        int missed = 0;
    };

    cv::Ptr<cv::BackgroundSubtractor> subtractor;
    std::vector<Track> tracks;
    cv::Mat lastMotion; // for calibration overlay + Mot3D height
    cv::Mat motionHeight; // smoothed motion field 0..1
    bool showTracks = false;
    bool showMotion = false;
    bool showHeight = false;
    bool showEdges = false;
    bool anaglyphOn = true;
    bool motion3DOn = false;
    static constexpr int kGrid = 80;

    struct EdgeTrack {
        float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        float nx = 0, ny = 1;
        float len = 0;
        float blend = 0.0f;
        int age = 0;
        int missed = 0;
    };
    std::vector<EdgeTrack> edgeTracks;

    struct Vert {
        float x, y, z, u, v;
        float sx, sy, iw;
    };

    static float rectIoU(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
        int uni = a.area() + b.area() - inter;
        return uni > 0 ? static_cast<float>(inter) / uni : 0.0f;
    }

    static cv::Vec3b sampleBilinear(const cv::Mat& img, float u, float v) {
        float x = u * (img.cols - 1);
        float y = v * (img.rows - 1);
        int x0 = std::max(0, std::min(img.cols - 1, static_cast<int>(std::floor(x))));
        int y0 = std::max(0, std::min(img.rows - 1, static_cast<int>(std::floor(y))));
        int x1 = std::max(0, std::min(img.cols - 1, x0 + 1));
        int y1 = std::max(0, std::min(img.rows - 1, y0 + 1));
        float tx = x - x0, ty = y - y0;
        cv::Vec3b c00 = img.at<cv::Vec3b>(y0, x0);
        cv::Vec3b c10 = img.at<cv::Vec3b>(y0, x1);
        cv::Vec3b c01 = img.at<cv::Vec3b>(y1, x0);
        cv::Vec3b c11 = img.at<cv::Vec3b>(y1, x1);
        cv::Vec3b out;
        for (int i = 0; i < 3; ++i) {
            float a = c00[i] * (1 - tx) + c10[i] * tx;
            float b = c01[i] * (1 - tx) + c11[i] * tx;
            out[i] = static_cast<uchar>(a * (1 - ty) + b * ty);
        }
        return out;
    }

    static void drawTriangle(cv::Mat& dst, cv::Mat& zbuf, const Vert& a, const Vert& b, const Vert& c,
                             const cv::Mat& tex) {
        int minX = std::max(0, static_cast<int>(std::floor(std::min({a.sx, b.sx, c.sx}))));
        int maxX = std::min(dst.cols - 1, static_cast<int>(std::ceil(std::max({a.sx, b.sx, c.sx}))));
        int minY = std::max(0, static_cast<int>(std::floor(std::min({a.sy, b.sy, c.sy}))));
        int maxY = std::min(dst.rows - 1, static_cast<int>(std::ceil(std::max({a.sy, b.sy, c.sy}))));
        float area = (b.sx - a.sx) * (c.sy - a.sy) - (c.sx - a.sx) * (b.sy - a.sy);
        if (std::fabs(area) < 1e-4f) return;
        float invArea = 1.0f / area;

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                float w0 = ((b.sx - x) * (c.sy - y) - (c.sx - x) * (b.sy - y)) * invArea;
                float w1 = ((c.sx - x) * (a.sy - y) - (a.sx - x) * (c.sy - y)) * invArea;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

                float iw = w0 * a.iw + w1 * b.iw + w2 * c.iw;
                float* zpx = zbuf.ptr<float>(y) + x;
                if (iw <= *zpx) continue;
                *zpx = iw;

                float u = (w0 * a.u * a.iw + w1 * b.u * b.iw + w2 * c.u * c.iw) / iw;
                float v = (w0 * a.v * a.iw + w1 * b.v * b.iw + w2 * c.v * c.iw) / iw;
                u = std::max(0.0f, std::min(1.0f, u));
                v = std::max(0.0f, std::min(1.0f, v));
                dst.at<cv::Vec3b>(y, x) = sampleBilinear(tex, u, v);
            }
        }
    }

    void updateTracks(const cv::Mat& frame) {
        if (!subtractor)
            subtractor = cv::createBackgroundSubtractorMOG2(300, 20.0, true);

        // param1 = Strictness (same idea as Motion Pop)
        cv::Mat motion;
        double learn = 0.03 + param1 * 0.06;
        subtractor->apply(frame, motion, learn);
        int thresh = static_cast<int>(80 + param1 * 100.0f);
        cv::threshold(motion, motion, thresh, 255, cv::THRESH_BINARY);
        cv::morphologyEx(motion, motion, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
        cv::morphologyEx(motion, motion, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {11, 11}));
        lastMotion = motion;

        const double frameArea = static_cast<double>(frame.rows) * frame.cols;
        // Slightly looser than Motion Pop so extrusions show up more readily
        const double minArea = frameArea * (0.006 + param1 * 0.022);
        const int minW = std::max(16, frame.cols / 22);
        const int minH = std::max(16, frame.rows / 22);
        const int maxMissed = 14;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motion, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        struct Detection { cv::Rect box; cv::Point2f center; double area; float halfW; float halfH; };
        std::vector<Detection> detections;
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < minArea) continue;
            cv::Rect box = cv::boundingRect(c);
            if (box.width < minW || box.height < minH) continue;
            float aspect = box.width / static_cast<float>(std::max(1, box.height));
            if (aspect > 8.0f || aspect < 1.0f / 8.0f) continue;
            detections.push_back({
                box,
                {box.x + box.width * 0.5f, box.y + box.height * 0.5f},
                area,
                box.width * 0.5f,
                box.height * 0.5f
            });
        }

        std::vector<char> used(detections.size(), 0);
        for (auto& tr : tracks) {
            int best = -1;
            float bestScore = 0.0f;
            for (size_t i = 0; i < detections.size(); ++i) {
                if (used[i]) continue;
                float iou = rectIoU(tr.box, detections[i].box);
                float dist = cv::norm(tr.center - detections[i].center);
                float maxDist = 0.12f * std::max(frame.cols, frame.rows);
                if (iou < 0.12f && dist > maxDist) continue;
                float score = iou * 2.0f + (1.0f - std::min(1.0f, dist / maxDist));
                if (score > bestScore) { bestScore = score; best = static_cast<int>(i); }
            }
            if (best >= 0) {
                const auto& d = detections[best];
                cv::Point2f rawVel = d.center - tr.center;
                tr.vel = tr.vel * 0.72f + rawVel * 0.28f;
                tr.box = d.box;
                tr.center = d.center;
                tr.area = d.area;
                tr.halfW = d.halfW;
                tr.halfH = d.halfH;
                float rel = static_cast<float>(std::sqrt(d.area / std::max(1.0, frameArea)));
                tr.sizeNorm = std::max(0.0f, std::min(1.0f, rel / 0.42f));
                tr.age += 1;
                tr.missed = 0;
                used[best] = 1;
            } else {
                tr.missed += 1;
                tr.vel *= 0.85f;
            }
        }

        // param11 = Wave Lerp: 0 = instant, 1 = very long ease (many frames)
        // param15 = Anticipate: earlier lead-in + snappier rise / slower fall
        const int minAgeBlend = 4 + static_cast<int>((0.25f + param7 * 0.75f) * (8 + param1 * 20.0f));
        const float lerpSlider = std::max(0.0f, std::min(1.0f, param11));
        const float antic = std::max(0.0f, std::min(1.0f, param15));
        // High Wave Lerp → tiny alpha (can take ~2–4s at 30fps to settle)
        const float blendAlpha = (lerpSlider <= 0.001f)
            ? 1.0f
            : (0.0035f + std::pow(1.0f - lerpSlider, 2.8f) * 0.52f);
        const float riseAlpha = std::min(1.0f, blendAlpha * (1.0f + antic * 2.2f));
        const float fallAlpha = std::max(0.002f, blendAlpha * (1.0f - antic * 0.55f));
        // Anticipation: start pop earlier relative to Hold/Strictness maturity
        const float startFrac = 0.50f - antic * 0.42f; // 0.50 → ~0.08
        const int startAge = std::max(1, static_cast<int>(minAgeBlend * startFrac));
        const int maxMissedFade = maxMissed + static_cast<int>(lerpSlider * 36.0f);

        for (auto& tr : tracks) {
            float target = 0.0f;
            if (tr.missed == 0 && tr.age >= startAge) {
                // Softer maturity ramp; anticipation stretches the early part
                float maturity = std::min(1.0f, tr.age / static_cast<float>(std::max(1, minAgeBlend)));
                float shaped = std::pow(maturity, 1.0f - antic * 0.45f);
                target = shaped;
            }
            float a = (target > tr.blend) ? riseAlpha : fallAlpha;
            if (a >= 0.999f)
                tr.blend = target;
            else
                tr.blend += (target - tr.blend) * a;
            if (tr.blend < 0.0f) tr.blend = 0.0f;
            if (tr.blend > 1.0f) tr.blend = 1.0f;
        }

        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [&](const Track& t) {
                                        if (t.missed <= maxMissedFade) return false;
                                        if (lerpSlider > 0.001f && t.blend > 0.015f) return false;
                                        return true;
                                    }),
                     tracks.end());
        for (size_t i = 0; i < detections.size(); ++i) {
            if (used[i]) continue;
            Track t;
            t.box = detections[i].box;
            t.center = detections[i].center;
            t.vel = {0.f, 0.f};
            t.area = detections[i].area;
            t.halfW = detections[i].halfW;
            t.halfH = detections[i].halfH;
            float rel = static_cast<float>(std::sqrt(detections[i].area / std::max(1.0, frameArea)));
            t.sizeNorm = std::max(0.0f, std::min(1.0f, rel / 0.42f));
            t.blend = 0.0f;
            t.age = 1;
            t.missed = 0;
            tracks.push_back(t);
        }
    }

    static float segAlignScore(float ax0, float ay0, float ax1, float ay1,
                               float bx0, float by0, float bx1, float by1,
                               float diag) {
        float amx = 0.5f * (ax0 + ax1), amy = 0.5f * (ay0 + ay1);
        float bmx = 0.5f * (bx0 + bx1), bmy = 0.5f * (by0 + by1);
        float dmid = std::sqrt((amx - bmx) * (amx - bmx) + (amy - bmy) * (amy - bmy));
        float adx = ax1 - ax0, ady = ay1 - ay0;
        float bdx = bx1 - bx0, bdy = by1 - by0;
        float al = std::sqrt(adx * adx + ady * ady);
        float bl = std::sqrt(bdx * bdx + bdy * bdy);
        if (al < 1e-3f || bl < 1e-3f) return 0.0f;
        float align = std::fabs((adx * bdx + ady * bdy) / (al * bl));
        float midOk = 1.0f - std::min(1.0f, dmid / (0.08f * diag));
        return align * midOk;
    }

    // Straight-ish edges with persistence: only long-lived tracks drive the split.
    void updateEdges(const cv::Mat& frame) {
        if (param12 < 0.02f) {
            for (auto& e : edgeTracks) {
                e.missed += 1;
                e.blend *= 0.85f;
            }
            edgeTracks.erase(std::remove_if(edgeTracks.begin(), edgeTracks.end(),
                                            [](const EdgeTrack& e) {
                                                return e.missed > 20 && e.blend < 0.02f;
                                            }),
                             edgeTracks.end());
            return;
        }

        cv::Mat gray, blur, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(7, 7), 0);

        // param13 = Edge Sense (kept conservative — high end still picky)
        const float sense = std::max(0.0f, std::min(1.0f, param13));
        const double cannyLo = 70.0 + (1.0 - sense) * 90.0;
        const double cannyHi = 160.0 + (1.0 - sense) * 90.0;
        cv::Canny(blur, edges, cannyLo, cannyHi);

        const int diag = std::max(frame.cols, frame.rows);
        const double minLen = std::max(40.0, diag * (0.20 - sense * 0.07));
        const int houghThresh = std::max(45, static_cast<int>(95 - sense * 35.0));
        const double maxGap = 4.0 + sense * 10.0;

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0, houghThresh, minLen, maxGap);

        struct Det { float x0, y0, x1, y1, nx, ny, len; };
        std::vector<Det> dets;
        for (const auto& L : lines) {
            float x0 = static_cast<float>(L[0]), y0 = static_cast<float>(L[1]);
            float x1 = static_cast<float>(L[2]), y1 = static_cast<float>(L[3]);
            float dx = x1 - x0, dy = y1 - y0;
            float len = std::sqrt(dx * dx + dy * dy);
            if (len < minLen * 0.9f) continue;
            float inv = 1.0f / len;
            dets.push_back({x0, y0, x1, y1, -dy * inv, dx * inv, len});
        }
        std::sort(dets.begin(), dets.end(),
                  [](const Det& a, const Det& b) { return a.len > b.len; });

        // Dedup detections
        std::vector<Det> unique;
        const int maxDet = 10;
        for (const auto& d : dets) {
            if (static_cast<int>(unique.size()) >= maxDet) break;
            bool dup = false;
            for (const auto& u : unique) {
                if (segAlignScore(d.x0, d.y0, d.x1, d.y1, u.x0, u.y0, u.x1, u.y1,
                                  static_cast<float>(diag)) > 0.78f) {
                    dup = true; break;
                }
            }
            if (!dup) unique.push_back(d);
        }

        std::vector<char> used(unique.size(), 0);
        for (auto& tr : edgeTracks) {
            int best = -1;
            float bestScore = 0.55f;
            for (size_t i = 0; i < unique.size(); ++i) {
                if (used[i]) continue;
                float sc = segAlignScore(tr.x0, tr.y0, tr.x1, tr.y1,
                                         unique[i].x0, unique[i].y0, unique[i].x1, unique[i].y1,
                                         static_cast<float>(diag));
                if (sc > bestScore) { bestScore = sc; best = static_cast<int>(i); }
            }
            if (best >= 0) {
                const auto& d = unique[best];
                // Smooth geometry so the tear doesn't jitter
                tr.x0 = tr.x0 * 0.72f + d.x0 * 0.28f;
                tr.y0 = tr.y0 * 0.72f + d.y0 * 0.28f;
                tr.x1 = tr.x1 * 0.72f + d.x1 * 0.28f;
                tr.y1 = tr.y1 * 0.72f + d.y1 * 0.28f;
                float dx = tr.x1 - tr.x0, dy = tr.y1 - tr.y0;
                tr.len = std::sqrt(dx * dx + dy * dy);
                if (tr.len > 1e-3f) {
                    float inv = 1.0f / tr.len;
                    float nnx = -dy * inv, nny = dx * inv;
                    // Keep normal continuous (don't flip each frame)
                    if (nnx * tr.nx + nny * tr.ny < 0.0f) { nnx = -nnx; nny = -nny; }
                    tr.nx = tr.nx * 0.7f + nnx * 0.3f;
                    tr.ny = tr.ny * 0.7f + nny * 0.3f;
                    float nlen = std::sqrt(tr.nx * tr.nx + tr.ny * tr.ny);
                    if (nlen > 1e-4f) { tr.nx /= nlen; tr.ny /= nlen; }
                }
                tr.age += 1;
                tr.missed = 0;
                used[best] = 1;
            } else {
                tr.missed += 1;
            }
        }

        // param14 = Edge Lerp
        const float edgeLerp = std::max(0.0f, std::min(1.0f, param14));
        const float edgeAlpha = (edgeLerp <= 0.001f)
            ? 1.0f
            : (0.004f + std::pow(1.0f - edgeLerp, 2.6f) * 0.48f);
        const int minEdgeAge = 8 + static_cast<int>((1.0f - sense) * 18.0f + edgeLerp * 20.0f);
        const int maxEdgeMissed = 10 + static_cast<int>(edgeLerp * 28.0f);

        for (auto& tr : edgeTracks) {
            float target = 0.0f;
            if (tr.missed == 0 && tr.age >= minEdgeAge)
                target = std::min(1.0f, (tr.age - minEdgeAge + 1) / static_cast<float>(std::max(6, minEdgeAge / 2)));
            if (edgeAlpha >= 0.999f)
                tr.blend = target;
            else
                tr.blend += (target - tr.blend) * edgeAlpha;
            tr.blend = std::max(0.0f, std::min(1.0f, tr.blend));
        }

        edgeTracks.erase(std::remove_if(edgeTracks.begin(), edgeTracks.end(),
                                        [&](const EdgeTrack& e) {
                                            if (e.missed <= maxEdgeMissed) return false;
                                            if (edgeLerp > 0.001f && e.blend > 0.015f) return false;
                                            return true;
                                        }),
                         edgeTracks.end());

        for (size_t i = 0; i < unique.size(); ++i) {
            if (used[i]) continue;
            if (static_cast<int>(edgeTracks.size()) >= 8) break;
            EdgeTrack t;
            t.x0 = unique[i].x0; t.y0 = unique[i].y0;
            t.x1 = unique[i].x1; t.y1 = unique[i].y1;
            t.nx = unique[i].nx; t.ny = unique[i].ny;
            t.len = unique[i].len;
            t.blend = 0.0f;
            t.age = 1;
            t.missed = 0;
            edgeTracks.push_back(t);
        }
    }

    struct EdgeSample {
        float h = 0.0f;
        float ox = 0.0f;
        float oy = 0.0f;
    };

    struct WaveSample {
        float h = 0.0f;
        float ox = 0.0f;
        float oy = 0.0f;
    };

    static float popScaleCurve(float t) {
        t = std::max(0.0f, std::min(1.0f, t));
        const float shaped = std::pow(t, 2.35f);
        return 0.012f + shaped * 2.98f;
    }

    // Push mesh sides apart across persistent, lerped edge tracks.
    EdgeSample sampleEdgeSplit(float u, float v, int frameW, int frameH) const {
        EdgeSample out;
        if (param12 < 0.02f || edgeTracks.empty()) return out;

        const float px = u * std::max(1, frameW - 1);
        const float py = v * std::max(1, frameH - 1);
        const float diag = std::sqrt(static_cast<float>(frameW * frameW + frameH * frameH));
        const float band = (0.018f + param12 * 0.09f) * diag;
        const float pop = popScaleCurve(param6);
        const float sepAmt = param12 * (0.05f + strength * 0.28f) * pop;

        float bestDist = 1e9f;
        float bestNx = 0.0f, bestNy = 0.0f;
        float bestSign = 1.0f;
        float bestBlend = 0.0f;

        for (const auto& e : edgeTracks) {
            if (e.blend < 0.02f) continue;
            float dx = e.x1 - e.x0;
            float dy = e.y1 - e.y0;
            float len2 = dx * dx + dy * dy;
            float t = (len2 > 1e-4f) ? ((px - e.x0) * dx + (py - e.y0) * dy) / len2 : 0.0f;
            t = std::max(0.0f, std::min(1.0f, t));
            float qx = e.x0 + t * dx;
            float qy = e.y0 + t * dy;
            float ex = px - qx;
            float ey = py - qy;
            float dist = std::sqrt(ex * ex + ey * ey);
            if (dist >= bestDist) continue;
            bestDist = dist;
            bestBlend = e.blend;
            float side = ex * e.nx + ey * e.ny;
            bestSign = (side >= 0.0f) ? 1.0f : -1.0f;
            if (dist > 0.75f) {
                bestNx = ex / dist;
                bestNy = ey / dist;
            } else {
                bestNx = e.nx * bestSign;
                bestNy = e.ny * bestSign;
                bestSign = 1.0f;
            }
        }

        if (bestDist > band * 1.05f || bestBlend < 0.02f) return out;

        float fall = 1.0f - bestDist / std::max(1e-3f, band);
        fall = fall * fall;
        float push = bestSign * sepAmt * fall * bestBlend;
        out.ox = bestNx * push;
        out.oy = -bestNy * push;

        float ridgeBand = band * 0.28f;
        float ridge = 1.0f - std::min(1.0f, bestDist / std::max(1e-3f, ridgeBand));
        ridge = ridge * ridge;
        out.h = ridge * param12 * (0.025f + strength * 0.14f) * pop * 0.55f * bestBlend;
        return out;
    }

    void updateMotionField() {
        if (!motion3DOn || lastMotion.empty()) {
            motionHeight.release();
            return;
        }
        cv::Mat f;
        lastMotion.convertTo(f, CV_32FC1, 1.0 / 255.0);
        int k = 3 + static_cast<int>(std::max(0.0f, std::min(1.0f, param17)) * 14.0f) * 2; // odd
        if (k % 2 == 0) ++k;
        cv::GaussianBlur(f, motionHeight, cv::Size(k, k), 0);
    }

    // Mot3D: extrude the mesh from the motion mask / field.
    WaveSample sampleMotion3D(float u, float v, int frameW, int frameH) const {
        WaveSample out;
        if (!motion3DOn || motionHeight.empty()) return out;
        const float depth = std::max(0.0f, std::min(1.0f, param16));
        if (depth < 0.01f) return out;

        float x = u * (motionHeight.cols - 1);
        float y = v * (motionHeight.rows - 1);
        int x0 = std::max(0, std::min(motionHeight.cols - 1, static_cast<int>(std::floor(x))));
        int y0 = std::max(0, std::min(motionHeight.rows - 1, static_cast<int>(std::floor(y))));
        int x1 = std::max(0, std::min(motionHeight.cols - 1, x0 + 1));
        int y1 = std::max(0, std::min(motionHeight.rows - 1, y0 + 1));
        float tx = x - x0, ty = y - y0;
        float m00 = motionHeight.at<float>(y0, x0);
        float m10 = motionHeight.at<float>(y0, x1);
        float m01 = motionHeight.at<float>(y1, x0);
        float m11 = motionHeight.at<float>(y1, x1);
        float m = (m00 * (1 - tx) + m10 * tx) * (1 - ty) + (m01 * (1 - tx) + m11 * tx) * ty;

        const float pop = popScaleCurve(param6);
        out.h = m * depth * (0.06f + strength * 0.85f) * pop;

        // Tip along local motion gradient (cheap finite difference)
        float gx = 0.0f, gy = 0.0f;
        if (x0 > 0 && x1 < motionHeight.cols - 1) {
            gx = motionHeight.at<float>(y0, std::min(motionHeight.cols - 1, x0 + 1))
               - motionHeight.at<float>(y0, std::max(0, x0 - 1));
        }
        if (y0 > 0 && y1 < motionHeight.rows - 1) {
            gy = motionHeight.at<float>(std::min(motionHeight.rows - 1, y0 + 1), x0)
               - motionHeight.at<float>(std::max(0, y0 - 1), x0);
        }
        float gLen = std::sqrt(gx * gx + gy * gy);
        if (gLen > 1e-4f && m > 0.08f) {
            float lean = depth * m * out.h * 0.55f;
            out.ox = (gx / gLen) * lean;
            out.oy = -(gy / gLen) * lean; // image Y down → world Y up
        }
        (void)frameW; (void)frameH;
        return out;
    }

    // Raised-cosine mound; depth scales with track size; tips toward motion.
    WaveSample sampleWave(float u, float v, int frameW, int frameH) const {
        const float radiusMul = 0.7f + param3 * 1.5f;
        const float softPow = 0.35f + param2 * 1.8f;
        const float pop = popScaleCurve(param6);
        const float baseAmp = (0.08f + strength * 0.92f) * pop;
        const float sizeAmt = param9;   // Size Depth
        const float pointAmt = param10; // Motion Point
        const float diag = std::sqrt(static_cast<float>(frameW * frameW + frameH * frameH));

        WaveSample best;
        float bestW = 0.0f;
        for (const auto& tr : tracks) {
            if (tr.blend < 0.01f) continue;
            float cu = tr.center.x / std::max(1, frameW - 1);
            float cv = tr.center.y / std::max(1, frameH - 1);
            float rx = std::max(0.03f, (tr.halfW / std::max(1, frameW - 1)) * radiusMul);
            float ry = std::max(0.03f, (tr.halfH / std::max(1, frameH - 1)) * radiusMul);
            float nx = (u - cu) / rx;
            float ny = (v - cv) / ry;
            float r2 = nx * nx + ny * ny;
            if (r2 >= 1.0f) continue;
            float t = std::sqrt(r2);
            float wave = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * t));
            wave = std::pow(std::max(0.0f, wave), softPow);

            // Larger tracks push farther out of the screen; small tracks stay subtle
            float sizeDepth = (1.0f - 0.75f * sizeAmt) + tr.sizeNorm * (1.55f * sizeAmt);
            // blend = smoothed presence (Wave Lerp); replaces hard maturity jump
            float h = wave * tr.blend * baseAmp * sizeDepth;

            // Tip / point the mound along smoothed velocity (leading edge rises ahead)
            float speed = std::sqrt(tr.vel.x * tr.vel.x + tr.vel.y * tr.vel.y);
            float lean = 0.0f, ldx = 0.0f, ldy = 0.0f;
            if (speed > 0.15f && pointAmt > 0.01f) {
                float invSp = 1.0f / speed;
                float dirU = (tr.vel.x * invSp); // +x right in image
                float dirV = (tr.vel.y * invSp); // +y down in image
                // Along-motion coordinate inside the ellipse (-1..1-ish)
                float along = nx * dirU * (rx / std::max(rx, ry)) + ny * dirV * (ry / std::max(rx, ry));
                float tip = 0.5f + 0.5f * along; // higher on leading side
                h *= (1.0f - 0.35f * pointAmt) + tip * (0.7f * pointAmt);
                lean = pointAmt * wave * tr.blend * std::min(1.0f, speed / (0.02f * diag)) * h;
                // World X follows image X; world Y is up so flip image V
                ldx = dirU * lean * 1.15f;
                ldy = -dirV * lean * 1.15f;
            }

            if (wave * tr.blend >= bestW) {
                bestW = wave * tr.blend;
                best.h = h;
                best.ox = ldx;
                best.oy = ldy;
            } else if (h > best.h) {
                best.h = h;
                best.ox = ldx;
                best.oy = ldy;
            }
        }
        return best;
    }

    float heightAt(float u, float v, int frameW, int frameH) const {
        return sampleWave(u, v, frameW, frameH).h
             + sampleEdgeSplit(u, v, frameW, frameH).h
             + sampleMotion3D(u, v, frameW, frameH).h;
    }

    cv::Mat buildHeightMap(int rows, int cols) const {
        const int gw = 96, gh = std::max(1, static_cast<int>(96.0f * rows / std::max(1, cols)));
        cv::Mat small(gh, gw, CV_32FC1);
        for (int y = 0; y < gh; ++y) {
            float v = y / static_cast<float>(std::max(1, gh - 1));
            float* row = small.ptr<float>(y);
            for (int x = 0; x < gw; ++x) {
                float u = x / static_cast<float>(std::max(1, gw - 1));
                row[x] = heightAt(u, v, cols, rows);
            }
        }
        cv::Mat full;
        cv::resize(small, full, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
        return full;
    }

    cv::Mat applyAnaglyph(const cv::Mat& scene, const cv::Mat& heightMap) const {
        float maxH = 1e-4f;
        for (int y = 0; y < heightMap.rows; ++y) {
            const float* row = heightMap.ptr<float>(y);
            for (int x = 0; x < heightMap.cols; ++x)
                maxH = std::max(maxH, row[x]);
        }
        float maxDisp = 1.0f + param8 * (4.0f + strength * 18.0f);
        cv::Mat mapLx(scene.rows, scene.cols, CV_32FC1);
        cv::Mat mapLy(scene.rows, scene.cols, CV_32FC1);
        cv::Mat mapRx(scene.rows, scene.cols, CV_32FC1);
        cv::Mat mapRy(scene.rows, scene.cols, CV_32FC1);
        for (int y = 0; y < scene.rows; ++y) {
            const float* hrow = heightMap.ptr<float>(y);
            float* lx = mapLx.ptr<float>(y);
            float* ly = mapLy.ptr<float>(y);
            float* rx = mapRx.ptr<float>(y);
            float* ry = mapRy.ptr<float>(y);
            for (int x = 0; x < scene.cols; ++x) {
                float d = (hrow[x] / maxH) * maxDisp;
                lx[x] = x + 0.5f * d;
                rx[x] = x - 0.5f * d;
                ly[x] = static_cast<float>(y);
                ry[x] = static_cast<float>(y);
            }
        }
        cv::Mat leftView, rightView;
        cv::remap(scene, leftView, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::remap(scene, rightView, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        std::vector<cv::Mat> lch, rch;
        cv::split(leftView, lch);
        cv::split(rightView, rch);
        cv::Mat output;
        std::vector<cv::Mat> outCh = {rch[0], rch[1], lch[2]};
        cv::merge(outCh, output);
        return output;
    }

public:
    cv::Mat apply(const cv::Mat& frame) override {
        if (!enabled || frame.empty()) return frame;

        updateTracks(frame);
        updateEdges(frame);
        updateMotionField();

        // Always render a textured mesh. Frontal (yaw=0.5, pitch=0) maps the flat
        // plane 1:1 into the frame; extruded tracks pop toward the camera. Orbit
        // rotates that same plane — no separate "background video" underlay.
        const float yaw = (param4 - 0.5f) * static_cast<float>(M_PI); // ±90°
        const float pitch = param5 * 1.15f;                            // 0 = flat
        const bool orbiting = std::fabs(yaw) > 0.02f || pitch > 0.02f;
        const float focal = 2.0f;
        const float camDist = focal; // identity fill when flat / no height
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float aspect = static_cast<float>(frame.cols) / std::max(1, frame.rows);

        const int maxW = 640;
        float resScale = 1.0f;
        int rw = frame.cols, rh = frame.rows;
        if (rw > maxW) {
            resScale = static_cast<float>(maxW) / rw;
            rw = maxW;
            rh = std::max(1, static_cast<int>(frame.rows * resScale));
        }
        cv::Mat tex;
        if (resScale < 0.999f)
            cv::resize(frame, tex, cv::Size(rw, rh), 0, 0, cv::INTER_AREA);
        else
            tex = frame;

        std::vector<Vert> verts(static_cast<size_t>((kGrid + 1) * (kGrid + 1)));
        for (int j = 0; j <= kGrid; ++j) {
            for (int i = 0; i <= kGrid; ++i) {
                float u = static_cast<float>(i) / kGrid;
                float v = static_cast<float>(j) / kGrid;
                Vert& vt = verts[static_cast<size_t>(j * (kGrid + 1) + i)];
                vt.u = u;
                vt.v = v;

                float px = (u - 0.5f) * 2.0f * aspect;
                float py = (0.5f - v) * 2.0f;
                WaveSample w = sampleWave(u, v, frame.cols, frame.rows);
                EdgeSample e = sampleEdgeSplit(u, v, frame.cols, frame.rows);
                WaveSample m = sampleMotion3D(u, v, frame.cols, frame.rows);
                px += w.ox + e.ox + m.ox;
                py += w.oy + e.oy + m.oy;
                float pz = w.h + e.h + m.h;

                // Yaw around Y, then pitch around X
                float x1 = px * cy + pz * sy;
                float z1 = -px * sy + pz * cy;
                float y2 = py * cp - z1 * sp;
                float z2 = py * sp + z1 * cp;

                float viewZ = camDist - z2;
                vt.iw = 1.0f / std::max(0.12f, viewZ);
                float persp = focal * vt.iw;
                vt.sx = (0.5f + 0.5f * (x1 * persp) / aspect) * (rw - 1);
                vt.sy = (0.5f - 0.5f * (y2 * persp)) * (rh - 1);
            }
        }

        // When orbiting, fit the projected plane into the frame so we don't clip oddly
        if (orbiting) {
            float minSX = 1e9f, maxSX = -1e9f, minSY = 1e9f, maxSY = -1e9f;
            for (const auto& vt : verts) {
                minSX = std::min(minSX, vt.sx); maxSX = std::max(maxSX, vt.sx);
                minSY = std::min(minSY, vt.sy); maxSY = std::max(maxSY, vt.sy);
            }
            float spanX = std::max(1.0f, maxSX - minSX);
            float spanY = std::max(1.0f, maxSY - minSY);
            float fit = 0.92f * std::min((rw - 1) / spanX, (rh - 1) / spanY);
            float midX = 0.5f * (minSX + maxSX);
            float midY = 0.5f * (minSY + maxSY);
            float cxScreen = 0.5f * (rw - 1);
            float cyScreen = 0.5f * (rh - 1);
            for (auto& vt : verts) {
                vt.sx = (vt.sx - midX) * fit + cxScreen;
                vt.sy = (vt.sy - midY) * fit + cyScreen;
            }
        }

        // Dark clear — mesh only (fixes "video in background + rotated copy")
        cv::Mat small(rh, rw, CV_8UC3, cv::Scalar(8, 8, 10));
        cv::Mat zbuf(rh, rw, CV_32FC1, cv::Scalar(0));

        auto idx = [&](int i, int j) { return static_cast<size_t>(j * (kGrid + 1) + i); };
        for (int j = 0; j < kGrid; ++j) {
            for (int i = 0; i < kGrid; ++i) {
                const Vert& v00 = verts[idx(i, j)];
                const Vert& v10 = verts[idx(i + 1, j)];
                const Vert& v01 = verts[idx(i, j + 1)];
                const Vert& v11 = verts[idx(i + 1, j + 1)];
                drawTriangle(small, zbuf, v00, v10, v11, tex);
                drawTriangle(small, zbuf, v00, v11, v01, tex);
            }
        }

        cv::Mat output;
        if (resScale < 0.999f)
            cv::resize(small, output, frame.size(), 0, 0, cv::INTER_LINEAR);
        else
            output = small;

        // Soft anaglyph (toggle + amount slider)
        cv::Mat heightMap = buildHeightMap(frame.rows, frame.cols);
        double minV = 0, maxV = 0;
        cv::minMaxLoc(heightMap, &minV, &maxV);
        if (anaglyphOn && maxV > 0.02 && param8 > 0.05f)
            output = applyAnaglyph(output, heightMap);

        // Calibration overlays (drawn in image space; helpful while tuning)
        if (showMotion && !lastMotion.empty()) {
            cv::Mat motionBgr;
            cv::cvtColor(lastMotion, motionBgr, cv::COLOR_GRAY2BGR);
            if (motionBgr.size() != output.size())
                cv::resize(motionBgr, motionBgr, output.size(), 0, 0, cv::INTER_NEAREST);
            cv::addWeighted(output, 0.55, motionBgr, 0.45, 0, output);
        }
        if (showHeight && maxV > 1e-4) {
            cv::Mat norm, heat;
            heightMap.convertTo(norm, CV_8UC1, 255.0 / maxV);
            cv::applyColorMap(norm, heat, cv::COLORMAP_TURBO);
            cv::addWeighted(output, 0.55, heat, 0.45, 0, output);
        }
        if (showTracks) {
            const int minAge = 4 + static_cast<int>((0.25f + param7 * 0.75f) * (8 + param1 * 20.0f));
            for (const auto& tr : tracks) {
                cv::Scalar col = tr.age >= minAge ? cv::Scalar(40, 220, 80) : cv::Scalar(40, 160, 255);
                float rx = tr.halfW * (0.7f + param3 * 1.5f);
                float ry = tr.halfH * (0.7f + param3 * 1.5f);
                cv::ellipse(output, tr.center, cv::Size(std::max(2, static_cast<int>(rx)),
                                                         std::max(2, static_cast<int>(ry))),
                            0, 0, 360, col, 2, cv::LINE_AA);
                cv::circle(output, tr.center, 3, col, -1, cv::LINE_AA);
                // Motion pointing arrow
                float speed = std::sqrt(tr.vel.x * tr.vel.x + tr.vel.y * tr.vel.y);
                if (speed > 0.4f) {
                    cv::Point2f tip = tr.center + tr.vel * (6.0f + param10 * 10.0f);
                    cv::arrowedLine(output, tr.center, tip, cv::Scalar(0, 220, 255), 2, cv::LINE_AA, 0, 0.35);
                }
                char buf[64];
                std::snprintf(buf, sizeof(buf), "a%d s%.2f", tr.age, tr.sizeNorm);
                cv::putText(output, buf,
                            {static_cast<int>(tr.center.x) + 6, static_cast<int>(tr.center.y) - 6},
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv::LINE_AA);
            }
        }
        if (showEdges) {
            for (const auto& e : edgeTracks) {
                cv::Scalar col = e.blend > 0.45f ? cv::Scalar(0, 220, 255)
                               : (e.age > 4 ? cv::Scalar(0, 140, 220) : cv::Scalar(80, 80, 160));
                int thickness = e.blend > 0.45f ? 2 : 1;
                cv::line(output,
                         {static_cast<int>(e.x0), static_cast<int>(e.y0)},
                         {static_cast<int>(e.x1), static_cast<int>(e.y1)},
                         col, thickness, cv::LINE_AA);
                if (e.blend > 0.2f) {
                    float mx = 0.5f * (e.x0 + e.x1);
                    float my = 0.5f * (e.y0 + e.y1);
                    cv::arrowedLine(output,
                                    {static_cast<int>(mx), static_cast<int>(my)},
                                    {static_cast<int>(mx + e.nx * 18.0f), static_cast<int>(my + e.ny * 18.0f)},
                                    cv::Scalar(255, 120, 40), 1, cv::LINE_AA, 0, 0.35);
                }
            }
        }

        return output;
    }

    void applyDefaultParams() {
        strength = 0.85f;
        param1 = 0.35f;  // Strictness
        param2 = 0.45f;  // Wave Soft
        param3 = 0.65f;  // Wave Radius
        param4 = 0.5f;   // Orbit Yaw
        param5 = 0.0f;   // Orbit Pitch
        param6 = 0.55f;  // Pop Scale
        param7 = 0.4f;   // Hold Time
        param8 = 0.55f;  // Ana Amount
        param9 = 0.65f;  // Size Depth
        param10 = 0.55f; // Motion Point
        param11 = 0.55f; // Wave Lerp (higher = much longer ease)
        param12 = 0.45f; // Edge Split
        param13 = 0.22f; // Edge Sense (conservative)
        param14 = 0.55f; // Edge Lerp
        param15 = 0.40f; // Anticipate
        param16 = 0.55f; // Mot Depth
        param17 = 0.45f; // Mot Soft
        anaglyphOn = true;
        motion3DOn = false;
        optionPage = 0;
    }

    void resetTracking() {
        tracks.clear();
        lastMotion.release();
        motionHeight.release();
        subtractor.release();
        edgeTracks.clear();
    }

    void reset() override {
        applyDefaultParams();
        resetTracking();
        showTracks = false;
        showMotion = false;
        showHeight = false;
        showEdges = false;
        motion3DOn = false;
        optionPage = 0;
    }

    std::string name() const override { return "EnceladusVision"; }
    bool hasParam1() const override { return true; }
    bool hasParam2() const override { return true; }
    bool hasParam3() const override { return true; }
    bool hasParam4() const override { return true; }
    bool hasParam5() const override { return true; }
    bool hasParam6() const override { return true; }
    bool hasParam7() const override { return true; }
    bool hasParam8() const override { return true; }
    bool hasParam9() const override { return true; }
    bool hasParam10() const override { return true; }
    bool hasParam11() const override { return true; }
    bool hasParam12() const override { return true; }
    bool hasParam13() const override { return true; }
    bool hasParam14() const override { return true; }
    bool hasParam15() const override { return true; }
    bool hasParam16() const override { return true; }
    bool hasParam17() const override { return true; }
    std::string param1Name() const override { return "Strictness"; }
    std::string param2Name() const override { return "Wave Soft"; }
    std::string param3Name() const override { return "Wave Radius"; }
    std::string param4Name() const override { return "Orbit Yaw"; }
    std::string param5Name() const override { return "Orbit Pitch"; }
    std::string param6Name() const override { return "Pop Scale"; }
    std::string param7Name() const override { return "Hold Time"; }
    std::string param8Name() const override { return "Ana Amount"; }
    std::string param9Name() const override { return "Size Depth"; }
    std::string param10Name() const override { return "Motion Point"; }
    std::string param11Name() const override { return "Wave Lerp"; }
    std::string param12Name() const override { return "Edge Split"; }
    std::string param13Name() const override { return "Edge Sense"; }
    std::string param14Name() const override { return "Edge Lerp"; }
    std::string param15Name() const override { return "Anticipate"; }
    std::string param16Name() const override { return "Mot Depth"; }
    std::string param17Name() const override { return "Mot Soft"; }
    bool hasReset() const override { return true; }

    int optionPageCount() const override { return 5; }
    std::string optionPageName(int i) const override {
        switch (i) {
            case 0: return "Pop";
            case 1: return "Time";
            case 2: return "Edge";
            case 3: return "Look";
            case 4: return "Mot3D";
            default: return "Params";
        }
    }
    bool showStrengthOnPage() const override { return optionPage == 0; }
    bool showParamOnPage(int n) const override {
        switch (optionPage) {
            case 0: return n == 1 || n == 2 || n == 3 || n == 6 || n == 7 || n == 9 || n == 10;
            case 1: return n == 11 || n == 15;
            case 2: return n == 12 || n == 13 || n == 14;
            case 3: return n == 4 || n == 5 || n == 8;
            case 4: return n == 16 || n == 17;
            default: return true;
        }
    }

    int toolButtonCount() const override { return 9; }
    std::string toolButtonName(int i) const override {
        switch (i) {
            case 0: return motion3DOn ? "Mot3D ON" : "Mot3D";
            case 1: return anaglyphOn ? "Ana ON" : "Ana OFF";
            case 2: return "Front";
            case 3: return "Reset";
            case 4: return "Tracks";
            case 5: return "Motion";
            case 6: return "Height";
            case 7: return "Edges";
            case 8: return "Relearn";
            default: return "";
        }
    }
    bool toolButtonActive(int i) const override {
        switch (i) {
            case 0: return motion3DOn;
            case 1: return anaglyphOn;
            case 4: return showTracks;
            case 5: return showMotion;
            case 6: return showHeight;
            case 7: return showEdges;
            default: return false;
        }
    }
    void toolButtonClick(int i) override {
        switch (i) {
            case 0: motion3DOn = !motion3DOn; if (motion3DOn) optionPage = 4; break;
            case 1: anaglyphOn = !anaglyphOn; break;
            case 2: param4 = 0.5f; param5 = 0.0f; break;
            case 3: reset(); break;
            case 4: showTracks = !showTracks; break;
            case 5: showMotion = !showMotion; break;
            case 6: showHeight = !showHeight; break;
            case 7: showEdges = !showEdges; break;
            case 8: resetTracking(); break;
            default: break;
        }
    }
    std::string statusLine() const override {
        int mature = 0;
        const int minAge = 4 + static_cast<int>((0.25f + param7 * 0.75f) * (8 + param1 * 20.0f));
        for (const auto& tr : tracks)
            if (tr.blend > 0.5f || tr.age >= minAge) ++mature;
        int liveEdges = 0;
        for (const auto& e : edgeTracks)
            if (e.blend > 0.45f) ++liveEdges;
        std::ostringstream ss;
        ss << optionPageName(optionPage)
           << " | tr " << tracks.size() << "/" << mature
           << " ed " << liveEdges << "/" << edgeTracks.size()
           << (motion3DOn ? " | Mot3D" : "")
           << (anaglyphOn ? " | ana" : "");
        return ss.str();
    }
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

static int x11IgnoreErrors(Display*, XErrorEvent*) { return 0; }

bool ensureXDisplay() {
    if (xDisplay) return true;
    xDisplay = XOpenDisplay(nullptr);
    if (!xDisplay) {
        std::cerr << "X11: cannot open display (window capture unavailable)." << std::endl;
        return false;
    }
    XSetErrorHandler(x11IgnoreErrors);
    return true;
}

void stopWindowCapture() {
    windowCaptureActive = false;
    captureXid = 0;
    captureWindowTitle.clear();
}

std::string x11WindowTitle(Display* dpy, Window win) {
    Atom netWmName = XInternAtom(dpy, "_NET_WM_NAME", True);
    Atom utf8 = XInternAtom(dpy, "UTF8_STRING", True);
    if (netWmName != None && utf8 != None) {
        Atom actualType;
        int format = 0;
        unsigned long nItems = 0, bytesAfter = 0;
        unsigned char* data = nullptr;
        if (XGetWindowProperty(dpy, win, netWmName, 0, 1024, False, utf8,
                               &actualType, &format, &nItems, &bytesAfter, &data) == Success &&
            data && nItems > 0) {
            std::string title(reinterpret_cast<char*>(data), nItems);
            XFree(data);
            return title;
        }
        if (data) XFree(data);
    }
    char* name = nullptr;
    if (XFetchName(dpy, win, &name) && name) {
        std::string title(name);
        XFree(name);
        return title;
    }
    return {};
}

bool x11WindowGeom(Display* dpy, Window win, int& w, int& h) {
    XWindowAttributes attr;
    if (!XGetWindowAttributes(dpy, win, &attr)) return false;
    if (attr.map_state != IsViewable) return false;
    w = attr.width;
    h = attr.height;
    return w >= 8 && h >= 8;
}

void scanCaptureWindows() {
    captureWindows.clear();
    if (!ensureXDisplay()) return;

    Window root = DefaultRootWindow(xDisplay);
    int sw = 0, sh = 0;
    if (x11WindowGeom(xDisplay, root, sw, sh)) {
        captureWindows.push_back({root, "Full Desktop / Screen", sw, sh, true});
    } else {
        Screen* scr = DefaultScreenOfDisplay(xDisplay);
        captureWindows.push_back({root, "Full Desktop / Screen",
                                  scr ? scr->width : 1920, scr ? scr->height : 1080, true});
    }

    Atom clientList = XInternAtom(xDisplay, "_NET_CLIENT_LIST", True);
    if (clientList == None)
        clientList = XInternAtom(xDisplay, "_NET_CLIENT_LIST_STACKING", True);

    std::vector<Window> wins;
    if (clientList != None) {
        Atom actualType;
        int format = 0;
        unsigned long nItems = 0, bytesAfter = 0;
        unsigned char* data = nullptr;
        if (XGetWindowProperty(xDisplay, root, clientList, 0, ~0L, False, XA_WINDOW,
                               &actualType, &format, &nItems, &bytesAfter, &data) == Success &&
            data && nItems > 0) {
            auto* list = reinterpret_cast<Window*>(data);
            wins.assign(list, list + nItems);
            XFree(data);
        } else if (data) {
            XFree(data);
        }
    }

    if (wins.empty()) {
        Window rootRet, parent;
        Window* children = nullptr;
        unsigned int nChildren = 0;
        if (XQueryTree(xDisplay, root, &rootRet, &parent, &children, &nChildren) && children) {
            wins.assign(children, children + nChildren);
            XFree(children);
        }
    }

    for (Window win : wins) {
        if (win == 0 || win == root) continue;
        int w = 0, h = 0;
        if (!x11WindowGeom(xDisplay, win, w, h)) continue;
        std::string title = x11WindowTitle(xDisplay, win);
        if (title.empty()) continue;
        // Skip our own UI windows
        if (title.find("Enceladus Video") != std::string::npos) continue;
        if (title.find("Enceladus Control") != std::string::npos) continue;
        captureWindows.push_back({static_cast<unsigned long>(win), title, w, h, false});
    }

    windowScroll = std::max(0, std::min(windowScroll,
        std::max(0, static_cast<int>(captureWindows.size()) - SOURCE_MENU_ROWS)));
}

bool grabX11WindowFrame(Window win, cv::Mat& outBgr) {
    if (!ensureXDisplay()) return false;
    XWindowAttributes attr;
    if (!XGetWindowAttributes(xDisplay, win, &attr)) return false;
    if (attr.map_state != IsViewable || attr.width < 2 || attr.height < 2) return false;

    XImage* img = XGetImage(xDisplay, win, 0, 0, attr.width, attr.height, AllPlanes, ZPixmap);
    if (!img || !img->data) {
        if (img) XDestroyImage(img);
        return false;
    }

    const int w = img->width;
    const int h = img->height;
    cv::Mat tmp;

    if (img->bits_per_pixel == 32) {
        tmp = cv::Mat(h, w, CV_8UC4);
        for (int y = 0; y < h; ++y) {
            const unsigned char* src = reinterpret_cast<unsigned char*>(img->data) + y * img->bytes_per_line;
            unsigned char* dst = tmp.ptr<unsigned char>(y);
            std::memcpy(dst, src, static_cast<size_t>(w) * 4);
        }
        // X11 ZPixmap is typically BGRA on little-endian
        cv::cvtColor(tmp, outBgr, cv::COLOR_BGRA2BGR);
    } else if (img->bits_per_pixel == 24) {
        tmp = cv::Mat(h, w, CV_8UC3);
        for (int y = 0; y < h; ++y) {
            const unsigned char* src = reinterpret_cast<unsigned char*>(img->data) + y * img->bytes_per_line;
            unsigned char* dst = tmp.ptr<unsigned char>(y);
            for (int x = 0; x < w; ++x) {
                dst[x * 3 + 0] = src[x * 3 + 0];
                dst[x * 3 + 1] = src[x * 3 + 1];
                dst[x * 3 + 2] = src[x * 3 + 2];
            }
        }
        outBgr = tmp;
    } else {
        // Fallback: sample pixels via XGetPixel (slower)
        outBgr.create(h, w, CV_8UC3);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                unsigned long p = XGetPixel(img, x, y);
                outBgr.at<cv::Vec3b>(y, x) = {
                    static_cast<uchar>(p & 0xff),
                    static_cast<uchar>((p >> 8) & 0xff),
                    static_cast<uchar>((p >> 16) & 0xff)
                };
            }
        }
    }

    XDestroyImage(img);
    return !outBgr.empty();
}

float captureTargetFps() {
    // Slider: 0 → ~8 fps, 0.55 → ~30, 1 → ~60
    return 8.0f + capFpsNorm * 52.0f;
}

void resetCaptureTune() {
    capCropL = capCropR = capCropT = capCropB = 0.0f;
    capDownscale = 1.0f;
    capFpsNorm = 0.55f;
}

cv::Mat applyCaptureTune(const cv::Mat& frame) {
    if (frame.empty()) return frame;
    cv::Mat out = frame;

    float cl = std::max(0.0f, std::min(0.45f, capCropL));
    float cr = std::max(0.0f, std::min(0.45f, capCropR));
    float ct = std::max(0.0f, std::min(0.45f, capCropT));
    float cb = std::max(0.0f, std::min(0.45f, capCropB));
    if (cl + cr >= 0.95f) { cl = 0.0f; cr = 0.0f; }
    if (ct + cb >= 0.95f) { ct = 0.0f; cb = 0.0f; }

    if (cl > 0.0005f || cr > 0.0005f || ct > 0.0005f || cb > 0.0005f) {
        int x0 = static_cast<int>(frame.cols * cl);
        int y0 = static_cast<int>(frame.rows * ct);
        int x1 = frame.cols - static_cast<int>(frame.cols * cr);
        int y1 = frame.rows - static_cast<int>(frame.rows * cb);
        x0 = std::max(0, std::min(frame.cols - 2, x0));
        y0 = std::max(0, std::min(frame.rows - 2, y0));
        x1 = std::max(x0 + 2, std::min(frame.cols, x1));
        y1 = std::max(y0 + 2, std::min(frame.rows, y1));
        out = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
    }

    float scale = std::max(0.25f, std::min(1.0f, capDownscale));
    if (scale < 0.995f && !out.empty()) {
        cv::Mat scaled;
        cv::resize(out, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
        out = scaled;
    }
    return out;
}

float formatAspectFromNorm(float n) {
    n = std::max(0.0f, std::min(1.0f, n));
    return FORMAT_ASPECT_MIN + n * (FORMAT_ASPECT_MAX - FORMAT_ASPECT_MIN);
}

float formatNormFromAspect(float aspect) {
    float n = (aspect - FORMAT_ASPECT_MIN) / (FORMAT_ASPECT_MAX - FORMAT_ASPECT_MIN);
    return std::max(0.0f, std::min(1.0f, n));
}

bool formatAspectActive() {
    return formatUseCustom || (formatPresetIndex >= 0 && formatPresetIndex < kAspectFormatCount);
}

float currentFormatAspect() {
    if (formatUseCustom)
        return formatAspectFromNorm(formatAspectNorm);
    if (formatPresetIndex >= 0 && formatPresetIndex < kAspectFormatCount)
        return kAspectFormats[formatPresetIndex].aspect;
    return formatAspectFromNorm(formatAspectNorm);
}

void syncFormatAspectNormFromPreset() {
    if (formatPresetIndex >= 0 && formatPresetIndex < kAspectFormatCount)
        formatAspectNorm = formatNormFromAspect(kAspectFormats[formatPresetIndex].aspect);
}

int forceResWidth() {
    if (forceResPreset >= 0 && forceResPreset < kForceResCount)
        return kForceResPresets[forceResPreset].w;
    if (forceResPreset == -2)
        return 320 + static_cast<int>(std::lround(std::max(0.0f, std::min(1.0f, forceResWNorm)) * (3840 - 320)));
    return 0;
}

int forceResHeight() {
    if (forceResPreset >= 0 && forceResPreset < kForceResCount)
        return kForceResPresets[forceResPreset].h;
    if (forceResPreset == -2)
        return 240 + static_cast<int>(std::lround(std::max(0.0f, std::min(1.0f, forceResHNorm)) * (2160 - 240)));
    return 0;
}

bool forceResActive() {
    return forceResPreset == -2 || (forceResPreset >= 0 && forceResPreset < kForceResCount);
}

cv::Mat applyFormatAspect(const cv::Mat& frame) {
    if (frame.empty() || !formatAspectActive())
        return frame;
    const float target = currentFormatAspect();
    if (target < 0.05f) return frame;
    const float src = static_cast<float>(frame.cols) / std::max(1, frame.rows);
    if (std::fabs(src - target) < 0.008f) return frame;

    if (formatMode == 0) {
        // Letterbox / pillarbox — keep full image, pad to target aspect
        if (src > target) {
            int newH = std::max(2, static_cast<int>(std::lround(frame.cols / target)));
            int pad = (newH - frame.rows) / 2;
            cv::Mat out;
            cv::copyMakeBorder(frame, out, pad, newH - frame.rows - pad, 0, 0,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            return out;
        }
        int newW = std::max(2, static_cast<int>(std::lround(frame.rows * target)));
        int pad = (newW - frame.cols) / 2;
        cv::Mat out;
        cv::copyMakeBorder(frame, out, 0, 0, pad, newW - frame.cols - pad,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        return out;
    }

    // Center crop to target aspect
    if (src > target) {
        int newW = std::max(2, static_cast<int>(std::lround(frame.rows * target)));
        int x0 = (frame.cols - newW) / 2;
        return frame(cv::Rect(x0, 0, newW, frame.rows)).clone();
    }
    int newH = std::max(2, static_cast<int>(std::lround(frame.cols / target)));
    int y0 = (frame.rows - newH) / 2;
    return frame(cv::Rect(0, y0, frame.cols, newH)).clone();
}

cv::Mat applyForcedResolution(const cv::Mat& frame) {
    if (frame.empty() || !forceResActive())
        return frame;
    int tw = std::max(2, forceResWidth());
    int th = std::max(2, forceResHeight());
    if (frame.cols == tw && frame.rows == th)
        return frame;
    cv::Mat out;
    int interp = (tw < frame.cols || th < frame.rows) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(frame, out, cv::Size(tw, th), 0, 0, interp);
    return out;
}

std::string makeWindowSourceId(const CaptureWindow& w) {
    if (w.isScreen) return "win:screen";
    std::ostringstream oss;
    oss << "win:0x" << std::hex << w.xid;
    return oss.str();
}

bool parseWindowSource(const std::string& source, unsigned long& xid, bool& isScreen) {
    isScreen = false;
    xid = 0;
    if (source == "win:screen" || source == "screen") {
        isScreen = true;
        return true;
    }
    if (source.rfind("win:", 0) != 0) return false;
    std::string rest = source.substr(4);
    try {
        xid = std::stoul(rest, nullptr, 0);
    } catch (...) {
        return false;
    }
    return xid != 0;
}

bool startWindowCapture(const std::string& source) {
    if (!ensureXDisplay()) return false;
    unsigned long xid = 0;
    bool isScreen = false;
    if (!parseWindowSource(source, xid, isScreen)) return false;

    Window win = isScreen ? DefaultRootWindow(xDisplay) : static_cast<Window>(xid);
    int w = 0, h = 0;
    if (!x11WindowGeom(xDisplay, win, w, h) && !isScreen) {
        std::cerr << "Window not viewable or missing: " << source << std::endl;
        return false;
    }

    cap.release();
    captureXid = win;
    windowCaptureActive = true;
    captureWindowTitle = isScreen ? "Full Desktop / Screen" : x11WindowTitle(xDisplay, win);
    if (captureWindowTitle.empty())
        captureWindowTitle = source;
    videoDuration = 0.0;
    lastCapGrabTime = 0.0;
    return true;
}

// Open a playlist entry: numeric = camera, win:* = X11 window/screen, else file/URL.
bool openVideoSource(const std::string& source) {
    stopWindowCapture();

    if (source.rfind("win:", 0) == 0 || source == "screen")
        return startWindowCapture(source);

    bool isCamera = !source.empty() &&
        std::all_of(source.begin(), source.end(),
                    [](unsigned char c) { return std::isdigit(c); });
    if (isCamera) {
        int index = std::stoi(source);
        hwDecodeStatus = "camera";
        // Prefer V4L2 on Linux so "0" is not parsed as a GStreamer bin.
        if (cap.open(index, cv::CAP_V4L2)) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            return true;
        }
        if (cap.open(index, cv::CAP_ANY)) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            return true;
        }
        return false;
    }

    cap.release();
    if (useHwDecode) {
        // OpenCV 4.5.2+: ask FFmpeg for any available HW decoder (VAAPI/MFX/…).
        // Decoded frames are still delivered as cv::Mat in system memory for filters.
        std::vector<int> params = {
            static_cast<int>(cv::CAP_PROP_HW_ACCELERATION),
            static_cast<int>(cv::VIDEO_ACCELERATION_ANY),
            static_cast<int>(cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL),
            useOpenCL ? 1 : 0,
        };
        if (cap.open(source, cv::CAP_FFMPEG, params)) {
            int accel = static_cast<int>(cap.get(cv::CAP_PROP_HW_ACCELERATION));
            if (accel == cv::VIDEO_ACCELERATION_VAAPI) hwDecodeStatus = "vaapi";
            else if (accel == cv::VIDEO_ACCELERATION_MFX) hwDecodeStatus = "mfx";
            else if (accel == cv::VIDEO_ACCELERATION_D3D11) hwDecodeStatus = "d3d11";
            else if (accel == cv::VIDEO_ACCELERATION_NONE) hwDecodeStatus = "sw-fallback";
            else hwDecodeStatus = "hw-any";
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            return true;
        }
        cap.release();
    }

    if (cap.open(source, cv::CAP_FFMPEG) || cap.open(source)) {
        hwDecodeStatus = useHwDecode ? "software" : "software(--no-hw-decode)";
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        return true;
    }
    hwDecodeStatus = "failed";
    return false;
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

std::string resolveAppRoot(const char* argv0) {
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !exe.empty()) {
        fs::path dir = fs::weakly_canonical(exe.parent_path(), ec);
        if (!ec && !dir.empty()) return dir.string();
        return exe.parent_path().string();
    }
    if (argv0 && argv0[0]) {
        fs::path a(argv0);
        if (a.has_parent_path()) {
            fs::path dir = fs::weakly_canonical(fs::absolute(a).parent_path(), ec);
            if (!ec && !dir.empty()) return dir.string();
            return fs::absolute(a).parent_path().string();
        }
    }
    fs::path cwd = fs::current_path(ec);
    return ec ? std::string(".") : cwd.string();
}

std::string playlistItemLabel(const std::string& source) {
    if (source.rfind("win:", 0) == 0 || source == "screen")
        return source;
    bool isCamera = !source.empty() &&
        std::all_of(source.begin(), source.end(),
                    [](unsigned char c) { return std::isdigit(c); });
    if (isCamera) return "Camera " + source;
    return fs::path(source).filename().string();
}

bool isCameraSourceId(const std::string& source) {
    return !source.empty() &&
        std::all_of(source.begin(), source.end(),
                    [](unsigned char c) { return std::isdigit(c); });
}

void refreshBrowserEntries() {
    browserEntries.clear();
    std::error_code ec;
    if (browserPath.empty()) {
        browserPath = !appRootPath.empty() ? appRootPath : ".";
    }
    fs::path path(browserPath);
    if (!fs::exists(path, ec) || !fs::is_directory(path, ec)) {
        browserPath = !appRootPath.empty() ? appRootPath : ".";
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

bool applyOpenedSourceTiming() {
    if (windowCaptureActive) {
        videoDuration = 0.0;
    } else {
        double fps = cap.get(cv::CAP_PROP_FPS);
        double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        if (fps > 1e-3 && frames > 0)
            videoDuration = frames / fps;
        else
            videoDuration = 0.0;
    }
    currentVideoTime = 0.0;
    lastFrameTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    latestFrame.release();
    isPaused = false;
    frameAvailable = true;
    return true;
}

bool enqueueSource(const std::string& source, bool playNow) {
    // Drop the idle default camera entry when building a real file queue.
    if (!playlist.empty() && playlist.size() == 1 && playlist[0] == "0" &&
        source != "0" && !isCameraSourceId(source) &&
        source.rfind("win:", 0) != 0 && source != "screen") {
        playlist.clear();
        currentVideoIndex = 0;
    }

    auto it = std::find(playlist.begin(), playlist.end(), source);
    int idx;
    if (it != playlist.end()) {
        idx = static_cast<int>(it - playlist.begin());
    } else {
        playlist.push_back(source);
        idx = static_cast<int>(playlist.size()) - 1;
    }

    if (!playNow)
        return true;

    cap.release();
    currentVideoIndex = idx;
    if (!openVideoSource(source)) {
        std::cerr << "Failed to open source: " << source << std::endl;
        return false;
    }
    return applyOpenedSourceTiming();
}

bool switchToSource(const std::string& source) {
    return enqueueSource(source, true);
}

bool jumpToPlaylistIndex(int idx) {
    if (idx < 0 || idx >= static_cast<int>(playlist.size()))
        return false;
    cap.release();
    currentVideoIndex = idx;
    if (!openVideoSource(playlist[idx])) {
        std::cerr << "Failed to open source: " << playlist[idx] << std::endl;
        return false;
    }
    return applyOpenedSourceTiming();
}

void removePlaylistIndex(int idx) {
    if (idx < 0 || idx >= static_cast<int>(playlist.size()))
        return;
    bool wasCurrent = (idx == currentVideoIndex);
    playlist.erase(playlist.begin() + idx);
    if (playlist.empty()) {
        playlist.push_back("0");
        currentVideoIndex = 0;
        cap.release();
        openVideoSource(playlist[0]);
        applyOpenedSourceTiming();
        return;
    }
    if (currentVideoIndex > idx)
        --currentVideoIndex;
    else if (wasCurrent) {
        if (currentVideoIndex >= static_cast<int>(playlist.size()))
            currentVideoIndex = static_cast<int>(playlist.size()) - 1;
        jumpToPlaylistIndex(currentVideoIndex);
    }
}

void clearPlaylistKeepCurrent() {
    if (playlist.empty()) return;
    std::string cur = playlist[std::max(0, std::min(currentVideoIndex,
                          static_cast<int>(playlist.size()) - 1))];
    playlist.clear();
    playlist.push_back(cur);
    currentVideoIndex = 0;
}

void openFileBrowserMenu() {
    showPieMenu = false;
    pieHoverIndex = -1;
    sourceMenuMode = SourceMenuMode::Files;
    if (browserPath.empty())
        browserPath = !appRootPath.empty() ? appRootPath : ".";
    browserScroll = 0;
    refreshBrowserEntries();
}

void openQueueMenu() {
    showPieMenu = false;
    pieHoverIndex = -1;
    sourceMenuMode = SourceMenuMode::Queue;
    queueScroll = std::max(0, std::min(queueScroll,
        std::max(0, static_cast<int>(playlist.size()) - SOURCE_MENU_ROWS)));
}

void openCameraMenu() {
    showPieMenu = false;
    pieHoverIndex = -1;
    sourceMenuMode = SourceMenuMode::Cameras;
    cameraScroll = 0;
    scanCameras();
}

void openWindowMenu() {
    showPieMenu = false;
    pieHoverIndex = -1;
    sourceMenuMode = SourceMenuMode::Windows;
    windowScroll = 0;
    scanCaptureWindows();
}

void closeSourceMenu() {
    sourceMenuMode = SourceMenuMode::Closed;
}

void drawText(float x, float y, const std::string& text, void* font = GLUT_BITMAP_HELVETICA_12);
void drawButton(float x, float y, float w, float h, const std::string& label, bool hover = false, bool active = false);
bool isInside(float mx, float my, float x, float y, float w, float h);

void drawSourceMenu() {
    if (sourceMenuMode == SourceMenuMode::Closed) return;

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
        drawText(x + 0.03f, y + h - 0.05f, "Open / Queue Videos", GLUT_BITMAP_HELVETICA_18);
        drawText(x + 0.03f, y + h - 0.11f,
                 truncateLabel(browserPath, 70), GLUT_BITMAP_HELVETICA_12);

        drawButton(x + 0.03f, y + 0.03f, 0.12f, 0.06f, "Up");
        drawButton(x + 0.16f, y + 0.03f, 0.12f, 0.06f, "App");
        drawButton(x + 0.29f, y + 0.03f, 0.10f, 0.06f, "~");
        drawButton(x + 0.40f, y + 0.03f, 0.12f, 0.06f, "Refresh");
        drawButton(x + 0.03f, y + h - 0.08f, 0.14f, 0.06f, "Play", false, !browserQueueMode);
        drawButton(x + 0.19f, y + h - 0.08f, 0.14f, 0.06f, "Queue", false, browserQueueMode);
        drawButton(x + 0.35f, y + h - 0.08f, 0.16f, 0.06f,
                   "Que(" + std::to_string(playlist.size()) + ")");
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
            std::string full = isDir ? "" : (fs::path(browserPath) / browserEntries[idx].first).string();
            bool queued = !isDir && std::find(playlist.begin(), playlist.end(), full) != playlist.end();
            glColor3f(isDir ? 0.18f : (queued ? 0.12f : 0.16f),
                      isDir ? 0.22f : (queued ? 0.28f : 0.16f),
                      isDir ? 0.30f : (queued ? 0.18f : 0.20f));
            glBegin(GL_QUADS);
            glVertex2f(x + 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY + 0.045f);
            glVertex2f(x + 0.03f, rowY + 0.045f);
            glEnd();
            glColor3f(isDir ? 0.75f : 0.95f, isDir ? 0.85f : 0.95f, 1.0f);
            std::string prefix = isDir ? "[DIR] " : (queued ? "[Q]   " : "      ");
            drawText(x + 0.05f, rowY, truncateLabel(prefix + browserEntries[idx].first, 60));
        }
        glColor3f(0.7f, 0.75f, 0.85f);
        drawText(x + 0.53f, y + h - 0.06f,
                 browserQueueMode ? "Click file = add to queue" : "Click file = play now",
                 GLUT_BITMAP_HELVETICA_10);
    } else if (sourceMenuMode == SourceMenuMode::Queue) {
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(x + 0.03f, y + h - 0.05f, "Playback Queue", GLUT_BITMAP_HELVETICA_18);
        {
            std::ostringstream info;
            info << playlist.size() << " item(s)  |  now #"
                 << (playlist.empty() ? 0 : currentVideoIndex + 1);
            drawText(x + 0.03f, y + h - 0.11f, info.str(), GLUT_BITMAP_HELVETICA_12);
        }

        drawButton(x + 0.03f, y + 0.03f, 0.16f, 0.06f, "Clear");
        drawButton(x + 0.21f, y + 0.03f, 0.16f, 0.06f, "Rem");
        drawButton(x + 0.39f, y + 0.03f, 0.16f, 0.06f, "Open");
        drawButton(x + w - 0.35f, y + 0.03f, 0.14f, 0.06f, "Prev");
        drawButton(x + w - 0.19f, y + 0.03f, 0.14f, 0.06f, "Next");
        drawButton(x + w - 0.19f, y + h - 0.08f, 0.14f, 0.06f, "Close");

        int total = static_cast<int>(playlist.size());
        if (total == 0) {
            glColor3f(0.7f, 0.7f, 0.75f);
            drawText(x + 0.05f, listTop, "Queue empty.");
        }
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = queueScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            bool cur = (idx == currentVideoIndex);
            glColor3f(cur ? 0.15f : 0.16f, cur ? 0.32f : 0.16f, cur ? 0.22f : 0.22f);
            glBegin(GL_QUADS);
            glVertex2f(x + 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY + 0.045f);
            glVertex2f(x + 0.03f, rowY + 0.045f);
            glEnd();
            glColor3f(0.95f, 0.95f, 1.0f);
            std::ostringstream label;
            label << (idx + 1) << ". " << playlistItemLabel(playlist[idx]);
            if (cur) label << "  <<";
            drawText(x + 0.05f, rowY, truncateLabel(label.str(), 62));
        }
    } else if (sourceMenuMode == SourceMenuMode::Cameras) {
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
    } else if (sourceMenuMode == SourceMenuMode::Windows) {
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(x + 0.03f, y + h - 0.05f, "Live Window Capture", GLUT_BITMAP_HELVETICA_18);
        drawText(x + 0.03f, y + h - 0.11f,
                 std::to_string(captureWindows.size()) + " target(s) — pick a window to process",
                 GLUT_BITMAP_HELVETICA_12);

        drawButton(x + 0.03f, y + 0.03f, 0.20f, 0.06f, "Refresh");
        drawButton(x + w - 0.35f, y + 0.03f, 0.14f, 0.06f, "Prev");
        drawButton(x + w - 0.19f, y + 0.03f, 0.14f, 0.06f, "Next");
        drawButton(x + w - 0.19f, y + h - 0.08f, 0.14f, 0.06f, "Close");

        int total = static_cast<int>(captureWindows.size());
        if (total == 0) {
            glColor3f(0.7f, 0.7f, 0.75f);
            drawText(x + 0.05f, listTop, "No windows found (need X11 display).");
        }
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = windowScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            bool screen = captureWindows[idx].isScreen;
            glColor3f(screen ? 0.22f : 0.16f, screen ? 0.18f : 0.16f, screen ? 0.28f : 0.24f);
            glBegin(GL_QUADS);
            glVertex2f(x + 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY - 0.015f);
            glVertex2f(x + w - 0.03f, rowY + 0.045f);
            glVertex2f(x + 0.03f, rowY + 0.045f);
            glEnd();
            glColor3f(0.95f, 0.92f, 1.0f);
            std::ostringstream label;
            label << captureWindows[idx].title
                  << "  (" << captureWindows[idx].width << "x" << captureWindows[idx].height << ")";
            drawText(x + 0.05f, rowY, truncateLabel(label.str(), 62));
        }
    }
}

bool hitTestSourceMenu(float fx, float fy) {
    if (sourceMenuMode == SourceMenuMode::Closed) return false;

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
        if (isInside(fx, fy, x + 0.03f, y + 0.03f, 0.12f, 0.06f)) {
            fs::path parent = fs::path(browserPath).parent_path();
            if (!parent.empty()) {
                browserPath = parent.string();
                browserScroll = 0;
                refreshBrowserEntries();
            }
            return true;
        }
        if (isInside(fx, fy, x + 0.16f, y + 0.03f, 0.12f, 0.06f)) {
            browserPath = !appRootPath.empty() ? appRootPath : ".";
            browserScroll = 0;
            refreshBrowserEntries();
            return true;
        }
        if (isInside(fx, fy, x + 0.29f, y + 0.03f, 0.10f, 0.06f)) {
            const char* home = std::getenv("HOME");
            browserPath = home ? home : ".";
            browserScroll = 0;
            refreshBrowserEntries();
            return true;
        }
        if (isInside(fx, fy, x + 0.40f, y + 0.03f, 0.12f, 0.06f)) {
            refreshBrowserEntries();
            return true;
        }
        if (isInside(fx, fy, x + 0.03f, y + h - 0.08f, 0.14f, 0.06f)) {
            browserQueueMode = false;
            return true;
        }
        if (isInside(fx, fy, x + 0.19f, y + h - 0.08f, 0.14f, 0.06f)) {
            browserQueueMode = true;
            return true;
        }
        if (isInside(fx, fy, x + 0.35f, y + h - 0.08f, 0.16f, 0.06f)) {
            openQueueMenu();
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
                    if (browserQueueMode) {
                        enqueueSource(full, false);
                    } else if (switchToSource(full)) {
                        closeSourceMenu();
                    }
                }
                return true;
            }
        }
    } else if (sourceMenuMode == SourceMenuMode::Queue) {
        if (isInside(fx, fy, x + 0.03f, y + 0.03f, 0.16f, 0.06f)) {
            clearPlaylistKeepCurrent();
            return true;
        }
        if (isInside(fx, fy, x + 0.21f, y + 0.03f, 0.16f, 0.06f)) {
            removePlaylistIndex(currentVideoIndex);
            return true;
        }
        if (isInside(fx, fy, x + 0.39f, y + 0.03f, 0.16f, 0.06f)) {
            openFileBrowserMenu();
            return true;
        }
        if (isInside(fx, fy, x + w - 0.35f, y + 0.03f, 0.14f, 0.06f)) {
            queueScroll = std::max(0, queueScroll - SOURCE_MENU_ROWS);
            return true;
        }
        if (isInside(fx, fy, x + w - 0.19f, y + 0.03f, 0.14f, 0.06f)) {
            int maxScroll = std::max(0, static_cast<int>(playlist.size()) - SOURCE_MENU_ROWS);
            queueScroll = std::min(maxScroll, queueScroll + SOURCE_MENU_ROWS);
            return true;
        }

        int total = static_cast<int>(playlist.size());
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = queueScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            if (isInside(fx, fy, x + 0.03f, rowY - 0.015f, w - 0.06f, 0.06f)) {
                jumpToPlaylistIndex(idx);
                return true;
            }
        }
    } else if (sourceMenuMode == SourceMenuMode::Cameras) {
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
    } else if (sourceMenuMode == SourceMenuMode::Windows) {
        if (isInside(fx, fy, x + 0.03f, y + 0.03f, 0.20f, 0.06f)) {
            scanCaptureWindows();
            return true;
        }
        if (isInside(fx, fy, x + w - 0.35f, y + 0.03f, 0.14f, 0.06f)) {
            windowScroll = std::max(0, windowScroll - SOURCE_MENU_ROWS);
            return true;
        }
        if (isInside(fx, fy, x + w - 0.19f, y + 0.03f, 0.14f, 0.06f)) {
            int maxScroll = std::max(0, static_cast<int>(captureWindows.size()) - SOURCE_MENU_ROWS);
            windowScroll = std::min(maxScroll, windowScroll + SOURCE_MENU_ROWS);
            return true;
        }

        int total = static_cast<int>(captureWindows.size());
        for (int i = 0; i < SOURCE_MENU_ROWS; ++i) {
            int idx = windowScroll + i;
            if (idx >= total) break;
            float rowY = listTop - i * rowH;
            if (isInside(fx, fy, x + 0.03f, rowY - 0.015f, w - 0.06f, 0.06f)) {
                if (switchToSource(makeWindowSourceId(captureWindows[idx])))
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
                activeSlider = idx * AE_SLIDER_STRIDE;
            return true;
        }
        y -= FILT_ROW_H;
    }
    return true; // absorb clicks inside panel chrome
}

// Active-effects panel on the left — kept above timeline / view transport.
const float AE_PANEL_X = -0.95f;
const float AE_PANEL_W = 0.48f;
const float AE_CONTENT_X = -0.93f;
const float AE_SLIDER_W = 0.42f;
const float AE_SLIDER_H = 0.020f;
const float AE_TOP_Y = 0.78f;
const float AE_TITLE_STEP = 0.085f;   // room under "Active Effects" header
const float AE_NAME_STEP = 0.048f;
const float AE_LABEL_STEP = 0.030f;   // slider name row
const float AE_SLIDER_GAP = 0.022f;   // gap after track before next label
const float AE_PARAM_STEP = AE_LABEL_STEP + AE_SLIDER_H + AE_SLIDER_GAP;
const float AE_FILTER_GAP = 0.035f;
const float AE_BOTTOM_LIMIT = -0.58f;
const float AE_PAGE_BAR_H = 0.058f;   // fixed page footer (not scrolled)
const float AE_TOOL_H = 0.038f;
const float AE_TOOL_GAP = 0.01f;
const float AE_TOOL_ROW = 0.050f;
const float AE_STATUS_STEP = 0.038f;
const float AE_TOOLS_PAD = 0.018f;    // gap before tool buttons

bool anyFilterEnabled() {
    for (const auto& filter : filters) {
        if (filter->enabled) return true;
    }
    return false;
}

static bool aeShowsStrength(const std::unique_ptr<VideoFilter>& f) {
    return f->hasStrength() && f->showStrengthOnPage();
}
static bool aeShowsParam(const std::unique_ptr<VideoFilter>& f, int n) {
    auto has = [&](int i) -> bool {
        switch (i) {
            case 1: return f->hasParam1(); case 2: return f->hasParam2();
            case 3: return f->hasParam3(); case 4: return f->hasParam4();
            case 5: return f->hasParam5(); case 6: return f->hasParam6();
            case 7: return f->hasParam7(); case 8: return f->hasParam8();
            case 9: return f->hasParam9(); case 10: return f->hasParam10();
            case 11: return f->hasParam11(); case 12: return f->hasParam12();
            case 13: return f->hasParam13(); case 14: return f->hasParam14();
            case 15: return f->hasParam15(); case 16: return f->hasParam16();
            case 17: return f->hasParam17();
            default: return false;
        }
    };
    return has(n) && f->showParamOnPage(n);
}

static bool aeNeedsPageBar() {
    for (const auto& filter : filters) {
        if (filter->enabled && filter->optionPageCount() > 1)
            return true;
    }
    return false;
}

static float aeScrollBottom() {
    return aeNeedsPageBar() ? (AE_BOTTOM_LIMIT + AE_PAGE_BAR_H) : AE_BOTTOM_LIMIT;
}

static VideoFilter* aePagedFilter() {
    for (auto& filter : filters) {
        if (filter->enabled && filter->optionPageCount() > 1)
            return filter.get();
    }
    return nullptr;
}

float activeEffectsContentHeight() {
    float h = AE_TITLE_STEP;
    for (const auto& filter : filters) {
        if (!filter->enabled) continue;
        h += AE_NAME_STEP;
        if (!filter->statusLine().empty()) h += AE_STATUS_STEP;
        if (aeShowsStrength(filter)) h += AE_PARAM_STEP;
        for (int p = 1; p <= 17; ++p)
            if (aeShowsParam(filter, p)) h += AE_PARAM_STEP;
        int tools = filter->toolButtonCount();
        if (tools > 0) {
            h += AE_TOOLS_PAD;
            int rows = (tools + 1) / 2;
            h += rows * AE_TOOL_ROW + 0.01f;
        }
        h += AE_FILTER_GAP;
    }
    return h;
}

void clampAePanelScroll() {
    const float contentH = activeEffectsContentHeight();
    const float viewH = AE_TOP_Y - aeScrollBottom();
    float maxScroll = std::max(0.0f, contentH - viewH);
    aePanelScroll = std::max(0.0f, std::min(aePanelScroll, maxScroll));
}

void drawSliderTrackOnly(float x, float y, float w, float h, float value) {
    glColor3f(sliderTrackColor.r, sliderTrackColor.g, sliderTrackColor.b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();
    float thumbPos = x + (w - h) * value;
    glColor3f(sliderThumbColor.r, sliderThumbColor.g, sliderThumbColor.b);
    glBegin(GL_QUADS);
    glVertex2f(thumbPos, y);
    glVertex2f(thumbPos + h, y);
    glVertex2f(thumbPos + h, y + h);
    glVertex2f(thumbPos, y + h);
    glEnd();
}

void drawActiveEffectsPanel() {
    if (!anyFilterEnabled()) return;

    clampAePanelScroll();
    const float contentH = activeEffectsContentHeight();
    const float listY = aeScrollBottom();
    const float viewH = AE_TOP_Y - listY;
    const bool pageBar = aeNeedsPageBar();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.82f);
    glBegin(GL_QUADS);
    glVertex2f(AE_PANEL_X, AE_BOTTOM_LIMIT);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, AE_BOTTOM_LIMIT);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, AE_TOP_Y);
    glVertex2f(AE_PANEL_X, AE_TOP_Y);
    glEnd();

    glColor4f(0.35f, 0.35f, 0.45f, 0.95f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(AE_PANEL_X, AE_BOTTOM_LIMIT);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, AE_BOTTOM_LIMIT);
    glVertex2f(AE_PANEL_X + AE_PANEL_W, AE_TOP_Y);
    glVertex2f(AE_PANEL_X, AE_TOP_Y);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1.0f, 1.0f, 1.0f);
    drawText(AE_CONTENT_X, AE_TOP_Y - 0.035f, "Active Effects", GLUT_BITMAP_HELVETICA_18);
    if (contentH > viewH + 0.01f) {
        glColor3f(0.65f, 0.65f, 0.75f);
        drawText(AE_CONTENT_X + 0.30f, AE_TOP_Y - 0.035f, "scroll", GLUT_BITMAP_HELVETICA_10);
    }

    // Clip scrolled content above the page footer
    float currentY = AE_TOP_Y - AE_TITLE_STEP + aePanelScroll;
    for (size_t i = 0; i < filters.size(); ++i) {
        if (!filters[i]->enabled) continue;

        auto inView = [&](float yBot, float yTop) {
            return yTop >= listY + 0.01f && yBot <= AE_TOP_Y - AE_TITLE_STEP + 0.02f;
        };

        if (inView(currentY - 0.02f, currentY + 0.02f)) {
            glColor3f(0.85f, 0.85f, 1.0f);
            drawText(AE_CONTENT_X, currentY, filters[i]->name(), GLUT_BITMAP_HELVETICA_12);
        }
        currentY -= AE_NAME_STEP;

        // Status under filter name (not under tools)
        std::string status = filters[i]->statusLine();
        if (!status.empty()) {
            if (inView(currentY - 0.02f, currentY + 0.02f)) {
                glColor3f(0.65f, 0.8f, 0.7f);
                drawText(AE_CONTENT_X, currentY, status, GLUT_BITMAP_HELVETICA_10);
            }
            currentY -= AE_STATUS_STEP;
        }

        auto drawParam = [&](float value, const std::string& label) {
            if (inView(currentY - AE_PARAM_STEP, currentY + 0.01f)) {
                char valBuf[32];
                std::snprintf(valBuf, sizeof(valBuf), "%.2f", value);
                glColor3f(0.78f, 0.78f, 0.88f);
                drawText(AE_CONTENT_X, currentY, label + "  " + valBuf, GLUT_BITMAP_HELVETICA_10);
            }
            currentY -= AE_LABEL_STEP;
            float sy = currentY - AE_SLIDER_H;
            if (inView(sy, currentY + 0.01f))
                drawSliderTrackOnly(AE_CONTENT_X, sy, AE_SLIDER_W, AE_SLIDER_H, value);
            currentY -= (AE_SLIDER_H + AE_SLIDER_GAP);
        };

        if (aeShowsStrength(filters[i])) drawParam(filters[i]->strength, "Strength");
        if (aeShowsParam(filters[i], 1)) drawParam(filters[i]->param1, filters[i]->param1Name());
        if (aeShowsParam(filters[i], 2)) drawParam(filters[i]->param2, filters[i]->param2Name());
        if (aeShowsParam(filters[i], 3)) drawParam(filters[i]->param3, filters[i]->param3Name());
        if (aeShowsParam(filters[i], 4)) drawParam(filters[i]->param4, filters[i]->param4Name());
        if (aeShowsParam(filters[i], 5)) drawParam(filters[i]->param5, filters[i]->param5Name());
        if (aeShowsParam(filters[i], 6)) drawParam(filters[i]->param6, filters[i]->param6Name());
        if (aeShowsParam(filters[i], 7)) drawParam(filters[i]->param7, filters[i]->param7Name());
        if (aeShowsParam(filters[i], 8)) drawParam(filters[i]->param8, filters[i]->param8Name());
        if (aeShowsParam(filters[i], 9)) drawParam(filters[i]->param9, filters[i]->param9Name());
        if (aeShowsParam(filters[i], 10)) drawParam(filters[i]->param10, filters[i]->param10Name());
        if (aeShowsParam(filters[i], 11)) drawParam(filters[i]->param11, filters[i]->param11Name());
        if (aeShowsParam(filters[i], 12)) drawParam(filters[i]->param12, filters[i]->param12Name());
        if (aeShowsParam(filters[i], 13)) drawParam(filters[i]->param13, filters[i]->param13Name());
        if (aeShowsParam(filters[i], 14)) drawParam(filters[i]->param14, filters[i]->param14Name());
        if (aeShowsParam(filters[i], 15)) drawParam(filters[i]->param15, filters[i]->param15Name());
        if (aeShowsParam(filters[i], 16)) drawParam(filters[i]->param16, filters[i]->param16Name());
        if (aeShowsParam(filters[i], 17)) drawParam(filters[i]->param17, filters[i]->param17Name());

        int tools = filters[i]->toolButtonCount();
        if (tools > 0) {
            currentY -= AE_TOOLS_PAD;
            const float btnW = (AE_SLIDER_W - AE_TOOL_GAP) * 0.5f;
            for (int t = 0; t < tools; ++t) {
                int col = t % 2;
                float bx = AE_CONTENT_X + col * (btnW + AE_TOOL_GAP);
                float by = currentY - AE_TOOL_H;
                if (inView(by, currentY)) {
                    drawButton(bx, by, btnW, AE_TOOL_H,
                               filters[i]->toolButtonName(t), false,
                               filters[i]->toolButtonActive(t));
                }
                if (col == 1 || t == tools - 1)
                    currentY -= AE_TOOL_ROW;
            }
        }

        currentY -= AE_FILTER_GAP;
    }

    // Fixed page footer — not part of tool button chrome
    if (pageBar) {
        VideoFilter* pf = aePagedFilter();
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.06f, 0.07f, 0.10f, 0.95f);
        glBegin(GL_QUADS);
        glVertex2f(AE_PANEL_X + 0.01f, AE_BOTTOM_LIMIT + 0.005f);
        glVertex2f(AE_PANEL_X + AE_PANEL_W - 0.01f, AE_BOTTOM_LIMIT + 0.005f);
        glVertex2f(AE_PANEL_X + AE_PANEL_W - 0.01f, AE_BOTTOM_LIMIT + AE_PAGE_BAR_H - 0.005f);
        glVertex2f(AE_PANEL_X + 0.01f, AE_BOTTOM_LIMIT + AE_PAGE_BAR_H - 0.005f);
        glEnd();
        glColor4f(0.45f, 0.5f, 0.6f, 0.9f);
        glBegin(GL_LINES);
        glVertex2f(AE_PANEL_X + 0.02f, AE_BOTTOM_LIMIT + AE_PAGE_BAR_H - 0.005f);
        glVertex2f(AE_PANEL_X + AE_PANEL_W - 0.02f, AE_BOTTOM_LIMIT + AE_PAGE_BAR_H - 0.005f);
        glEnd();
        glDisable(GL_BLEND);

        if (pf) {
            float midY = AE_BOTTOM_LIMIT + AE_PAGE_BAR_H * 0.38f;
            glColor3f(0.7f, 0.75f, 0.9f);
            drawText(AE_CONTENT_X, midY, "<", GLUT_BITMAP_HELVETICA_18);
            std::string page = pf->optionPageName(pf->optionPage);
            std::ostringstream label;
            label << "page  " << page << "  (" << (pf->optionPage + 1)
                  << "/" << pf->optionPageCount() << ")";
            glColor3f(0.85f, 0.88f, 1.0f);
            drawText(AE_CONTENT_X + 0.08f, midY + 0.005f, label.str(), GLUT_BITMAP_HELVETICA_12);
            glColor3f(0.7f, 0.75f, 0.9f);
            drawText(AE_CONTENT_X + 0.36f, midY, ">", GLUT_BITMAP_HELVETICA_18);
        }
    }
}

// Returns true if a slider, tool, or page control was hit.
bool hitTestActiveEffectsSliders(float fx, float fy) {
    if (!anyFilterEnabled()) return false;

    clampAePanelScroll();
    if (fx < AE_PANEL_X || fx > AE_PANEL_X + AE_PANEL_W ||
        fy < AE_BOTTOM_LIMIT || fy > AE_TOP_Y) {
        return false;
    }

    // Page footer hits (fixed)
    if (aeNeedsPageBar() && fy >= AE_BOTTOM_LIMIT && fy <= AE_BOTTOM_LIMIT + AE_PAGE_BAR_H) {
        VideoFilter* pf = aePagedFilter();
        if (pf) {
            int pages = pf->optionPageCount();
            if (fx < AE_CONTENT_X + 0.07f) {
                pf->optionPage = (pf->optionPage + pages - 1) % pages;
                return true;
            }
            if (fx > AE_CONTENT_X + 0.34f) {
                pf->optionPage = (pf->optionPage + 1) % pages;
                return true;
            }
        }
        return true;
    }

    const float listY = aeScrollBottom();
    if (fy < listY) return true;

    float currentY = AE_TOP_Y - AE_TITLE_STEP + aePanelScroll;
    for (size_t i = 0; i < filters.size(); ++i) {
        if (!filters[i]->enabled) continue;

        currentY -= AE_NAME_STEP;
        if (!filters[i]->statusLine().empty())
            currentY -= AE_STATUS_STEP;

        auto trySlider = [&](int paramType, float& value) -> bool {
            currentY -= AE_LABEL_STEP;
            float sy = currentY - AE_SLIDER_H;
            if (isInside(fx, fy, AE_CONTENT_X, sy, AE_SLIDER_W, AE_SLIDER_H)) {
                activeSlider = static_cast<int>(i) * AE_SLIDER_STRIDE + paramType;
                value = std::max(0.0f, std::min(1.0f, (fx - AE_CONTENT_X) / AE_SLIDER_W));
                return true;
            }
            currentY -= (AE_SLIDER_H + AE_SLIDER_GAP);
            return false;
        };

        if (aeShowsStrength(filters[i]) && trySlider(0, filters[i]->strength)) return true;
        if (aeShowsParam(filters[i], 1) && trySlider(1, filters[i]->param1)) return true;
        if (aeShowsParam(filters[i], 2) && trySlider(2, filters[i]->param2)) return true;
        if (aeShowsParam(filters[i], 3) && trySlider(3, filters[i]->param3)) return true;
        if (aeShowsParam(filters[i], 4) && trySlider(4, filters[i]->param4)) return true;
        if (aeShowsParam(filters[i], 5) && trySlider(5, filters[i]->param5)) return true;
        if (aeShowsParam(filters[i], 6) && trySlider(6, filters[i]->param6)) return true;
        if (aeShowsParam(filters[i], 7) && trySlider(7, filters[i]->param7)) return true;
        if (aeShowsParam(filters[i], 8) && trySlider(8, filters[i]->param8)) return true;
        if (aeShowsParam(filters[i], 9) && trySlider(9, filters[i]->param9)) return true;
        if (aeShowsParam(filters[i], 10) && trySlider(10, filters[i]->param10)) return true;
        if (aeShowsParam(filters[i], 11) && trySlider(11, filters[i]->param11)) return true;
        if (aeShowsParam(filters[i], 12) && trySlider(12, filters[i]->param12)) return true;
        if (aeShowsParam(filters[i], 13) && trySlider(13, filters[i]->param13)) return true;
        if (aeShowsParam(filters[i], 14) && trySlider(14, filters[i]->param14)) return true;
        if (aeShowsParam(filters[i], 15) && trySlider(15, filters[i]->param15)) return true;
        if (aeShowsParam(filters[i], 16) && trySlider(16, filters[i]->param16)) return true;
        if (aeShowsParam(filters[i], 17) && trySlider(17, filters[i]->param17)) return true;

        int tools = filters[i]->toolButtonCount();
        if (tools > 0) {
            currentY -= AE_TOOLS_PAD;
            const float btnW = (AE_SLIDER_W - AE_TOOL_GAP) * 0.5f;
            for (int t = 0; t < tools; ++t) {
                int col = t % 2;
                float bx = AE_CONTENT_X + col * (btnW + AE_TOOL_GAP);
                float by = currentY - AE_TOOL_H;
                if (isInside(fx, fy, bx, by, btnW, AE_TOOL_H)) {
                    filters[i]->toolButtonClick(t);
                    return true;
                }
                if (col == 1 || t == tools - 1)
                    currentY -= AE_TOOL_ROW;
            }
        }

        currentY -= AE_FILTER_GAP;
    }
    return true;
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
bool hitTestFormatPanel(float fx, float fy);
bool hitTestCapturePanel(float fx, float fy);
bool hitTestFilterListPanel(float fx, float fy);

void mouse(int button, int state, int x, int y) {
    if (!showUI || windowWidth < 1 || windowHeight < 1) return;

    float fx = (float)x / windowWidth * 2.0f - 1.0f;
    float fy = 1.0f - (float)y / windowHeight * 2.0f;

    // Right-click toggles pie menu (also closes if already open)
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        if (sourceMenuMode != SourceMenuMode::Closed) {
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

    // Mouse wheel over filter list / active effects (FreeGLUT: button 3=up, 4=down)
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
        if (anyFilterEnabled() &&
            fx >= AE_PANEL_X && fx <= AE_PANEL_X + AE_PANEL_W &&
            fy >= AE_BOTTOM_LIMIT && fy <= AE_TOP_Y) {
            clampAePanelScroll();
            if (button == 3)
                aePanelScroll = std::max(0.0f, aePanelScroll - AE_PARAM_STEP);
            else
                aePanelScroll += AE_PARAM_STEP;
            clampAePanelScroll();
            return;
        }
    }

    if (button != GLUT_LEFT_BUTTON) return;

    if (button == GLUT_LEFT_BUTTON) {
        mouseLeftDown = (state == GLUT_DOWN);

        if (state == GLUT_DOWN) {
            // Source menus capture input while open
            if (sourceMenuMode != SourceMenuMode::Closed) {
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

            // Check main control buttons (positions match displayControlWindow)
            if (isInside(fx, fy, -0.95f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                isPaused = !isPaused;
                return;
            }
            else if (isInside(fx, fy, -0.78f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                seekVideo(std::max(0.0, currentVideoTime - 5.0));
                return;
            }
            else if (isInside(fx, fy, -0.61f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                seekVideo(std::min(videoDuration, currentVideoTime + 5.0));
                return;
            }
            else if (isInside(fx, fy, -0.44f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                isLooping = !isLooping;
                return;
            }
            else if (isInside(fx, fy, -0.27f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                if (!playlist.empty())
                    jumpToPlaylistIndex((currentVideoIndex + 1) % static_cast<int>(playlist.size()));
                return;
            }
            else if (isInside(fx, fy, -0.10f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                openFileBrowserMenu();
                return;
            }
            else if (isInside(fx, fy, 0.07f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                openQueueMenu();
                return;
            }
            else if (isInside(fx, fy, 0.24f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                openCameraMenu();
                return;
            }
            else if (isInside(fx, fy, 0.41f, -0.85f, 0.12f * uiScale, 0.08f * uiScale)) {
                openWindowMenu();
                return;
            }
            else if (isInside(fx, fy, 0.58f, -0.85f, 0.13f * uiScale, 0.08f * uiScale)) {
                setVideoFullscreen(!videoFullscreen);
                return;
            }
            else if (isInside(fx, fy, 0.75f, -0.85f, 0.13f * uiScale, 0.08f * uiScale)) {
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
            if (hitTestFormatPanel(fx, fy)) {
                return;
            }
            if (hitTestCapturePanel(fx, fy)) {
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
                        activeSlider = selected * AE_SLIDER_STRIDE;
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
                if (fx >= AE_PANEL_X && fx <= AE_PANEL_X + AE_PANEL_W &&
                    fy >= AE_BOTTOM_LIMIT && fy <= AE_TOP_Y) {
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
            activeCapSlider = -1;
            activeFormatSlider = -1;
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

    if (activeCapSlider >= 0 && mouseLeftDown) {
        float left = (activeCapSlider == 1 || activeCapSlider == 3 || activeCapSlider == 5)
            ? (CAP_PANEL_X + 0.28f) : (CAP_PANEL_X + 0.02f);
        float norm = std::max(0.0f, std::min(1.0f, (fx - left) / 0.22f));
        switch (activeCapSlider) {
            case 0: capCropL = std::min(0.45f, norm); break;
            case 1: capCropR = std::min(0.45f, norm); break;
            case 2: capCropT = std::min(0.45f, norm); break;
            case 3: capCropB = std::min(0.45f, norm); break;
            case 4: capDownscale = std::max(0.25f, norm); break;
            case 5: capFpsNorm = norm; break;
        }
        glutPostRedisplay();
        return;
    }

    if (activeFormatSlider >= 0 && mouseLeftDown) {
        const float panelW = FILT_PANEL_W;
        if (activeFormatSlider == 0) {
            formatAspectNorm = std::max(0.0f, std::min(1.0f,
                (fx - (FORMAT_PANEL_X + 0.02f)) / (panelW - 0.04f)));
            formatUseCustom = true;
        } else if (activeFormatSlider == 1) {
            forceResWNorm = std::max(0.0f, std::min(1.0f, (fx - (FORMAT_PANEL_X + 0.02f)) / 0.20f));
            forceResPreset = -2;
        } else if (activeFormatSlider == 2) {
            forceResHNorm = std::max(0.0f, std::min(1.0f, (fx - (FORMAT_PANEL_X + 0.24f)) / 0.20f));
            forceResPreset = -2;
        }
        glutPostRedisplay();
        return;
    }

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

    int filterIdx = activeSlider / AE_SLIDER_STRIDE;
    int paramType = activeSlider % AE_SLIDER_STRIDE;
    if (filterIdx < 0 || filterIdx >= static_cast<int>(filters.size())) return;

    float normalizedValue = std::max(0.0f, std::min(1.0f, (fx - AE_CONTENT_X) / AE_SLIDER_W));
    switch (paramType) {
        case 0: filters[filterIdx]->strength = normalizedValue; break;
        case 1: filters[filterIdx]->param1 = normalizedValue; break;
        case 2: filters[filterIdx]->param2 = normalizedValue; break;
        case 3: filters[filterIdx]->param3 = normalizedValue; break;
        case 4: filters[filterIdx]->param4 = normalizedValue; break;
        case 5: filters[filterIdx]->param5 = normalizedValue; break;
        case 6: filters[filterIdx]->param6 = normalizedValue; break;
        case 7: filters[filterIdx]->param7 = normalizedValue; break;
        case 8: filters[filterIdx]->param8 = normalizedValue; break;
        case 9: filters[filterIdx]->param9 = normalizedValue; break;
        case 10: filters[filterIdx]->param10 = normalizedValue; break;
        case 11: filters[filterIdx]->param11 = normalizedValue; break;
        case 12: filters[filterIdx]->param12 = normalizedValue; break;
        case 13: filters[filterIdx]->param13 = normalizedValue; break;
        case 14: filters[filterIdx]->param14 = normalizedValue; break;
        case 15: filters[filterIdx]->param15 = normalizedValue; break;
        case 16: filters[filterIdx]->param16 = normalizedValue; break;
        case 17: filters[filterIdx]->param17 = normalizedValue; break;
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
            if (sourceMenuMode != SourceMenuMode::Closed) {
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
            if (!playlist.empty())
                jumpToPlaylistIndex((currentVideoIndex + 1) % static_cast<int>(playlist.size()));
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
        case 'q':
        case 'Q':
            openQueueMenu();
            break;
        case 'c':
        case 'C':
            openCameraMenu();
            break;
        case 'i':
        case 'I':
            openWindowMenu();
            break;
        case 'r':
        case 'R':
            if (sourceMenuMode == SourceMenuMode::Files) {
                refreshBrowserEntries();
            } else if (sourceMenuMode == SourceMenuMode::Cameras) {
                scanCameras();
            } else if (sourceMenuMode == SourceMenuMode::Windows) {
                scanCaptureWindows();
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
                int filterIdx = activeSlider / AE_SLIDER_STRIDE;
                int paramType = activeSlider % AE_SLIDER_STRIDE;
                float step = 0.05f;
                switch (paramType) {
                    case 0: filters[filterIdx]->strength = std::max(0.0f, filters[filterIdx]->strength - step); break;
                    case 1: filters[filterIdx]->param1 = std::max(0.0f, filters[filterIdx]->param1 - step); break;
                    case 2: filters[filterIdx]->param2 = std::max(0.0f, filters[filterIdx]->param2 - step); break;
                    case 3: filters[filterIdx]->param3 = std::max(0.0f, filters[filterIdx]->param3 - step); break;
                    case 4: filters[filterIdx]->param4 = std::max(0.0f, filters[filterIdx]->param4 - step); break;
                    case 5: filters[filterIdx]->param5 = std::max(0.0f, filters[filterIdx]->param5 - step); break;
                    case 6: filters[filterIdx]->param6 = std::max(0.0f, filters[filterIdx]->param6 - step); break;
                    case 7: filters[filterIdx]->param7 = std::max(0.0f, filters[filterIdx]->param7 - step); break;
                    case 8: filters[filterIdx]->param8 = std::max(0.0f, filters[filterIdx]->param8 - step); break;
                    case 9: filters[filterIdx]->param9 = std::max(0.0f, filters[filterIdx]->param9 - step); break;
                    case 10: filters[filterIdx]->param10 = std::max(0.0f, filters[filterIdx]->param10 - step); break;
                    case 11: filters[filterIdx]->param11 = std::max(0.0f, filters[filterIdx]->param11 - step); break;
                    case 12: filters[filterIdx]->param12 = std::max(0.0f, filters[filterIdx]->param12 - step); break;
                    case 13: filters[filterIdx]->param13 = std::max(0.0f, filters[filterIdx]->param13 - step); break;
                    case 14: filters[filterIdx]->param14 = std::max(0.0f, filters[filterIdx]->param14 - step); break;
                    case 15: filters[filterIdx]->param15 = std::max(0.0f, filters[filterIdx]->param15 - step); break;
                    case 16: filters[filterIdx]->param16 = std::max(0.0f, filters[filterIdx]->param16 - step); break;
                    case 17: filters[filterIdx]->param17 = std::max(0.0f, filters[filterIdx]->param17 - step); break;
                }
            } else {
                videoScale = std::max(VIDEO_SCALE_MIN, videoScale - 0.05f);
            }
            break;
        case '=':
        case '+':
            if (activeSlider >= 0) {
                int filterIdx = activeSlider / AE_SLIDER_STRIDE;
                int paramType = activeSlider % AE_SLIDER_STRIDE;
                float step = 0.05f;
                switch (paramType) {
                    case 0: filters[filterIdx]->strength = std::min(1.0f, filters[filterIdx]->strength + step); break;
                    case 1: filters[filterIdx]->param1 = std::min(1.0f, filters[filterIdx]->param1 + step); break;
                    case 2: filters[filterIdx]->param2 = std::min(1.0f, filters[filterIdx]->param2 + step); break;
                    case 3: filters[filterIdx]->param3 = std::min(1.0f, filters[filterIdx]->param3 + step); break;
                    case 4: filters[filterIdx]->param4 = std::min(1.0f, filters[filterIdx]->param4 + step); break;
                    case 5: filters[filterIdx]->param5 = std::min(1.0f, filters[filterIdx]->param5 + step); break;
                    case 6: filters[filterIdx]->param6 = std::min(1.0f, filters[filterIdx]->param6 + step); break;
                    case 7: filters[filterIdx]->param7 = std::min(1.0f, filters[filterIdx]->param7 + step); break;
                    case 8: filters[filterIdx]->param8 = std::min(1.0f, filters[filterIdx]->param8 + step); break;
                    case 9: filters[filterIdx]->param9 = std::min(1.0f, filters[filterIdx]->param9 + step); break;
                    case 10: filters[filterIdx]->param10 = std::min(1.0f, filters[filterIdx]->param10 + step); break;
                    case 11: filters[filterIdx]->param11 = std::min(1.0f, filters[filterIdx]->param11 + step); break;
                    case 12: filters[filterIdx]->param12 = std::min(1.0f, filters[filterIdx]->param12 + step); break;
                    case 13: filters[filterIdx]->param13 = std::min(1.0f, filters[filterIdx]->param13 + step); break;
                    case 14: filters[filterIdx]->param14 = std::min(1.0f, filters[filterIdx]->param14 + step); break;
                    case 15: filters[filterIdx]->param15 = std::min(1.0f, filters[filterIdx]->param15 + step); break;
                    case 16: filters[filterIdx]->param16 = std::min(1.0f, filters[filterIdx]->param16 + step); break;
                    case 17: filters[filterIdx]->param17 = std::min(1.0f, filters[filterIdx]->param17 + step); break;
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
                activeSlider = (key - '1') * AE_SLIDER_STRIDE; // Activate strength slider
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
    if (!rgb.isContinuous())
        rgb = rgb.clone();

    float frameAspect = static_cast<float>(rgb.cols) / static_cast<float>(rgb.rows);
    float windowAspect = static_cast<float>(videoWinW) / static_cast<float>(videoWinH);

    // Fit-contain size, then apply user scale (GPU scales via textured quad)
    float fitW, fitH;
    if (frameAspect > windowAspect) {
        fitW = static_cast<float>(videoWinW);
        fitH = fitW / frameAspect;
    } else {
        fitH = static_cast<float>(videoWinH);
        fitW = fitH * frameAspect;
    }

    float drawWidth = std::max(1.0f, fitW * videoScale);
    float drawHeight = std::max(1.0f, fitH * videoScale);
    float offsetX = (static_cast<float>(videoWinW) - drawWidth) * 0.5f
                  + videoOffsetX * static_cast<float>(videoWinW) * 0.5f;
    float offsetY = (static_cast<float>(videoWinH) - drawHeight) * 0.5f
                  + videoOffsetY * static_cast<float>(videoWinH) * 0.5f;

    if (!videoTexId)
        glGenTextures(1, &videoTexId);
    glBindTexture(GL_TEXTURE_2D, videoTexId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (videoTexW != rgb.cols || videoTexH != rgb.rows) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        videoTexW = rgb.cols;
        videoTexH = rgb.rows;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgb.cols, rgb.rows,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    }

    glViewport(0, 0, videoWinW, videoWinH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(videoWinW), 0.0, static_cast<double>(videoWinH));
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    // Frame was vertically flipped for legacy glDrawPixels; tex v=0 is image bottom.
    glTexCoord2f(0.0f, 0.0f); glVertex2f(offsetX, offsetY);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(offsetX + drawWidth, offsetY);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(offsetX + drawWidth, offsetY + drawHeight);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(offsetX, offsetY + drawHeight);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

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
               guideAspect43 || guideAspect239 ||
               (formatShowBorder && formatAspectActive());
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
    if (formatShowBorder && formatAspectActive()) {
        glColor4f(1.0f, 0.85f, 0.35f, 0.95f);
        glLineWidth(2.0f);
        drawAspectGuide(currentFormatAspect());
        glLineWidth(1.0f);
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
    if (windowCaptureActive) {
        double now = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
        double minDt = 1.0 / static_cast<double>(captureTargetFps());
        if (lastCapGrabTime > 0.0 && (now - lastCapGrabTime) < minDt)
            return;
        if (!grabX11WindowFrame(captureXid, frame)) {
            return;
        }
        lastCapGrabTime = now;
        frame = applyCaptureTune(frame);
        currentVideoTime += deltaTime;
    } else {
        cap >> frame;
        if (frame.empty()) {
            if (isLooping) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            } else if (!playlist.empty()) {
                jumpToPlaylistIndex((currentVideoIndex + 1) % static_cast<int>(playlist.size()));
            }
            return;
        }
        currentVideoTime = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    }

    for (auto& filter : filters)
        frame = filter->apply(frame);
    frame = applyFormatAspect(frame);
    frame = applyForcedResolution(frame);
    cv::flip(frame, frame, 0);
    latestFrame = frame;
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

void drawFormatPanel() {
    const float panelH = FORMAT_PANEL_H;
    const float panelW = FILT_PANEL_W;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.08f, 0.12f, 0.85f);
    glBegin(GL_QUADS);
    glVertex2f(FORMAT_PANEL_X, FORMAT_PANEL_Y - panelH);
    glVertex2f(FORMAT_PANEL_X + panelW, FORMAT_PANEL_Y - panelH);
    glVertex2f(FORMAT_PANEL_X + panelW, FORMAT_PANEL_Y + 0.04f);
    glVertex2f(FORMAT_PANEL_X, FORMAT_PANEL_Y + 0.04f);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1, 1, 1);
    drawText(FORMAT_PANEL_X + 0.02f, FORMAT_PANEL_Y + 0.01f, "Aspect / Resolution", GLUT_BITMAP_HELVETICA_12);

    drawButton(FORMAT_PANEL_X + 0.02f, FORMAT_PANEL_Y - 0.05f, 0.08f, 0.040f, "Off", false, !formatAspectActive());
    drawButton(FORMAT_PANEL_X + 0.11f, FORMAT_PANEL_Y - 0.05f, 0.09f, 0.040f, "Crop", false, formatMode == 1);
    drawButton(FORMAT_PANEL_X + 0.21f, FORMAT_PANEL_Y - 0.05f, 0.10f, 0.040f, "Letter", false, formatMode == 0);
    drawButton(FORMAT_PANEL_X + 0.32f, FORMAT_PANEL_Y - 0.05f, 0.07f, 0.040f, "Bord", false, formatShowBorder);
    drawButton(FORMAT_PANEL_X + 0.40f, FORMAT_PANEL_Y - 0.05f, 0.06f, 0.040f, "Up");

    formatListScroll = std::max(0, std::min(formatListScroll,
        std::max(0, kAspectFormatCount - FORMAT_LIST_ROWS)));

    const float rowH = 0.048f;
    const float listTop = FORMAT_PANEL_Y - 0.10f;
    for (int i = 0; i < FORMAT_LIST_ROWS; ++i) {
        int idx = formatListScroll + i;
        if (idx >= kAspectFormatCount) break;
        float y = listTop - i * rowH;
        bool on = (!formatUseCustom && idx == formatPresetIndex);
        drawButton(FORMAT_PANEL_X + 0.02f, y - 0.036f, panelW - 0.04f, 0.040f,
                   kAspectFormats[idx].name, false, on);
    }

    const float sliderY = FORMAT_PANEL_Y - 0.32f;
    drawSlider(FORMAT_PANEL_X + 0.02f, sliderY, panelW - 0.04f, 0.018f, formatAspectNorm, "Aspect");
    char aspectBuf[48];
    std::snprintf(aspectBuf, sizeof(aspectBuf), "%.2f%s", currentFormatAspect(),
                  formatUseCustom ? " custom" : (formatAspectActive() ? "" : " (off)"));
    glColor3f(0.75f, 0.85f, 1.0f);
    drawText(FORMAT_PANEL_X + 0.02f, sliderY - 0.035f, aspectBuf, GLUT_BITMAP_HELVETICA_10);

    const float resY = FORMAT_PANEL_Y - 0.42f;
    drawButton(FORMAT_PANEL_X + 0.02f, resY, 0.07f, 0.038f, "Def", false, forceResPreset == -1);
    drawButton(FORMAT_PANEL_X + 0.10f, resY, 0.07f, 0.038f, "720", false, forceResPreset == 1);
    drawButton(FORMAT_PANEL_X + 0.18f, resY, 0.08f, 0.038f, "1080", false, forceResPreset == 2);
    drawButton(FORMAT_PANEL_X + 0.27f, resY, 0.08f, 0.038f, "1440", false, forceResPreset == 3);
    drawButton(FORMAT_PANEL_X + 0.36f, resY, 0.06f, 0.038f, "4K", false, forceResPreset == 4);
    drawButton(FORMAT_PANEL_X + 0.02f, resY - 0.05f, 0.07f, 0.038f, "480", false, forceResPreset == 0);
    drawButton(FORMAT_PANEL_X + 0.10f, resY - 0.05f, 0.08f, 0.038f, "Sq1K", false, forceResPreset == 5);
    drawButton(FORMAT_PANEL_X + 0.19f, resY - 0.05f, 0.10f, 0.038f, "Custom", false, forceResPreset == -2);

    drawSlider(FORMAT_PANEL_X + 0.02f, resY - 0.12f, 0.20f, 0.018f, forceResWNorm, "Res W");
    drawSlider(FORMAT_PANEL_X + 0.24f, resY - 0.12f, 0.20f, 0.018f, forceResHNorm, "Res H");

    char resBuf[64];
    if (forceResActive()) {
        std::snprintf(resBuf, sizeof(resBuf), "Out %dx%d", forceResWidth(), forceResHeight());
    } else {
        std::snprintf(resBuf, sizeof(resBuf), "Out default");
    }
    glColor3f(0.75f, 0.85f, 1.0f);
    drawText(FORMAT_PANEL_X + 0.02f, resY - 0.16f, resBuf, GLUT_BITMAP_HELVETICA_10);

    drawButton(FORMAT_PANEL_X + 0.02f, FORMAT_PANEL_Y - panelH + 0.02f, 0.21f, 0.040f, "Prev");
    drawButton(FORMAT_PANEL_X + 0.25f, FORMAT_PANEL_Y - panelH + 0.02f, 0.21f, 0.040f, "Next");
}

bool hitTestFormatPanel(float fx, float fy) {
    const float panelH = FORMAT_PANEL_H;
    const float panelW = FILT_PANEL_W;
    if (fx < FORMAT_PANEL_X || fx > FORMAT_PANEL_X + panelW ||
        fy < FORMAT_PANEL_Y - panelH || fy > FORMAT_PANEL_Y + 0.04f)
        return false;

    if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, FORMAT_PANEL_Y - 0.05f, 0.08f, 0.040f)) {
        formatPresetIndex = -1;
        formatUseCustom = false;
        return true;
    }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.11f, FORMAT_PANEL_Y - 0.05f, 0.09f, 0.040f)) {
        formatMode = 1;
        return true;
    }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.21f, FORMAT_PANEL_Y - 0.05f, 0.10f, 0.040f)) {
        formatMode = 0;
        return true;
    }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.32f, FORMAT_PANEL_Y - 0.05f, 0.07f, 0.040f)) {
        formatShowBorder = !formatShowBorder;
        return true;
    }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.40f, FORMAT_PANEL_Y - 0.05f, 0.06f, 0.040f)) {
        formatListScroll = std::max(0, formatListScroll - FORMAT_LIST_ROWS);
        return true;
    }

    const float rowH = 0.048f;
    const float listTop = FORMAT_PANEL_Y - 0.10f;
    for (int i = 0; i < FORMAT_LIST_ROWS; ++i) {
        int idx = formatListScroll + i;
        if (idx >= kAspectFormatCount) break;
        float y = listTop - i * rowH;
        if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, y - 0.036f, panelW - 0.04f, 0.040f)) {
            formatPresetIndex = idx;
            formatUseCustom = false;
            syncFormatAspectNormFromPreset();
            return true;
        }
    }

    const float sliderY = FORMAT_PANEL_Y - 0.32f;
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, sliderY, panelW - 0.04f, 0.018f)) {
        activeFormatSlider = 0;
        formatAspectNorm = std::max(0.0f, std::min(1.0f, (fx - (FORMAT_PANEL_X + 0.02f)) / (panelW - 0.04f)));
        formatUseCustom = true;
        return true;
    }

    const float resY = FORMAT_PANEL_Y - 0.42f;
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, resY, 0.07f, 0.038f)) { forceResPreset = -1; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.10f, resY, 0.07f, 0.038f)) { forceResPreset = 1; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.18f, resY, 0.08f, 0.038f)) { forceResPreset = 2; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.27f, resY, 0.08f, 0.038f)) { forceResPreset = 3; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.36f, resY, 0.06f, 0.038f)) { forceResPreset = 4; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, resY - 0.05f, 0.07f, 0.038f)) { forceResPreset = 0; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.10f, resY - 0.05f, 0.08f, 0.038f)) { forceResPreset = 5; return true; }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.19f, resY - 0.05f, 0.10f, 0.038f)) { forceResPreset = -2; return true; }

    if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, resY - 0.12f, 0.20f, 0.018f)) {
        activeFormatSlider = 1;
        forceResWNorm = std::max(0.0f, std::min(1.0f, (fx - (FORMAT_PANEL_X + 0.02f)) / 0.20f));
        forceResPreset = -2;
        return true;
    }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.24f, resY - 0.12f, 0.20f, 0.018f)) {
        activeFormatSlider = 2;
        forceResHNorm = std::max(0.0f, std::min(1.0f, (fx - (FORMAT_PANEL_X + 0.24f)) / 0.20f));
        forceResPreset = -2;
        return true;
    }

    if (isInside(fx, fy, FORMAT_PANEL_X + 0.02f, FORMAT_PANEL_Y - panelH + 0.02f, 0.21f, 0.040f)) {
        formatListScroll = std::max(0, formatListScroll - FORMAT_LIST_ROWS);
        return true;
    }
    if (isInside(fx, fy, FORMAT_PANEL_X + 0.25f, FORMAT_PANEL_Y - panelH + 0.02f, 0.21f, 0.040f)) {
        int maxScroll = std::max(0, kAspectFormatCount - FORMAT_LIST_ROWS);
        formatListScroll = std::min(maxScroll, formatListScroll + FORMAT_LIST_ROWS);
        return true;
    }
    return true;
}

void drawCapturePanel() {
    if (!windowCaptureActive) return;
    const float panelH = 0.30f;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.08f, 0.10f, 0.14f, 0.90f);
    glBegin(GL_QUADS);
    glVertex2f(CAP_PANEL_X, CAP_PANEL_Y - panelH);
    glVertex2f(CAP_PANEL_X + 0.52f, CAP_PANEL_Y - panelH);
    glVertex2f(CAP_PANEL_X + 0.52f, CAP_PANEL_Y + 0.03f);
    glVertex2f(CAP_PANEL_X, CAP_PANEL_Y + 0.03f);
    glEnd();
    glDisable(GL_BLEND);

    glColor3f(1, 1, 1);
    drawText(CAP_PANEL_X + 0.02f, CAP_PANEL_Y + 0.005f, "Window Capture Tune", GLUT_BITMAP_HELVETICA_12);

    drawSlider(CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.05f, 0.22f, 0.020f, capCropL, "Crop L");
    drawSlider(CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.05f, 0.22f, 0.020f, capCropR, "Crop R");
    drawSlider(CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.12f, 0.22f, 0.020f, capCropT, "Crop T");
    drawSlider(CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.12f, 0.22f, 0.020f, capCropB, "Crop B");
    drawSlider(CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.19f, 0.22f, 0.020f, capDownscale, "Scale");
    drawSlider(CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.19f, 0.22f, 0.020f, capFpsNorm, "Cap FPS");

    char fpsBuf[48];
    std::snprintf(fpsBuf, sizeof(fpsBuf), "~%.0f fps", captureTargetFps());
    glColor3f(0.75f, 0.85f, 1.0f);
    drawText(CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.24f, fpsBuf, GLUT_BITMAP_HELVETICA_10);
    drawButton(CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.28f, 0.22f, 0.045f, "Reset Cap");
}

bool hitTestCapturePanel(float fx, float fy) {
    if (!windowCaptureActive) return false;
    const float panelH = 0.30f;
    if (fx < CAP_PANEL_X || fx > CAP_PANEL_X + 0.52f ||
        fy < CAP_PANEL_Y - panelH || fy > CAP_PANEL_Y + 0.03f)
        return false;

    auto tryS = [&](int id, float x, float y, float& val) -> bool {
        if (isInside(fx, fy, x, y, 0.22f, 0.020f)) {
            activeCapSlider = id;
            val = std::max(0.0f, std::min(1.0f, (fx - x) / 0.22f));
            if (id <= 3) val = std::min(0.45f, val);
            if (id == 4) val = std::max(0.25f, val);
            return true;
        }
        return false;
    };
    if (tryS(0, CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.05f, capCropL)) return true;
    if (tryS(1, CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.05f, capCropR)) return true;
    if (tryS(2, CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.12f, capCropT)) return true;
    if (tryS(3, CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.12f, capCropB)) return true;
    if (tryS(4, CAP_PANEL_X + 0.02f, CAP_PANEL_Y - 0.19f, capDownscale)) return true;
    if (tryS(5, CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.19f, capFpsNorm)) return true;
    if (isInside(fx, fy, CAP_PANEL_X + 0.28f, CAP_PANEL_Y - 0.28f, 0.22f, 0.045f)) {
        resetCaptureTune();
        return true;
    }
    return true;
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

        drawButton(-0.95f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, isPaused ? ">" : "||");
        drawButton(-0.78f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, "<<");
        drawButton(-0.61f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, ">>");
        drawButton(-0.44f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, isLooping ? "Loop" : "NoLoop");
        drawButton(-0.27f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, "Next");
        drawButton(-0.10f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, "Open");
        drawButton(0.07f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, "Que");
        drawButton(0.24f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, "Cam");
        drawButton(0.41f, -0.85f, 0.12f * uiScale, 0.08f * uiScale, "Win");
        drawButton(0.58f, -0.85f, 0.13f * uiScale, 0.08f * uiScale, "Vid FS", false, videoFullscreen);
        drawButton(0.75f, -0.85f, 0.13f * uiScale, 0.08f * uiScale, "Ctl FS", false, controlFullscreen);

        drawFilterListPanel();

        drawViewControls();
        drawCapturePanel();
        drawPlacementPanel();
        drawGuidesPanel();
        drawFormatPanel();
        drawActiveEffectsPanel();

        drawText(-0.95f, 0.95f, "F/H FS | V Open | Q Queue | C Cam | I Win | AE: <Pg Pg>");
        {
            std::ostringstream perf;
            perf << "Decode " << hwDecodeStatus
                 << " | CL " << (cv::ocl::useOpenCL() ? "on" : "off");
            if (formatAspectActive() || forceResActive()) {
                perf << " | ";
                if (formatAspectActive()) {
                    if (formatUseCustom)
                        perf << "asp " << std::fixed << std::setprecision(2) << currentFormatAspect();
                    else if (formatPresetIndex >= 0)
                        perf << kAspectFormats[formatPresetIndex].name;
                }
                if (forceResActive())
                    perf << " " << forceResWidth() << "x" << forceResHeight();
            }
            if (windowCaptureActive && !captureWindowTitle.empty())
                perf << " | " << truncateLabel(captureWindowTitle, 28);
            drawText(-0.95f, 0.90f, perf.str());
        }
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
            videoDragMode = VideoDragMode::Idle;
            syncVideoGeomFromWindow();
        }
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        // right-click video toggles crosshair quickly
        guideCrosshair = !guideCrosshair;
    }
}

void motionVideo(int x, int y) {
    if (videoDragMode == VideoDragMode::Idle || videoFullscreen) return;
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
    if (windowCaptureActive) return;
    if (cap.isOpened() && !isPaused) {
        currentVideoTime = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    }
}

int main(int argc, char** argv) {
    XInitThreads();
    appRootPath = resolveAppRoot(argv[0] ? argv[0] : "");
    browserPath = appRootPath;
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
        else if (arg == "--no-hw-decode")
            useHwDecode = false;
        else if (arg == "--hw-decode")
            useHwDecode = true;
        else if (arg == "--no-opencl")
            useOpenCL = false;
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
        } else if (arg == "--queue" || arg == "--playlist") {
            // Consume following non-option args into the playback queue
            while (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (next.rfind("--", 0) == 0) break;
                playlist.push_back(argv[++i]);
            }
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

    // Resolve relative media paths (cwd first, then app executable folder)
    for (auto& item : playlist) {
        if (item.empty() || isCameraSourceId(item) ||
            item.rfind("win:", 0) == 0 || item == "screen")
            continue;
        std::error_code ec;
        fs::path p(item);
        if (p.is_absolute()) {
            item = fs::weakly_canonical(p, ec).string();
            if (ec) item = p.string();
            continue;
        }
        if (fs::exists(p, ec)) {
            item = fs::weakly_canonical(fs::absolute(p), ec).string();
            if (ec) item = fs::absolute(p).string();
            continue;
        }
        fs::path fromApp = fs::path(appRootPath) / p;
        if (fs::exists(fromApp, ec)) {
            item = fs::weakly_canonical(fromApp, ec).string();
            if (ec) item = fromApp.string();
        }
    }

    if (playlist.empty()) playlist.push_back("0");

    if (useOpenCL && cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
    } else {
        cv::ocl::setUseOpenCL(false);
        useOpenCL = false;
    }

    // Initialize video capture
    if (!openVideoSource(playlist[0])) {
        std::cerr << "Failed to open source." << std::endl;
        return -1;
    }
    currentVideoIndex = 0;
    std::cerr << "Decode: " << hwDecodeStatus
              << " | OpenCL: " << (cv::ocl::useOpenCL() ? "on" : "off")
              << " | display: GL texture" << std::endl;

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

    auto motionPop = std::make_unique<MotionPopAnaglyphFilter>();
    motionPop->enabled = enabled.count("motionpop") || enabled.count("motion-pop") ||
                         enabled.count("motion3d");
    // Default mid sensitivity / mild pop when enabled from CLI without tweaks
    motionPop->param1 = 0.55f;
    motionPop->param2 = 0.45f;
    filters.push_back(std::move(motionPop));

    auto vision = std::make_unique<EnceladusVisionFilter>();
    vision->enabled = enabled.count("enceladusvision") || enabled.count("vision") ||
                      enabled.count("wavepop") || enabled.count("wave-pop") ||
                      enabled.count("wave3d");
    vision->applyDefaultParams();
    filters.push_back(std::move(vision));

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
