# Enceladus Video Patcher

## Overview

Enceladus Video Patcher is a realtime video processing application that applies various visual effects to video streams or files. Built with OpenCV and OpenGL.

## Features

- **Real-time video processing** with multiple simultaneous filters
- **15+ visual effects** including:
  - Anaglyph 3D
  - Grayscale
  - Edge Detection
  - Sepia
  - Pixelation
  - Inversion
  - Blur
  - Vignette
  - Noise
  - Color Balance
  - Scanlines
  - CRT Effect
  - Glitch
  - Kaleidoscope
  - Night Vision
  - Colorize
- **Interactive UI** with:
  - Playback controls (play/pause, seek, fast forward/rewind)
  - Filter toggles and parameter adjustment
  - Pie menu for quick filter selection
  - Fullscreen mode
- **Keyboard shortcuts** for quick control
- **Playlist support** for multiple video files
- **Adjustable playback speed**

## Requirements

- OpenCV (built with videoio module)
- OpenGL/GLUT
- C++17 compatible compiler

## Installation

1. Ensure you have the required dependencies installed:
   ```
   sudo apt-get install build-essential libopencv-dev freeglut3-dev
   ```

2. Clone the repository or download the source files

3. Compile the application:
   ```
   g++ -std=c++17 -o enceladus evp_dev.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lGL -lGLU -lglut
   ```

## Usage

```
./enceladus [video_file...] [options]
```

### Options

- `--enable-[filter]`: Enable specific filter at startup (e.g., `--enable-glitch`)
- `--no-ui`: Start with UI hidden
- `--fullscreen`: Start in fullscreen mode

### Controls

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| L | Toggle looping |
| N | Next video in playlist |
| F | Toggle fullscreen |
| U | Toggle UI visibility |
| 1-9 | Toggle corresponding filter |
| -/= | Decrease/increase active slider value |
| </> | Seek backward/forward 5 seconds |
| [/] | Decrease/increase playback speed |
| ESC | Exit fullscreen or quit |

#### Mouse Controls

- Click progress bar to seek
- Click filter buttons to toggle effects
- Drag sliders to adjust parameters
- Right-click to open pie menu for filter selection

## Configuration

Filters can be enabled/disabled and adjusted in real-time through the UI. The application remembers the last used settings for each filter during the session.

## Arguments
```
--enable-anaglyph
--enable-gray
--enable-edge
--enable-sepia
--enable-pixelate
--enable-invert
--enable-blur
--enable-vignette
--enable-noise
--enable-colorbalance
--enable-scanlines
--enable-crt
--enable-glitch
--enable-kaleidoscope
--enable-nightvision
--enable-colorize

--no-ui          // Disables the user interface
--fullscreen     // Enables fullscreen mode
```

### Example

```
enceladus --enable-gray --enable-sepia myvideo.mp4
enceladus --no-ui --enable-blur --enable-vignette video1.avi video2.mov
```

## Known Issues

- Some filters may impact performance on lower-end systems
- Seeking may not be frame-accurate with certain video formats
- Fullscreen mode may behave differently across platforms

## Future Improvements

- Save/Load filter presets
- Screenshot capture functionality
- Additional filter effects
- Improved performance optimization

## License

This project is licensed under the GPL v3, (please submit patches upstream)

## Acknowledgments

- OpenCV team for the excellent computer vision library
- FreeGLUT developers for the OpenGL utility toolkit
- All contributors and testers
