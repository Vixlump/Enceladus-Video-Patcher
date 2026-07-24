# Enceladus Video Patcher

## Overview

Enceladus Video Patcher is a realtime video processing application that applies visual effects to video streams or files. Built with OpenCV and OpenGL/FreeGLUT.

The app runs as **two windows**:
- **Enceladus Video** — video/camera stream only (plus optional calibration guides)
- **Enceladus Control Panel** — playback, filters, sources, placement, and guides

## Features

- Real-time video processing with multiple simultaneous filters
- 15+ visual effects (anaglyph, grayscale, edge, sepia, pixelate, invert, blur, vignette, noise, color balance, scanlines, CRT, glitch, kaleidoscope, night vision, colorize)
- Dual-window control panel + clean output window
- File browser and webcam picker
- Video-window placement (sliders, presets, mouse drag move/resize)
- Theater calibration guides (crosshair, edge, thirds, grid, safe areas, aspect frames)
- Scriptable geometry and guide flags via CLI
- Playlist support and adjustable playback speed

## Requirements

- OpenCV (videoio module)
- OpenGL / FreeGLUT
- C++17 compiler

Primary OS: Ubuntu. Other platforms may work with matching dependencies.

## Installation

```bash
sudo apt-get install build-essential libopencv-dev freeglut3-dev
./compile.sh
# or:
g++ -std=c++17 -o Enceladus_Video_Patcher evp_dev.cpp \
  `pkg-config --cflags --libs opencv4` -lGL -lGLU -lglut -O2
```

If some videos will not play:

```bash
sudo apt install ubuntu-restricted-extras
```

## Usage

```bash
./Enceladus_Video_Patcher [video_file...] [options]
```

With no files, camera `0` is used.

### Dual windows

| Window | Role |
|--------|------|
| Enceladus Video | Stream output; drag center to move, edges/corners to resize |
| Enceladus Control Panel | All controls, menus, placement, guides |

`--no-ui` starts **video window only** (useful for scripted theater playback).

### Placement options (scriptable)

```bash
--video-x N --video-y N --video-w N --video-h N
--video-geometry WxH+X+Y
--control-x N --control-y N --control-w N --control-h N
--control-geometry WxH+X+Y
--preset center|topleft|topright|bottomleft|bottomright|lefthalf|righthalf
--fullscreen
```

Examples:

```bash
./Enceladus_Video_Patcher --video-geometry 1920x1080+0+0 --fullscreen movie.mp4
./Enceladus_Video_Patcher --preset righthalf --no-ui
./Enceladus_Video_Patcher --video-x 100 --video-y 50 --video-w 1280 --video-h 720
```

### Calibration guides

```bash
--guides
--guides=cross,edge,thirds,grid,action,title,16:9,4:3,2.39
--guides=all
```

Toggle the same guides from the control panel **Calibration Guides** section.

### Filter / UI options

```bash
--enable-[filter]   # e.g. --enable-glitch
--no-ui
--fullscreen
```

Filter enable flags: `anaglyph`, `gray`, `edge`, `sepia`, `pixelate`, `invert`, `blur`, `vignette`, `noise`, `colorbalance`, `scanlines`, `crt`, `glitch`, `kaleidoscope`, `nightvision`, `colorize`.

### Controls

#### Keyboard (control panel)

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| L | Loop |
| N | Next playlist item |
| F | Toggle **video** fullscreen |
| U | Toggle control UI chrome |
| V | Open file browser |
| C | Camera menu |
| 1-9 | Toggle filters |
| WASD / O P / R | Content pan/scale / reset view |
| [/] | Playback speed |
| ESC | Close menu / exit video fullscreen / quit |

#### Keyboard (video window)

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| F | Fullscreen |
| G | Toggle crosshair + edge guides |
| ESC | Exit fullscreen or quit |

#### Mouse

- **Control panel:** transport, filters, sliders, Open/Cam, placement presets, guide toggles; right-click pie menu
- **Video window:** drag interior to move; drag edges/corners to resize; right-click toggles crosshair

## Examples

```bash
./Enceladus_Video_Patcher --enable-gray --enable-sepia myvideo.mp4
./Enceladus_Video_Patcher --guides=cross,thirds,action --preset center
./Enceladus_Video_Patcher --no-ui --video-geometry 1920x1080+1920+0 --guides=all
```

## Known Issues

- Some filters are heavy on low-end GPUs/CPUs
- Seeking may not be frame-accurate for all codecs
- Multi-monitor placement is manual (geometry/presets/mouse), not OS monitor enumeration

## License

GPL v3 — please submit patches upstream.

## Acknowledgments

- OpenCV
- FreeGLUT
- Contributors and testers
