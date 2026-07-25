# Enceladus Video Patcher

## Overview

Enceladus Video Patcher is a realtime video processing application that applies visual effects to video streams or files. Built with OpenCV and OpenGL/FreeGLUT.

The app runs as **two windows**:
- **Enceladus Video** — video/camera stream only (plus optional calibration guides)
- **Enceladus Control Panel** — playback, filters, sources, placement, and guides

## Features

- Real-time video processing with multiple simultaneous filters
- 15+ visual effects (anaglyph, sequential anaglyph, motion-pop 3D anaglyph, grayscale, edge, sepia, pixelate, invert, blur, vignette, noise, color balance, scanlines, CRT, glitch, kaleidoscope, night vision, colorize)
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

Filter enable flags: `anaglyph`, `anaglyphseq`, `motionpop`, `enceladusvision` (aliases: `vision`, `wavepop`), `gray`, `edge`, `sepia`, `pixelate`, `invert`, `blur`, `vignette`, `noise`, `colorbalance`, `scanlines`, `crt`, `glitch`, `kaleidoscope`, `nightvision`, `colorize`.

**Motion Pop 3D** (`--enable-motionpop`): tracks large moving regions over time and only pops objects that stay on screen long enough (~0.5–1.5s) and cover a meaningful part of the frame. Brief flicker / small motion is ignored. Sliders: Strictness (higher = fewer / harder-to-qualify pops), Pop Scale.

**EnceladusVision** (`--enable-enceladusvision` / `--enable-wavepop`): soft raised-cosine wave extrusion on tracked regions (not hard boxes). Default view keeps video in-frame; orbit to inspect 3D. **Ana ON/OFF** toggles anaglyph; **Ana Amount** sets disparity when on. **Wave Lerp** eases waves in/out (0 = instant jump).

Sliders: Strength, Strictness, Wave Soft, Wave Radius, Orbit Yaw/Pitch, Pop Scale, Hold Time, Ana Amount, Size Depth, Motion Point, Wave Lerp.

Calibration tools: Reset, Front, Ana ON/OFF, Tracks / Motion / Height overlays, Relearn.

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
| I | Live window / screen capture menu |
| 1-9 | Toggle filters |
| WASD / O P / R | Content pan/scale / reset view (R also refreshes open source menus) |
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

- **Control panel:** transport, filters, sliders, Open/Cam/Win, placement presets, guide toggles; right-click pie menu
- **Video window:** drag interior to move; drag edges/corners to resize; right-click toggles crosshair

### Live window capture (X11)

**Win** button or **I** opens a list of top-level windows plus **Full Desktop / Screen**. Selecting one streams that region into the filter pipeline in real time. Requires an X11 session.

**Window Capture Tune** (appears while live): Crop L/R/T/B, Scale (downscale for performance), Cap FPS (~8–60), Reset Cap.

### Aspect format presets

**Aspect Formats** panel (right column, under Filters): pick cinema/delivery ratios and apply as **Crop** (center-cut) or **Letter** (pad bars). **Off** restores native aspect. **Bord** toggles the gold aspect guide line on the video window.

- **Aspect** slider: continuous custom ratio (~0.40–3.60); dragging switches to custom.
- **Resolution force:** **Def** (source/default), **480 / 720 / 1080 / 1440 / 4K / Sq1K**, or **Custom** with **Res W / Res H** sliders (about 320–3840 × 240–2160). Forced size is applied after aspect.

Presets include: 1:1, 5:4, 4:3, 3:2, 16:10, 16:9, 1.85 Flat, **1.90 IMAX Digital**, 2:1, 21:9, 2.20 70mm, 2.35/2.39/2.40 Scope, 2.76 Ultra Panavision, **1.43 IMAX GT**, 4:5, 9:16, 32:9.

```bash
./Enceladus_Video_Patcher --enable-enceladusvision
# I / Win → pick window; use Capture Tune + Aspect Formats panels
```

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
