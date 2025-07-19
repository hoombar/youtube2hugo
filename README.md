# YouTube Video to Hugo Blog Post Automation

A Python application that intelligently converts YouTube videos into Hugo blog posts with strategically placed images and automatically extracted, AI-cleaned transcripts.

## Features

- **Automatic Transcript Extraction**: Uses OpenAI Whisper to extract transcripts directly from video
- **AI-Powered Transcript Cleanup**: Uses Claude API to fix speech recognition errors and typos
- **Smart Frame Analysis**: Uses computer vision to identify frames containing visual aids (not talking head shots)  
- **Multi-format Support**: Handles existing SRT, VTT, and plain text transcripts or extracts new ones
- **Intelligent Image Placement**: Only extracts frames where visual content is prominent
- **Hugo Integration**: Generates properly formatted Hugo markdown with front matter
- **Configurable Processing**: Adjustable thresholds for face detection and frame selection
- **Batch Processing**: Handle multiple videos at once
- **CLI Interface**: Easy-to-use command-line interface

## Installation

1. Clone the repository:
```bash
git clone git@github.com:hoombar/youtube2hugo.git
cd youtube2hugo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for video processing):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Quick Start

### Basic Usage

Convert a video with automatic transcript extraction:

```bash
python main.py convert --video video.mp4 --output blog-post.md
```

### With Claude API for transcript cleanup:

```bash
export ANTHROPIC_API_KEY="your-claude-api-key"
python main.py convert \
  --video presentation.mp4 \
  --output content/posts/my-presentation \
  --title "My Amazing Presentation" \
  --save-transcript
```

### Using existing transcript:

```bash
python main.py convert \
  --video video.mp4 \
  --transcript existing-transcript.srt \
  --output blog-post.md
```

### Generate Configuration Template

```bash
python main.py generate-config --output my-config.yaml
```

### Batch Processing

```bash
python main.py batch-process batch-config.yaml
```

## Configuration

### Configuration File Example

```yaml
video_processing:
  frame_sample_interval: 15      # Extract frame every 15 seconds
  min_face_ratio: 0.4           # Skip frames with face > 40% of screen
  max_face_ratio: 0.2           # Prefer frames with face < 20% of screen
  face_detection_confidence: 0.5

image_settings:
  quality: 95                   # JPEG quality (1-100)
  max_width: 1920              # Maximum image width
  max_height: 1080             # Maximum image height

hugo_settings:
  use_hugo_shortcodes: false   # Use {{< figure >}} shortcode instead of ![]()

transcript_settings:
  context_window: 30           # Seconds of context around frames
  whisper_model: "base"        # Whisper model: tiny, base, small, medium, large
  claude_model: "claude-3-haiku-20240307"  # Claude model for cleanup
  claude_api_key: "your-key"   # Or set ANTHROPIC_API_KEY env var

front_matter_defaults:
  tags: ["video", "auto-generated"]
  categories: ["video"]
  author: "YouTube2Hugo"
```

### Batch Processing Configuration

```yaml
settings:
  frame_sample_interval: 20
  min_face_ratio: 0.35
  whisper_model: "base"
  claude_api_key: "your-claude-api-key"

videos:
  - video: "videos/presentation1.mp4"
    output: "content/posts/presentation1.md"
    title: "Introduction to AI"
    save_transcript: true
    
  - video: "videos/tutorial.mp4"
    transcript: "transcripts/tutorial.vtt"  # Use existing transcript
    output: "content/posts/tutorial.md"
    front_matter:
      tags: ["tutorial", "programming"]
      categories: ["education"]
      
  - video: "videos/lecture.mp4"
    output: "content/posts/lecture.md"
    title: "Advanced Machine Learning"
    save_transcript: true
```

## How It Works

### 1. Frame Selection Logic

The application uses MediaPipe for face detection to analyze video frames:

- **Samples frames** every 10-30 seconds (configurable)
- **Calculates face-to-screen ratio** for each frame
- **Extracts frames where**:
  - No face detected (full visual aids)
  - Face occupies <20% of screen (visual aids with presenter in corner)
  - Skips frames where face occupies >40% (talking head shots)

### 2. Transcript Processing

- **Automatic extraction** using OpenAI Whisper (multiple model sizes available)
- **AI cleanup** with Claude API to fix speech recognition errors
- **Multi-format support**: SRT, VTT, and plain text with timestamps
- Matches extracted frame timestamps to nearby transcript segments
- Inserts images at natural paragraph breaks in the text

### 3. Hugo Output

- Generates proper Hugo front matter (title, date, tags, etc.)
- Creates clean markdown with embedded images
- **Page Bundle Structure**: Creates a folder for each post with `index.md` and images
- **Relative Image Paths**: Images are referenced relative to the post (e.g., `image.jpg` not `/static/images/image.jpg`)
- Supports both standard markdown and Hugo shortcodes

## Output Structure

The tool creates Hugo page bundles, which are self-contained folders for each blog post:

```
content/posts/
└── my-video-post/           # Page bundle directory
    ├── index.md             # Blog post content with front matter
    ├── frame_45.0s.jpg      # Video frame at 45 seconds
    ├── frame_120.0s.jpg     # Video frame at 120 seconds
    └── frame_300.0s.jpg     # Video frame at 300 seconds
```

**Benefits of Page Bundles:**
- **Self-contained**: All resources (images) are stored with the post
- **Portable**: Easy to move or backup entire posts
- **Relative paths**: Images use simple filenames like `frame_45.0s.jpg`
- **Hugo native**: Follows Hugo's recommended page bundle structure

**Example output:**
```markdown
---
title: "My Video Presentation"
date: "2024-01-15T10:30:00"
---

Welcome to this presentation on machine learning fundamentals.

![Visual content from video at 45s](frame_45.0s.jpg)

Let's explore the key concepts that drive modern AI systems...
```

## Project File Structure

```
youtube2hugo/
├── main.py                 # Main CLI application
├── video_processor.py      # Video analysis and frame extraction
├── transcript_extractor.py # Automatic transcript extraction with Whisper + Claude
├── transcript_parser.py    # Existing transcript file processing
├── hugo_generator.py       # Hugo markdown generation
├── config.py              # Configuration management
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── examples/              # Example files
    ├── sample-transcript.srt
    ├── config-template.yaml
    └── batch-config.yaml
```

## Advanced Usage

### Custom Front Matter

Create a JSON file with additional front matter:

```json
{
  "author": "John Doe",
  "tags": ["presentation", "ai", "machine-learning"],
  "categories": ["technology"],
  "series": "AI Fundamentals",
  "weight": 10
}
```

Use with:
```bash
python main.py convert --video video.mp4 --transcript script.srt --output post.md --front-matter custom.json
```

### Hugo Shortcodes

Enable Hugo figure shortcodes in your config:

```yaml
hugo_settings:
  use_hugo_shortcodes: true
```

This generates:
```markdown
{{< figure src="/images/frame_45.0s.jpg" alt="Visual content showing diagram, interface" >}}
```

Instead of:
```markdown
![Visual content showing diagram, interface](/images/frame_45.0s.jpg)
```

## Supported Transcript Formats

### SRT (SubRip)
```
1
00:00:10,500 --> 00:00:13,000
Welcome to this presentation about AI.

2
00:00:15,000 --> 00:00:18,500
Today we'll cover machine learning basics.
```

### VTT (WebVTT)
```
WEBVTT

00:00:10.500 --> 00:00:13.000
Welcome to this presentation about AI.

00:00:15.000 --> 00:00:18.500
Today we'll cover machine learning basics.
```

### Plain Text with Timestamps
```
0:10 Welcome to this presentation about AI.
0:15 Today we'll cover machine learning basics.
0:30 Let's start with this diagram...
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```
   Error: ffmpeg not found in PATH
   ```
   Solution: Install FFmpeg and ensure it's in your system PATH.

2. **MediaPipe installation issues**
   ```
   Error: No module named 'mediapipe'
   ```
   Solution: Install with pip: `pip install mediapipe==0.10.7`

3. **No frames extracted**
   ```
   Warning: No suitable frames found
   ```
   Solution: Lower the `min_face_ratio` threshold in config or check video content.

4. **Poor image quality**
   ```
   Images appear blurry or low quality
   ```
   Solution: Increase `image_quality` and adjust `max_width`/`max_height` in config.

### Debug Mode

Enable verbose logging:
```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG)" main.py convert --video video.mp4 --transcript script.srt --output post.md
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Requirements

- Python 3.8+
- FFmpeg
- OpenAI Whisper
- Claude API key (optional, for transcript cleanup)
- OpenCV
- MediaPipe
- PyTorch (for Whisper)
- See `requirements.txt` for complete list

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Claude API key for transcript cleanup (optional)

## Example Output

The generated Hugo blog post will look like:

```markdown
---
title: "Introduction to Machine Learning"
date: "2024-01-15T10:30:00"
draft: false
tags: ["video", "machine-learning"]
categories: ["education"]
description: "Blog post generated from video content"
video_duration: "1800s"
author: "YouTube2Hugo"
---

Welcome to this comprehensive introduction to machine learning. In this presentation, we'll explore the fundamental concepts that drive modern AI systems.

![Visual content showing diagram, architecture](/images/frame_45.0s.jpg)

Machine learning algorithms can be broadly categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning. Each approach has its own strengths and use cases.

![Visual content showing chart, visualization](/images/frame_120.0s.jpg)

Let's dive deeper into supervised learning, which is perhaps the most commonly used approach in practical applications today.
```