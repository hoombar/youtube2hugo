# YouTube to Hugo Blog Post Converter

Convert YouTube videos into structured Hugo blog posts with intelligent frame selection and AI-powered formatting.

## Features

- **Automatic Transcript Extraction**: Uses OpenAI Whisper to extract transcripts directly from video
- **Two-Pass AI Enhancement**: 
  - **Pass 1**: Claude API cleans up transcription errors and typos
  - **Pass 2**: Claude API transforms transcript into engaging blog post with headers and structure
- **Smart Frame Analysis**: Uses computer vision to identify frames containing visual aids (not talking head shots)  
- **Multi-format Support**: Handles existing SRT, VTT, and plain text transcripts or extracts new ones
- **Intelligent Image Placement**: Only extracts frames where visual content is prominent
- **Template System**: Use custom templates with placeholders ({{title}}, {{content}}, {{date}}, etc.)
- **Hugo Integration**: Generates properly formatted Hugo markdown with front matter
- **Page Bundle Structure**: Creates self-contained post folders with relative image paths
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

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Local Configuration**
   ```bash
   cp config.local.yaml.example config.local.yaml
   ```
   
   Edit `config.local.yaml` with your settings:
   ```yaml
   claude:
     api_key: "your-anthropic-api-key-here"
   
   output:
     base_folder: "/path/to/your/hugo/site"
     posts_folder: "content/posts"
   ```

3. **Convert a Video**
   ```bash
   python main.py convert --video video.mp4 --title "My Amazing Tutorial"
   ```
   
   The post will be created at: `/path/to/your/hugo/site/content/posts/my-amazing-tutorial/`

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

### Local Configuration File (`config.local.yaml`)

This file is excluded from git and contains sensitive settings:

```yaml
# Claude API configuration
claude:
  api_key: "your-anthropic-api-key-here"
  model: "claude-4-sonnet-20250514"

# Output configuration  
output:
  base_folder: "/Users/you/hugo-site"
  posts_folder: "content/posts"

# Template configuration (optional)
template:
  path: "/path/to/custom/template.md"

# Hugo configuration
hugo:
  static_path: "static/images"
  use_page_bundles: true
  use_shortcodes: false

# Processing configuration
processing:
  cleanup_temp_files: true
  save_transcripts: false
  default_whisper_model: "base"
```

### CLI Options

```bash
python main.py convert --help
```

Key options:
- `--video`: Path to video file (required)
- `--title`: Blog post title (creates kebab-case folder)
- `--output`: Output path (optional if base_folder configured)
- `--claude-api-key`: Override API key from config
- `--template`: Custom blog post template

## Advanced Usage

### Batch Processing

Create a batch configuration file:

```yaml
settings:
  claude_api_key: "your-key"
  output_base_folder: "/path/to/hugo"

videos:
  - video: "video1.mp4"
    title: "First Tutorial"
  - video: "video2.mp4" 
    title: "Second Tutorial"
```

Run batch processing:
```bash
python main.py batch-process batch_config.yaml
```

### Custom Whisper Models

Choose speed vs accuracy:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

## How It Works

## Frame Selection Algorithm

The algorithm intelligently selects frames by:

1. **Content-Aware Sampling**: Analyzes transcript for visual keywords
2. **Quality Scoring**: Prioritizes frames with screen content, devices, UI elements
3. **Talking Head Avoidance**: Filters out frames dominated by faces
4. **Temporal Diversity**: Ensures varied content across time
5. **Clustered Content**: Groups rapid-fire sequences with smaller images

### Testing Frame Selection

Use the testing script to optimize frame selection:

```bash
# Test current algorithm
python test_frame_selection.py video.mp4 --mode test --duration 60

# Reverse engineer from known good timestamps  
python test_frame_selection.py video.mp4 --mode reverse --timestamps "8.0,15.0,22.0"
```

## Blog Post Formatting

Claude AI transforms raw transcripts into structured blog posts with:

- **Clear section headers** (Introduction, main topics, Conclusion)
- **Engaging introductions** that hook readers
- **Logical flow** with smooth transitions
- **Technical accuracy** preservation
- **Image integration** with contextual placement

## Templates

Create custom blog post templates with placeholders:

```markdown
---
title: "{{title}}"
date: "{{date}}"
categories: ["tutorial"]
---

# {{title}}

{{content}}

---
*Generated from video content*
```

## Output Structure

With title "My Super Interesting YouTube Video", creates:

```
/your/hugo/site/content/posts/my-super-interesting-youtube-video/
├── index.md          # Blog post content
├── frame_8.5s.jpg    # Selected frames
├── frame_15.0s.jpg
└── frame_29.0s.jpg
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

## Template System

The tool supports custom templates with placeholder variables for flexible blog post formatting.

### Available Templates

- **Basic Template**: `examples/templates/basic-template.md` - Simple front matter + content
- **Tech Blog**: `examples/templates/tech-blog-template.md` - Technology-focused with TOC and summary
- **Tutorial**: `examples/templates/tutorial-template.md` - Step-by-step tutorial format
- **Minimal**: `examples/templates/minimal-template.md` - Bare minimum structure

### Template Placeholders

- `{{title}}` - Blog post title
- `{{date}}` - Publication date (ISO format)
- `{{content}}` - Main blog content (formatted by Claude)
- `{{description}}` - Post description
- `{{author}}` - Author name
- `{{tags}}` - Comma-separated tags
- Custom variables from `--front-matter` JSON file

### Example Template

```markdown
---
title: "{{title}}"
date: {{date}}
author: "{{author}}"
tags: ["tutorial", "{{category}}"]
---

# {{title}}

*Generated from video content*

{{content}}

---
*Published: {{date}}*
```

### Using Templates

```bash
python main.py convert \
  --video video.mp4 \
  --output content/posts/my-post \
  --template examples/templates/tech-blog-template.md \
  --front-matter custom-vars.json
```

## Project File Structure

```
youtube2hugo/
├── main.py                 # Main CLI application
├── video_processor.py      # Video analysis and frame extraction
├── transcript_extractor.py # Automatic transcript extraction with Whisper + Claude
├── transcript_parser.py    # Existing transcript file processing
├── blog_formatter.py       # Two-pass Claude AI content enhancement
├── hugo_generator.py       # Hugo markdown generation with template support
├── config.py              # Configuration management
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── examples/              # Example files
    ├── sample-transcript.srt
    ├── config-template.yaml
    ├── batch-config.yaml
    └── templates/         # Blog post templates
        ├── basic-template.md
        ├── tech-blog-template.md
        ├── tutorial-template.md
        └── minimal-template.md
```

## Two-Pass AI Enhancement

When a Claude API key is provided, the tool performs two passes of AI enhancement:

### Pass 1: Transcript Cleanup
- Fixes speech recognition errors and typos
- Removes filler words ("um", "uh", repeated phrases)
- Corrects grammar and punctuation
- Maintains original meaning and conversational flow

### Pass 2: Blog Post Formatting
- Transforms transcript into engaging blog post
- Adds proper headers and section structure
- Improves readability and flow
- **Preserves all image placements** from the original transcript timing
- Creates introduction and conclusion sections
- Adds smooth transitions between topics

### Example Transformation

**Before (raw transcript):**
```
Um, so today we're going to talk about, uh, machine learning and, you know, how it works. So basically machine learning is, is when computers learn patterns from data...
```

**After Pass 1 (cleaned):**
```
Today we're going to talk about machine learning and how it works. Machine learning is when computers learn patterns from data...
```

**After Pass 2 (formatted):**
```
# Introduction to Machine Learning

Welcome to this comprehensive guide on machine learning fundamentals.

## What is Machine Learning?

Machine learning is the process by which computers learn patterns from data...
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

### No Claude API Key
```
⚠️ Warning: No Claude API key found
```
**Solution**: Add API key to `config.local.yaml` or set `ANTHROPIC_API_KEY` environment variable

### Missing Output Path
```
❌ Error: --output is required unless output.base_folder is configured
```
**Solution**: Either provide `--output` or configure `output.base_folder` in `config.local.yaml`

### Poor Frame Selection
**Solution**: Use the testing script to analyze and tune frame selection parameters

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