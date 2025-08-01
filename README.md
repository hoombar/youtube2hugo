# YouTube to Hugo Blog Post Converter

Convert YouTube videos into structured Hugo blog posts with intelligent frame selection and AI-powered formatting.

## Features

### Core Functionality
- **Automatic Transcript Extraction**: Uses OpenAI Whisper to extract transcripts directly from video
- **AI-Powered Content Enhancement**: 
  - **Semantic Frame Selection**: Gemini AI analyzes transcript content to intelligently select relevant frames
  - **Content-Aware Formatting**: Single-pass blog post generation with contextual image placement
  - **Multi-Strategy Prompting**: Robust AI processing with fallback strategies to handle content restrictions
- **Smart Frame Analysis**: Uses computer vision to identify frames containing visual aids (not talking head shots)  
- **Multi-format Support**: Handles existing SRT, VTT, and plain text transcripts or extracts new ones
- **Intelligent Image Placement**: Only extracts frames where visual content is prominent
- **Template System**: Use custom templates with placeholders ({{title}}, {{content}}, {{date}}, etc.)
- **Hugo Integration**: Generates properly formatted Hugo markdown with front matter
- **Page Bundle Structure**: Creates self-contained post folders with relative image paths

### Processing Modes
- **CLI Mode**: Traditional command-line interface for automated processing
- **Hybrid Mode**: Web-based interface combining AI processing with manual frame selection
- **Batch Processing**: Handle multiple videos at once with configuration files

### Advanced Tools
- **Frame Selection Training**: Machine learning tools to optimize frame selection algorithms
- **Performance Analysis**: Scripts to evaluate and tune frame selection quality
- **Testing Suite**: Comprehensive testing tools for algorithm validation
- **Debug Tools**: Detailed debugging capabilities for troubleshooting processing issues

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

### Method 1: Command Line Interface (CLI)

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
   gemini:
     api_key: "your-google-api-key-here"
     model: "gemini-2.5-flash"
   
   output:
     base_folder: "/path/to/your/hugo/site"
     posts_folder: "content/posts"
   ```

3. **Convert a Video**
   ```bash
   python main.py convert --video video.mp4 --title "My Amazing Tutorial"
   ```
   
   The post will be created at: `/path/to/your/hugo/site/content/posts/my-amazing-tutorial/`

### Method 2: Hybrid Web Interface (Recommended)

For more control over frame selection and better results:

1. **Start the Web Interface**
   ```bash
   python create_blog.py
   ```

2. **Open Your Browser**
   - Navigate to `http://127.0.0.1:5002`
   - Upload your video file path and title
   - Choose processing mode (smart/dedupe/raw)

3. **Review and Select Frames**
   - AI processes transcript and creates sections
   - Review candidate frames for each section
   - Select the best frames manually
   - Generate final blog post

This method provides better quality control and allows manual frame curation.

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
# Gemini API configuration
gemini:
  api_key: "your-google-api-key-here"
  model: "gemini-2.5-flash"

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
  
# Date configuration
date:
  offset_days: 1  # Set post date this many days in the past
```

### CLI Options

```bash
python main.py convert --help
```

Key options:
- `--video`: Path to video file (required)
- `--title`: Blog post title (creates kebab-case folder)
- `--output`: Output path (optional if base_folder configured)
- `--gemini-api-key`: Override API key from config
- `--template`: Custom blog post template

## Advanced Usage

### Batch Processing

Create a batch configuration file:

```yaml
settings:
  gemini_api_key: "your-key"
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

### Frame Processing Modes

The hybrid interface offers three processing modes:

- **Smart**: Uses AI-powered analysis to extract semantically relevant frames
- **Dedupe**: Extracts frames every 0.5s and removes duplicates using perceptual hashing
- **Raw**: Fast extraction of frames every 0.5s without duplicate removal

### Algorithm Training and Optimization

The project includes sophisticated tools for improving frame selection:

#### Frame Selection Training
```bash
# Train the algorithm on known good timestamps
python train_frame_selection.py video.mp4 --good-timestamps "8.0,15.0,22.0"

# Cumulative learning from multiple videos
python cumulative_trainer.py --videos video1.mp4,video2.mp4,video3.mp4
```

#### Performance Analysis
```bash
# Analyze frame selection quality
python frame_selection_analyzer.py video.mp4

# Test different similarity thresholds
python test_similarity_thresholds.py video.mp4

# Quick frame analysis for debugging
python quick_frame_analysis.py video.mp4
```

#### Debug Tools
```bash
# Debug timing and boundary issues
python debug_boundary_markers.py video.mp4
python debug_frame_timing.py video.mp4

# Test AI processing independently
python test_ai_processing.py
```

### Custom Whisper Models

Choose speed vs accuracy:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

## How It Works

## Semantic Frame Selection Algorithm

The new semantic algorithm intelligently selects frames by:

1. **AI Content Analysis**: Gemini AI analyzes transcript to identify semantic sections and topics
2. **Frame-Content Matching**: Each frame is analyzed for visual content and matched to relevant topics
3. **Contextual Relevance**: Frames are selected based on how well they illustrate the discussed concepts
4. **Quality Scoring**: Prioritizes frames with screen content, diagrams, code, and UI elements
5. **Talking Head Avoidance**: Filters out frames dominated by faces using computer vision
6. **Section-Aware Placement**: Images are placed with rich context from their semantic sections

### Testing Frame Selection

Use the testing script to optimize frame selection:

```bash
# Test current algorithm
python test_frame_selection.py video.mp4 --mode test --duration 60

# Reverse engineer from known good timestamps  
python test_frame_selection.py video.mp4 --mode reverse --timestamps "8.0,15.0,22.0"
```

## Blog Post Formatting

Gemini AI transforms raw transcripts into structured blog posts with:

- **Semantic content analysis** to understand topics and concepts
- **Clear section headers** based on content themes
- **Contextual image placement** that matches visual content to discussed topics
- **Enhanced alt text** with section context and descriptions
- **Logical flow** with smooth transitions
- **Technical accuracy** preservation

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
├── create_blog.py          # Hybrid web interface launcher
├── hybrid_blog_creator.py  # Web-based blog creation with manual frame selection
├── video_processor.py      # Video analysis and frame extraction
├── transcript_extractor.py # Automatic transcript extraction with Whisper
├── transcript_parser.py    # Existing transcript file processing
├── semantic_frame_selector.py # AI-powered semantic frame selection
├── blog_formatter.py       # Gemini AI content enhancement
├── hugo_generator.py       # Hugo markdown generation with template support
├── config.py              # Configuration management
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── algorithm_comparison.md # Algorithm performance analysis
├── training/              # Frame selection training tools
│   ├── train_frame_selection.py
│   ├── cumulative_trainer.py
│   ├── frame_selection_trainer.py
│   └── demo_cumulative_learning.py
├── analysis/              # Performance analysis tools
│   ├── frame_selection_analyzer.py
│   ├── quick_frame_analysis.py
│   ├── score_threshold_tuner.py
│   └── apply_recommendations.py
├── testing/               # Testing and validation tools
│   ├── test_frame_selection.py
│   ├── test_semantic_selection.py
│   ├── test_similarity_thresholds.py
│   ├── test_ai_processing.py
│   ├── test_full_frame_extraction.py
│   ├── test_frame_cleanup.py
│   └── test_boundary_fix.py
├── debug/                 # Debug utilities
│   ├── debug_boundary_markers.py
│   ├── debug_frame_timing.py
│   └── debug_session_creation.py
├── templates/             # Web interface templates
│   ├── hybrid_blog_creator.html
│   └── frame_selector.html
├── hugo-shortcodes/       # Hugo shortcode templates
│   ├── README.md
│   ├── grid-image.html
│   └── image-grid.html
└── examples/              # Example files
    ├── sample-transcript.srt
    ├── config-template.yaml
    ├── batch-config.yaml
    ├── example_config_tuning.yaml
    └── templates/         # Blog post templates
        ├── basic-template.md
        ├── tech-blog-template.md
        ├── tutorial-template.md
        └── minimal-template.md
```

## AI-Powered Content Enhancement

When a Gemini API key is provided, the tool performs intelligent content processing:

### Semantic Analysis & Frame Selection
- Analyzes transcript content to identify semantic sections and topics
- Extracts frames that visually represent the discussed concepts
- Matches visual content to textual content using AI analysis
- Scores frames based on relevance to the discussion topics

### Blog Post Formatting
- Transforms transcript into engaging blog post with semantic context
- Places images based on content relevance rather than just timing
- Generates enhanced alt text with section context
- Creates smooth content flow with contextually appropriate visuals
- Preserves technical accuracy while improving readability

### Robust AI Processing
- **Multi-Strategy Prompting**: Uses multiple prompting strategies to work around AI safety filters
- **Graceful Fallbacks**: When AI processing fails, creates enhanced basic sections from transcript
- **Content Quality Validation**: Verifies generated content meets blog post standards
- **Safety Filter Handling**: Automatically detects and adapts to content restrictions
- **Error Recovery**: Comprehensive error handling with informative feedback

### Example Transformation

**Before (raw transcript):**
```
Um, so today we're going to talk about, uh, machine learning and, you know, how it works. So basically machine learning is, is when computers learn patterns from data...
```

**After AI Processing (semantic analysis + formatting):**
```
# Introduction to Machine Learning

Welcome to this comprehensive guide on machine learning fundamentals.

## What is Machine Learning?

Machine learning is the process by which computers learn patterns from data...

![Machine learning workflow diagram demonstration at 45.2s](frame_45.2s.jpg)
*Configuration and Setup*

The above diagram illustrates the core components of a machine learning pipeline...
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

### No Gemini API Key
```
⚠️ Warning: No Gemini API key found
```
**Solution**: Add API key to `config.local.yaml` or set `GOOGLE_API_KEY` environment variable

### Missing Output Path
```
❌ Error: --output is required unless output.base_folder is configured
```
**Solution**: Either provide `--output` or configure `output.base_folder` in `config.local.yaml`

### Poor Frame Selection
**Solution**: Use the testing script to analyze and tune frame selection parameters

### Hugo Not Publishing Posts
```
Post created but doesn't appear on Hugo site
```
**Solution**: Hugo doesn't publish posts with future dates. The tool now defaults to yesterday's date. If needed, adjust in config:
```yaml
date:
  offset_days: 1  # Days in the past (1 = yesterday)
```

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

## Web Interface Workflow

The hybrid web interface provides a streamlined workflow:

1. **Video Processing**: Upload video path and title, choose processing mode
2. **AI Analysis**: Automatic transcript extraction and AI-powered content creation
3. **Section Review**: Review generated sections with timing information
4. **Frame Selection**: Browse candidate frames for each section and select the best ones
5. **Blog Generation**: Automatically generate final Hugo blog post with selected frames

### Interface Features

- **Real-time Processing**: Live feedback during video analysis
- **Image Preview**: Thumbnail previews of all candidate frames
- **Section-based Organization**: Frames organized by content sections
- **Manual Override**: Full control over frame selection
- **Progress Tracking**: Clear indication of processing status
- **Error Handling**: Graceful handling of processing failures

## Requirements

- Python 3.8+
- FFmpeg
- OpenAI Whisper
- Google Generative AI (Gemini API) for semantic frame selection and content enhancement
- OpenCV
- MediaPipe
- PyTorch (for Whisper)
- Flask (for web interface)
- Additional dependencies for machine learning training tools
- See `requirements.txt` for complete list

## Environment Variables

- `GOOGLE_API_KEY`: Your Gemini API key for semantic frame selection and content enhancement

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