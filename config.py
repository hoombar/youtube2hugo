"""Configuration settings for the YouTube to Hugo converter."""

import os
from typing import Dict, Any

class Config:
    """Configuration class for video processing and Hugo generation."""
    
    # Video processing settings
    FRAME_SAMPLE_INTERVAL = 15  # seconds
    MIN_FACE_RATIO = 0.4  # Skip frames where face > 40% of screen
    MAX_FACE_RATIO = 0.2  # Prefer frames where face < 20% of screen
    FACE_DETECTION_CONFIDENCE = 0.5
    
    # Image settings
    IMAGE_QUALITY = 95
    IMAGE_MAX_WIDTH = 1920
    IMAGE_MAX_HEIGHT = 1080
    
    # Hugo settings
    HUGO_STATIC_PATH = "static/images"
    HUGO_CONTENT_PATH = "content/posts"
    
    # Transcript settings
    CONTEXT_WINDOW = 30  # seconds before/after frame for transcript context
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large
    CLAUDE_MODEL = "claude-4-sonnet-20250514"
    
    # Output settings
    DEFAULT_OUTPUT_DIR = "output"
    
    @classmethod
    def load_from_file(cls, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        if not os.path.exists(config_file):
            return {}
            
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration as dictionary."""
        return {
            'frame_sample_interval': cls.FRAME_SAMPLE_INTERVAL,
            'min_face_ratio': cls.MIN_FACE_RATIO,
            'max_face_ratio': cls.MAX_FACE_RATIO,
            'face_detection_confidence': cls.FACE_DETECTION_CONFIDENCE,
            'image_quality': cls.IMAGE_QUALITY,
            'image_max_width': cls.IMAGE_MAX_WIDTH,
            'image_max_height': cls.IMAGE_MAX_HEIGHT,
            'hugo_static_path': cls.HUGO_STATIC_PATH,
            'hugo_content_path': cls.HUGO_CONTENT_PATH,
            'context_window': cls.CONTEXT_WINDOW,
            'whisper_model': cls.WHISPER_MODEL,
            'claude_model': cls.CLAUDE_MODEL,
            'default_output_dir': cls.DEFAULT_OUTPUT_DIR
        }