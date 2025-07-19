"""Configuration settings for the YouTube to Hugo converter."""

import os
from typing import Dict, Any

class Config:
    """Configuration class for video processing and Hugo generation."""
    
    # Video processing settings
    FRAME_SAMPLE_INTERVAL = 20  # seconds (increased for better selection)
    MIN_FACE_RATIO = 0.15  # Skip frames where face > 15% of screen (more strict)
    MAX_FACE_RATIO = 0.05  # Prefer frames where face < 5% of screen  
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
    def load_local_config(cls) -> Dict[str, Any]:
        """Load local configuration from config.local.yaml if it exists."""
        import yaml
        
        local_config_path = 'config.local.yaml'
        if not os.path.exists(local_config_path):
            return {}
        
        try:
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
            
            # Flatten the nested structure for easier access
            flattened = {}
            
            # Handle Claude API configuration
            if 'claude' in local_config:
                claude_config = local_config['claude']
                if 'api_key' in claude_config:
                    flattened['claude_api_key'] = claude_config['api_key']
                if 'model' in claude_config:
                    flattened['claude_model'] = claude_config['model']
            
            # Handle output configuration
            if 'output' in local_config:
                output_config = local_config['output']
                if 'base_folder' in output_config:
                    flattened['output_base_folder'] = output_config['base_folder']
                if 'posts_folder' in output_config:
                    flattened['output_posts_folder'] = output_config['posts_folder']
            
            # Handle template configuration
            if 'template' in local_config:
                template_config = local_config['template']
                if 'path' in template_config:
                    flattened['template_path'] = template_config['path']
            
            # Handle Hugo configuration
            if 'hugo' in local_config:
                hugo_config = local_config['hugo']
                if 'static_path' in hugo_config:
                    flattened['hugo_static_path'] = hugo_config['static_path']
                if 'use_page_bundles' in hugo_config:
                    flattened['use_page_bundles'] = hugo_config['use_page_bundles']
                if 'use_shortcodes' in hugo_config:
                    flattened['use_hugo_shortcodes'] = hugo_config['use_shortcodes']
            
            # Handle processing configuration
            if 'processing' in local_config:
                proc_config = local_config['processing']
                if 'cleanup_temp_files' in proc_config:
                    flattened['cleanup_temp_files'] = proc_config['cleanup_temp_files']
                if 'save_transcripts' in proc_config:
                    flattened['save_transcripts'] = proc_config['save_transcripts']
                if 'default_whisper_model' in proc_config:
                    flattened['whisper_model'] = proc_config['default_whisper_model']
            
            return flattened
            
        except Exception as e:
            print(f"Warning: Could not load local config: {e}")
            return {}
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration as dictionary, merged with local config."""
        # Start with defaults
        config = {
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
        
        # Merge with local configuration
        local_config = cls.load_local_config()
        config.update(local_config)
        
        return config