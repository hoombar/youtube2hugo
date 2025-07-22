"""Main application entry point for YouTube to Hugo converter."""

import os
import sys
import click
import logging
from typing import Dict, Optional
import json
import yaml

from config import Config
from video_processor import VideoProcessor
from transcript_parser import TranscriptParser
from transcript_extractor import TranscriptExtractor
from hugo_generator import HugoGenerator
from semantic_frame_selector import SemanticFrameSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTube2Hugo:
    """Main application class that orchestrates the conversion process."""
    
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.video_processor = VideoProcessor(config_dict)
        self.transcript_parser = TranscriptParser(config_dict)
        self.transcript_extractor = TranscriptExtractor(config_dict)
        self.hugo_generator = HugoGenerator(config_dict)
        self.semantic_frame_selector = SemanticFrameSelector(config_dict, self.video_processor)
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        title: Optional[str] = None,
        video_info: Optional[Dict] = None,
        front_matter: Optional[Dict] = None,
        transcript_path: Optional[str] = None,
        save_transcript: bool = False,
        template_path: Optional[str] = None
    ) -> str:
        """Process a video and transcript into a Hugo blog post."""
        
        logger.info(f"Processing video: {video_path}")
        
        # Create temporary directory for frame extraction
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
        
        try:
            # Get transcript first (needed for intelligent frame selection)
            if transcript_path and os.path.exists(transcript_path):
                logger.info(f"Using existing transcript: {transcript_path}")
                transcript_segments = self.transcript_parser.parse_transcript(transcript_path)
            else:
                logger.info("Extracting transcript from video...")
                transcript_segments = self.transcript_extractor.extract_transcript(video_path)
                
                # Save extracted transcript if requested
                if save_transcript:
                    transcript_output = os.path.splitext(output_path)[0] + '.srt'
                    self.transcript_extractor.save_transcript(
                        transcript_segments, transcript_output, 'srt'
                    )
            
            # Extract and analyze video frames using semantic approach
            logger.info("Extracting video frames with semantic content analysis...")
            semantic_frames = self.semantic_frame_selector.select_frames_semantically(
                video_path, transcript_segments, temp_dir
            )
            
            # Optimize semantic frames (ensure proper formatting and optimization)
            logger.info("Optimizing semantic frames...")
            optimized_frames = self.video_processor.optimize_images(semantic_frames)
            
            # Generate blog post title if not provided
            if not title:
                title = self._generate_title_from_path(video_path)
            
            # Prepare video info
            if not video_info:
                video_info = self._extract_video_info(video_path, transcript_segments)
            
            # Adjust output path based on config
            final_output_path = self._get_configured_output_path(output_path, title)
            
            # Generate Hugo blog post
            logger.info("Generating Hugo blog post...")
            blog_post = self.hugo_generator.generate_blog_post(
                title=title,
                transcript_segments=transcript_segments,
                frame_data=optimized_frames,
                video_info=video_info,
                output_path=final_output_path,
                front_matter_data=front_matter,
                template_path=self.config.get('template_path', template_path)
            )
            
            # Note: Images are now copied to page bundle directory automatically
            # No need for separate Hugo static directory copying
            
            logger.info(f"Successfully generated blog post: {output_path}")
            logger.info(f"Extracted {len(optimized_frames)} relevant images")
            
            return blog_post
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_dir)
    
    def _get_configured_output_path(self, provided_output_path: str, title: str) -> str:
        """Get the final output path based on configuration and title."""
        # If local config specifies base folder, use it
        base_folder = self.config.get('output_base_folder')
        posts_folder = self.config.get('output_posts_folder', 'content/posts')
        
        if base_folder:
            # Use configured base folder structure
            full_posts_path = os.path.join(base_folder, posts_folder)
            os.makedirs(full_posts_path, exist_ok=True)
            
            # Create kebab-case filename from title
            kebab_title = self._title_to_kebab_case(title)
            return os.path.join(full_posts_path, f"{kebab_title}.md")
        else:
            # Use provided output path
            return provided_output_path
    
    def _title_to_kebab_case(self, title: str) -> str:
        """Convert title to kebab-case for folder naming."""
        import re
        # Remove special characters and convert to lowercase
        kebab = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
        # Replace spaces and multiple hyphens with single hyphens
        kebab = re.sub(r'[\s-]+', '-', kebab)
        # Convert to lowercase and strip leading/trailing hyphens
        kebab = kebab.lower().strip('-')
        # Ensure it's not empty
        if not kebab:
            kebab = 'untitled-post'
        return kebab
    
    def _generate_title_from_path(self, video_path: str) -> str:
        """Generate a title from the video file path."""
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Clean up filename to make it title-like
        title = filename.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    def _extract_video_info(self, video_path: str, transcript_segments: list) -> Dict:
        """Extract basic video information."""
        import ffmpeg
        
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            duration = float(video_stream.get('duration', 0))
            
            return {
                'duration': duration,
                'filename': os.path.basename(video_path),
                'transcript_segments': len(transcript_segments)
            }
        except Exception as e:
            logger.warning(f"Could not extract video info: {e}")
            return {'filename': os.path.basename(video_path)}
    
    def _cleanup_temp_files(self, temp_dir: str) -> None:
        """Clean up temporary files."""
        cleanup_enabled = self.config.get('cleanup_temp_files', True)
        
        if cleanup_enabled and os.path.exists(temp_dir):
            import shutil
            try:
                # Count files and calculate size before deletion for reporting
                file_count = 0
                total_size = 0
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                        except OSError:
                            pass  # File might have been deleted already
                
                shutil.rmtree(temp_dir)
                
                if file_count > 0:
                    size_mb = total_size / (1024 * 1024)
                    logger.info(f"Cleaned up {file_count} temporary files ({size_mb:.2f} MB): {temp_dir}")
                else:
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                    
            except Exception as e:
                logger.warning(f"Could not clean up temp files: {e}")

@click.group()
def cli():
    """YouTube to Hugo Blog Post Converter."""
    pass

@click.command()
@click.option('--video', '-v', required=True, help='Path to video file')
@click.option('--transcript', '-t', help='Path to existing transcript file (optional - will extract if not provided)')
@click.option('--output', '-o', help='Output path for Hugo markdown file (can be configured in config.local.yaml)')
@click.option('--title', help='Blog post title (auto-generated if not provided)')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--gemini-api-key', help='Gemini API key for transcript cleanup (or configure in config.local.yaml)')
@click.option('--whisper-model', default='base', help='Whisper model size (tiny, base, small, medium, large)')
@click.option('--save-transcript', is_flag=True, help='Save extracted transcript to .srt file')
@click.option('--template', help='Path to blog post template file with {{placeholders}}')
@click.option('--front-matter', help='Path to JSON file with additional front matter data')
@click.option('--date-offset-days', type=int, help='Set post date this many days in the past (default: 1 to avoid Hugo future date issues)')
def convert(video, transcript, output, title, config, gemini_api_key, whisper_model, save_transcript, template, front_matter, date_offset_days):
    """Convert a video and transcript into a Hugo blog post."""
    
    # Load configuration
    config_dict = Config.get_default_config()
    
    if config and os.path.exists(config):
        custom_config = Config.load_from_file(config)
        config_dict.update(custom_config)
    
    # Add command line options to config (CLI overrides local config)
    if gemini_api_key:
        config_dict['gemini_api_key'] = gemini_api_key
    if whisper_model:
        config_dict['whisper_model'] = whisper_model
    if date_offset_days is not None:
        config_dict['date_offset_days'] = date_offset_days
    
    # Check if we have Gemini API key from somewhere
    if not config_dict.get('gemini_api_key') and not os.environ.get('GOOGLE_API_KEY'):
        click.echo("⚠️  Warning: No Gemini API key found. Set GOOGLE_API_KEY env var or configure in config.local.yaml", err=True)
        click.echo("   Blog formatting will be skipped without API key.", err=True)
    
    # If no output path provided and no base folder configured, require output
    if not output and not config_dict.get('output_base_folder'):
        click.echo("❌ Error: --output is required unless output.base_folder is configured in config.local.yaml", err=True)
        click.echo("   Create config.local.yaml or provide --output path", err=True)
        sys.exit(1)
    
    # Use a default output if base folder is configured
    if not output and config_dict.get('output_base_folder'):
        output = 'post.md'  # Will be replaced by configured path
    
    # Load additional front matter
    front_matter_data = None
    if front_matter and os.path.exists(front_matter):
        with open(front_matter, 'r') as f:
            front_matter_data = json.load(f)
    
    # Initialize and run converter
    converter = YouTube2Hugo(config_dict)
    
    try:
        blog_post = converter.process_video(
            video_path=video,
            output_path=output,
            title=title,
            front_matter=front_matter_data,
            transcript_path=transcript,
            save_transcript=save_transcript,
            template_path=template
        )
        
        click.echo(f"✅ Successfully generated Hugo blog post: {output}")
        if save_transcript and not transcript:
            transcript_file = os.path.splitext(output)[0] + '.srt'
            click.echo(f"📝 Extracted transcript saved to: {transcript_file}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@click.command()
@click.option('--output', '-o', default='config.yaml', help='Output path for config template')
def generate_config(output):
    """Generate a configuration template file."""
    
    generator = HugoGenerator({})
    generator.generate_config_template(output)
    
    click.echo(f"✅ Generated configuration template: {output}")

@click.command()
@click.argument('config_file')
def batch_process(config_file):
    """Process multiple videos using a batch configuration file."""
    
    if not os.path.exists(config_file):
        click.echo(f"❌ Config file not found: {config_file}", err=True)
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        batch_config = yaml.safe_load(f)
    
    base_config = batch_config.get('settings', Config.get_default_config())
    videos = batch_config.get('videos', [])
    
    converter = YouTube2Hugo(base_config)
    
    for i, video_config in enumerate(videos, 1):
        click.echo(f"Processing video {i}/{len(videos)}: {video_config.get('video')}")
        
        try:
            converter.process_video(
                video_path=video_config['video'],
                output_path=video_config['output'],
                title=video_config.get('title'),
                front_matter=video_config.get('front_matter'),
                transcript_path=video_config.get('transcript'),
                save_transcript=video_config.get('save_transcript', False)
            )
            click.echo(f"✅ Completed: {video_config['output']}")
            
        except Exception as e:
            click.echo(f"❌ Error processing {video_config.get('video')}: {e}", err=True)
            continue
    
    click.echo("🎉 Batch processing completed!")

# Add commands to CLI group
cli.add_command(convert)
cli.add_command(generate_config)
cli.add_command(batch_process)

if __name__ == '__main__':
    cli()