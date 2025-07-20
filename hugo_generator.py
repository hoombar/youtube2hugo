"""Hugo blog post generator module."""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml
import logging
from blog_formatter import BlogFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HugoGenerator:
    """Handles generation of Hugo-compatible markdown files."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.blog_formatter = BlogFormatter(config)
    
    def generate_blog_post(
        self, 
        title: str,
        transcript_segments: List[Dict],
        frame_data: List[Dict],
        video_info: Dict,
        output_path: str,
        front_matter_data: Optional[Dict] = None,
        template_path: Optional[str] = None
    ) -> str:
        """Generate a complete Hugo blog post with front matter and content in page bundle format."""
        
        # Create page bundle directory structure with kebab-case naming
        bundle_dir = self._create_page_bundle_structure(output_path, title)
        
        # Copy images to bundle directory
        self._copy_images_to_bundle(frame_data, bundle_dir)
        
        # Generate front matter
        front_matter = self._generate_front_matter(
            title, video_info, front_matter_data
        )
        
        # Generate content with smart image placement (using relative paths)
        raw_content = self._generate_content_with_images(
            transcript_segments, frame_data, bundle_dir
        )
        
        # Second pass: Format content as blog post with Gemini
        formatted_content = self.blog_formatter.format_content_with_images(
            raw_content, title, frame_data
        )
        
        # Apply template if provided
        if template_path:
            template_variables = {
                'title': title,
                'date': datetime.now().strftime('%Y-%m-%dT00:00:00Z'),
                'content': formatted_content
            }
            # Add custom front matter variables
            if front_matter_data:
                template_variables.update(front_matter_data)
            
            final_content = self.blog_formatter.apply_template(
                formatted_content, template_path, template_variables
            )
        else:
            # Use default front matter + content structure
            final_content = f"---\n{front_matter}\n---\n\n{formatted_content}"
        
        # Write index.md file in bundle directory
        index_path = os.path.join(bundle_dir, 'index.md')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Clean up unused images from the bundle directory
        self._cleanup_unused_images(bundle_dir, final_content)
        
        logger.info(f"Generated Hugo page bundle: {bundle_dir}")
        logger.info(f"Blog post created at: {index_path}")
        return final_content
    
    def _generate_front_matter(
        self, 
        title: str, 
        video_info: Dict, 
        custom_data: Optional[Dict] = None
    ) -> str:
        """Generate Hugo front matter in YAML format."""
        
        # Generate date that's definitely in the past to avoid Hugo future date issues
        # Check config for date offset (default: 1 day ago to ensure publication)
        date_offset_days = self.config.get('date_offset_days', 1)
        publish_date = datetime.now() - timedelta(days=date_offset_days)
        date_string = publish_date.strftime('%Y-%m-%dT00:00:00Z')
        
        front_matter_dict = {
            'title': title,
            'date': date_string,
            'draft': False,
            'tags': [],
            'categories': ['video'],
            'description': f"Blog post generated from video content",
        }
        
        # Add video-specific metadata
        if video_info:
            if 'duration' in video_info:
                front_matter_dict['video_duration'] = f"{video_info['duration']:.0f}s"
            if 'url' in video_info:
                front_matter_dict['video_url'] = video_info['url']
            if 'author' in video_info:
                front_matter_dict['author'] = video_info['author']
        
        # Merge with custom front matter data
        if custom_data:
            front_matter_dict.update(custom_data)
        
        return yaml.dump(front_matter_dict, default_flow_style=False)
    
    def _create_page_bundle_structure(self, output_path: str, title: str = None) -> str:
        """Create Hugo page bundle directory structure with kebab-case naming."""
        # If title is provided, create kebab-case folder name
        if title:
            folder_name = self._title_to_kebab_case(title)
            base_dir = os.path.dirname(output_path)
            bundle_dir = os.path.join(base_dir, folder_name)
        else:
            # Fallback to original behavior
            if output_path.endswith('.md'):
                bundle_dir = output_path[:-3]
            else:
                bundle_dir = output_path
            
        # Create the bundle directory
        os.makedirs(bundle_dir, exist_ok=True)
        
        return bundle_dir
    
    def _title_to_kebab_case(self, title: str) -> str:
        """Convert title to kebab-case for folder naming."""
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
    
    def _copy_images_to_bundle(self, frame_data: List[Dict], bundle_dir: str) -> None:
        """Copy optimized images to the page bundle directory."""
        import shutil
        
        for frame in frame_data:
            if not frame.get('should_include', False):
                continue
                
            source_path = frame.get('optimized_path', frame['path'])
            if not os.path.exists(source_path):
                continue
                
            filename = os.path.basename(source_path)
            dest_path = os.path.join(bundle_dir, filename)
            
            shutil.copy2(source_path, dest_path)
            
            # Update frame data with new bundle-relative path
            frame['bundle_path'] = filename
            
            logger.info(f"Copied image to bundle: {dest_path}")
    
    def _generate_content_with_images(
        self, 
        transcript_segments: List[Dict], 
        frame_data: List[Dict],
        bundle_dir: Optional[str] = None
    ) -> str:
        """Generate blog content with intelligently placed images."""
        
        content_parts = []
        used_frames = set()
        
        # Sort frame data by timestamp
        sorted_frames = sorted(
            [f for f in frame_data if f.get('should_include', False)],
            key=lambda x: x['timestamp']
        )
        
        # Group transcript segments into paragraphs
        paragraphs = self._group_segments_into_paragraphs(transcript_segments)
        
        for paragraph in paragraphs:
            # Add paragraph text
            paragraph_text = self._format_paragraph_text(paragraph)
            content_parts.append(paragraph_text)
            
            # Find all relevant frames for this paragraph (including clustered ones)
            relevant_frames = self._find_relevant_frames_for_paragraph(
                paragraph, sorted_frames, used_frames
            )
            
            if relevant_frames:
                # Group clustered images together
                clustered_images = []
                regular_images = []
                
                for frame in relevant_frames:
                    if frame.get('is_clustered', False):
                        clustered_images.append(frame)
                    else:
                        regular_images.append(frame)
                
                # Add regular images
                for frame in regular_images:
                    image_markdown = self._generate_image_markdown(frame, paragraph)
                    content_parts.append(image_markdown)
                    used_frames.add(frame['timestamp'])
                
                # Add clustered images as a group
                if clustered_images:
                    clustered_markdown = self._generate_clustered_images_markdown(clustered_images, paragraph)
                    content_parts.append(clustered_markdown)
                    for frame in clustered_images:
                        used_frames.add(frame['timestamp'])
        
        return '\n\n'.join(content_parts)
    
    def _group_segments_into_paragraphs(self, segments: List[Dict]) -> List[List[Dict]]:
        """Group transcript segments into logical paragraphs."""
        paragraphs = []
        current_paragraph = []
        
        for segment in segments:
            current_paragraph.append(segment)
            
            # End paragraph on natural breaks or after certain duration
            paragraph_duration = (
                current_paragraph[-1]['end_time'] - current_paragraph[0]['start_time']
            )
            
            text = segment['text']
            ends_with_sentence = text.strip().endswith(('.', '!', '?'))
            is_long_enough = paragraph_duration >= 60  # 1 minute paragraphs
            
            if (ends_with_sentence and is_long_enough) or paragraph_duration >= 120:
                paragraphs.append(current_paragraph)
                current_paragraph = []
        
        # Add remaining segments as final paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs
    
    def _format_paragraph_text(self, paragraph: List[Dict]) -> str:
        """Format a paragraph from transcript segments."""
        text_parts = []
        
        for segment in paragraph:
            text = segment['text'].strip()
            if text:
                text_parts.append(text)
        
        # Join and clean up the text
        paragraph_text = ' '.join(text_parts)
        
        # Add proper sentence spacing
        paragraph_text = re.sub(r'\.(\w)', r'. \1', paragraph_text)
        paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
        
        return paragraph_text.strip()
    
    def _find_best_frame_for_paragraph(
        self, 
        paragraph: List[Dict], 
        available_frames: List[Dict],
        used_frames: set
    ) -> Optional[Dict]:
        """Find the best frame to accompany a paragraph."""
        
        paragraph_start = paragraph[0]['start_time']
        paragraph_end = paragraph[-1]['end_time']
        
        # Find frames that fall within or near the paragraph timespan
        candidate_frames = []
        for frame in available_frames:
            if frame['timestamp'] in used_frames:
                continue
                
            # Frame should be within paragraph or close to it
            if (paragraph_start <= frame['timestamp'] <= paragraph_end or
                abs(frame['timestamp'] - paragraph_start) <= 30 or
                abs(frame['timestamp'] - paragraph_end) <= 30):
                candidate_frames.append(frame)
        
        if not candidate_frames:
            return None
        
        # Prefer frames with lower face ratios (more visual content)
        best_frame = min(candidate_frames, key=lambda x: x['face_ratio'])
        return best_frame
    
    def _find_relevant_frames_for_paragraph(self, paragraph: List[Dict], sorted_frames: List[Dict], used_frames: set) -> List[Dict]:
        """Find all relevant frames for a paragraph, including clustered ones."""
        paragraph_start = paragraph[0]['start_time']
        paragraph_end = paragraph[-1]['end_time']
        
        # Find frames that overlap with this paragraph timeframe
        candidate_frames = []
        for frame in sorted_frames:
            if frame['timestamp'] in used_frames:
                continue
                
            # Include frames within the paragraph timeframe
            if paragraph_start <= frame['timestamp'] <= paragraph_end:
                candidate_frames.append(frame)
            # Also include frames slightly before/after (context)
            elif abs(frame['timestamp'] - paragraph_start) <= 10 or abs(frame['timestamp'] - paragraph_end) <= 10:
                candidate_frames.append(frame)
        
        if not candidate_frames:
            return []
        
        # For clustered images, return up to 3 from the same time window
        # For regular images, return the best one
        clustered_frames = [f for f in candidate_frames if f.get('is_clustered', False)]
        regular_frames = [f for f in candidate_frames if not f.get('is_clustered', False)]
        
        result = []
        
        # Add best regular frame
        if regular_frames:
            best_regular = min(regular_frames, key=lambda x: x['face_ratio'])
            result.append(best_regular)
        
        # Add clustered frames (up to 3)
        if clustered_frames:
            # Sort by score and take best 3
            clustered_frames.sort(key=lambda x: x['score'], reverse=True)
            result.extend(clustered_frames[:3])
        
        return result
    
    def _generate_clustered_images_markdown(self, frames: List[Dict], paragraph: List[Dict]) -> str:
        """Generate markdown for clustered images displayed side by side."""
        image_tags = []
        
        for frame in frames:
            alt_text = self._generate_alt_text(frame, paragraph)
            image_path = self._get_hugo_image_path(frame)
            
            if self.config.get('use_hugo_shortcodes', False):
                image_tags.append(f'{{{{< figure src="{image_path}" alt="{alt_text}" width="300" class="inline" >}}}}')
            else:
                image_tags.append(f'<img src="{image_path}" alt="{alt_text}" width="300" style="display: inline-block; margin: 5px;">')
        
        # Wrap in a container div for better layout
        if self.config.get('use_hugo_shortcodes', False):
            return '\n'.join(image_tags)
        else:
            return f'<div class="image-cluster">\n{"".join(image_tags)}\n</div>'
    
    def _generate_image_markdown(self, frame: Dict, paragraph: List[Dict]) -> str:
        """Generate markdown for an image with appropriate alt text."""
        
        # Generate descriptive alt text based on context
        alt_text = self._generate_alt_text(frame, paragraph)
        
        # Get relative path for Hugo
        image_path = self._get_hugo_image_path(frame)
        
        # Check if this is a clustered image (from jump cuts/rapid content)
        is_clustered = frame.get('is_clustered', False)
        
        # Generate markdown with Hugo shortcode or standard markdown
        if self.config.get('use_hugo_shortcodes', False):
            if is_clustered:
                return f'{{{{< figure src="{image_path}" alt="{alt_text}" width="400" class="inline" >}}}}'
            else:
                return f'{{{{< figure src="{image_path}" alt="{alt_text}" >}}}}'
        else:
            if is_clustered:
                return f'<img src="{image_path}" alt="{alt_text}" width="400" style="display: inline-block; margin: 5px;">'
            else:
                return f'![{alt_text}]({image_path})'
    
    def _generate_alt_text(self, frame: Dict, paragraph: List[Dict]) -> str:
        """Generate descriptive alt text for an image."""
        
        # Extract key topics from paragraph text
        paragraph_text = ' '.join(seg['text'] for seg in paragraph)
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = paragraph_text.lower().split()
        
        # Common technical/visual keywords that might indicate diagram content
        visual_keywords = [
            'diagram', 'chart', 'graph', 'visualization', 'screen', 'interface',
            'dashboard', 'workflow', 'process', 'architecture', 'design',
            'example', 'demonstration', 'slide', 'presentation'
        ]
        
        found_keywords = [word for word in visual_keywords if word in paragraph_text.lower()]
        
        if found_keywords:
            return f"Visual content showing {', '.join(found_keywords[:2])}"
        else:
            # Fallback to timestamp-based description
            timestamp_str = f"{frame['timestamp']:.0f}s"
            return f"Visual content from video at {timestamp_str}"
    
    def _get_hugo_image_path(self, frame: Dict) -> str:
        """Get the Hugo-compatible path for an image."""
        
        # If we have a bundle path (relative to the post), use that
        if 'bundle_path' in frame:
            return frame['bundle_path']
        
        # Fallback to original behavior for compatibility
        image_path = frame.get('optimized_path', frame['path'])
        filename = os.path.basename(image_path)
        static_path = self.config.get('hugo_static_path', 'static/images')
        
        # Return path relative to Hugo static directory
        return f"/{static_path.replace('static/', '')}/{filename}"
    
    def generate_config_template(self, output_path: str) -> None:
        """Generate a configuration template file."""
        
        config_template = {
            'video_processing': {
                'frame_sample_interval': 15,
                'min_face_ratio': 0.4,
                'max_face_ratio': 0.2,
                'face_detection_confidence': 0.5
            },
            'image_settings': {
                'quality': 95,
                'max_width': 1920,
                'max_height': 1080
            },
            'hugo_settings': {
                'static_path': 'static/images',
                'content_path': 'content/posts',
                'use_hugo_shortcodes': False
            },
            'transcript_settings': {
                'context_window': 30
            },
            'front_matter_defaults': {
                'tags': ['video', 'auto-generated'],
                'categories': ['video'],
                'author': 'YouTube2Hugo'
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated config template: {output_path}")
    
    def _cleanup_unused_images(self, bundle_dir: str, blog_content: str) -> None:
        """Remove image files from bundle directory that aren't referenced in the blog post."""
        import glob
        
        # Extract all image references from the blog content
        referenced_images = self._extract_referenced_images(blog_content)
        
        # Find all image files in the bundle directory
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']
        all_images = []
        
        for pattern in image_patterns:
            all_images.extend(glob.glob(os.path.join(bundle_dir, pattern)))
        
        # Get just the filenames for comparison
        all_image_files = [os.path.basename(img) for img in all_images]
        
        # Find unused images
        unused_images = []
        for image_file in all_image_files:
            if not any(image_file in ref for ref in referenced_images):
                unused_images.append(image_file)
        
        # Remove unused images
        removed_count = 0
        total_size_saved = 0
        
        for unused_image in unused_images:
            image_path = os.path.join(bundle_dir, unused_image)
            try:
                # Get file size before deletion
                file_size = os.path.getsize(image_path)
                os.remove(image_path)
                removed_count += 1
                total_size_saved += file_size
                logger.info(f"Removed unused image: {unused_image}")
            except OSError as e:
                logger.warning(f"Could not remove {unused_image}: {e}")
        
        if removed_count > 0:
            size_mb = total_size_saved / (1024 * 1024)
            logger.info(f"Cleanup complete: Removed {removed_count} unused images, saved {size_mb:.2f} MB")
        else:
            logger.info("No unused images found - all images are referenced in the blog post")
    
    def _extract_referenced_images(self, content: str) -> List[str]:
        """Extract all image filenames referenced in markdown content."""
        import re
        
        # Match ![alt text](filename) pattern and extract just the filename
        image_pattern = r'!\[.*?\]\(([^)]+)\)'
        matches = re.findall(image_pattern, content)
        
        # Extract just the filename from full paths
        referenced_files = []
        for match in matches:
            # Handle both relative paths and just filenames
            filename = os.path.basename(match)
            referenced_files.append(filename)
        
        return referenced_files