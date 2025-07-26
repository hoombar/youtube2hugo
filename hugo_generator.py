"""Hugo blog post generator module."""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml
import logging
import cv2
import numpy as np
from PIL import Image
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HugoGenerator:
    """Handles generation of Hugo-compatible markdown files."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.blog_formatter = None  # Initialize only when needed
        self.modal_added = False  # Track if modal has been added to avoid duplicates
    
    def _get_blog_formatter(self):
        """Lazy initialization of blog formatter to avoid unnecessary Gemini API initialization."""
        if self.blog_formatter is None:
            from blog_formatter import BlogFormatter
            self.blog_formatter = BlogFormatter(self.config)
        return self.blog_formatter
    
    def _get_image_modal_html(self):
        """Generate image modal HTML and JavaScript - only once per blog post."""
        if self.modal_added:
            return ""
        
        self.modal_added = True
        return '''<!-- Image Modal for Click-to-Enlarge -->
<div id="imageModal" style="display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); cursor: pointer;" onclick="closeImageModal()">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 90%; max-height: 90%;">
        <img id="modalImage" src="" alt="" style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <div id="modalCaption" style="color: white; text-align: center; margin-top: 10px; font-size: 16px;"></div>
    </div>
    <span style="position: absolute; top: 15px; right: 35px; color: white; font-size: 40px; font-weight: bold; cursor: pointer; user-select: none;" onclick="closeImageModal()">&times;</span>
</div>

<script>
function openImageModal(src, alt) {
    document.getElementById('imageModal').style.display = 'block';
    document.getElementById('modalImage').src = src;
    document.getElementById('modalCaption').textContent = alt;
    document.body.style.overflow = 'hidden';
}

function closeImageModal() {
    document.getElementById('imageModal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Close modal on Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeImageModal();
    }
});
</script>'''
    
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
        
        # Reset modal flag for new blog post
        self.modal_added = False
        
        # Create page bundle directory structure with kebab-case naming
        bundle_dir = self._create_page_bundle_structure(output_path, title)
        
        # Copy images to bundle directory
        self._copy_images_to_bundle(frame_data, bundle_dir)
        
        # Generate front matter
        front_matter = self._generate_front_matter(
            title, video_info, front_matter_data
        )
        
        # Check if frames have semantic section information
        has_semantic_info = any(f.get('section_title') for f in frame_data)
        
        if has_semantic_info:
            # Generate content with semantic frame placement
            raw_content = self._generate_content_with_semantic_frames(
                transcript_segments, frame_data, bundle_dir
            )
        else:
            # Fallback to traditional temporal placement
            raw_content = self._generate_content_with_images(
                transcript_segments, frame_data, bundle_dir
            )
        
        # Format content as blog post with Gemini (applies transcript corrections and blog structure)
        formatted_content = self._get_blog_formatter().format_content_with_images(
            raw_content, title, frame_data
        )
        
        # Apply template if provided (simple substitution, no additional Gemini processing)
        if template_path:
            template_variables = {
                'title': title,
                'date': datetime.now().strftime('%Y-%m-%dT00:00:00Z'),
                'content': formatted_content
            }
            # Add custom front matter variables
            if front_matter_data:
                template_variables.update(front_matter_data)
            
            # Simple template substitution without additional Gemini processing
            final_content = self._apply_simple_template(
                template_path, template_variables
            )
        else:
            # Use default front matter + content structure
            final_content = f"---\n{front_matter}\n---\n\n{formatted_content}"
        
        # Add modal HTML at the end if images were used
        modal_html = self._get_image_modal_html()
        if modal_html:
            final_content += f"\n\n{modal_html}"
        
        # Write index.md file in bundle directory
        index_path = os.path.join(bundle_dir, 'index.md')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Clean up unused images from the bundle directory
        self._cleanup_unused_images(bundle_dir, final_content)
        
        logger.info(f"Generated Hugo page bundle: {bundle_dir}")
        logger.info(f"Blog post created at: {index_path}")
        return final_content
    
    def generate_blog_post_with_formatted_content(
        self, 
        title: str,
        formatted_content: str,
        frame_data: List[Dict],
        video_info: Dict,
        output_path: str,
        front_matter_data: Optional[Dict] = None,
        template_path: Optional[str] = None
    ) -> str:
        """Generate a complete Hugo blog post using pre-formatted content (skips transcript formatting)."""
        
        # Reset modal flag for new blog post
        self.modal_added = False
        
        # Create page bundle directory structure with kebab-case naming
        bundle_dir = self._create_page_bundle_structure(output_path, title)
        
        # Copy images to bundle directory
        self._copy_images_to_bundle(frame_data, bundle_dir)
        
        # Generate front matter
        front_matter = self._generate_front_matter(
            title, video_info, front_matter_data
        )
        
        # Insert frames into the pre-formatted content
        logger.info("📝 Inserting frames into pre-formatted blog content...")
        content_with_images = self._insert_frames_into_formatted_content(
            formatted_content, frame_data, bundle_dir
        )
        
        # Apply template if provided (simple substitution, no additional Gemini processing)
        if template_path:
            template_variables = {
                'title': title,
                'date': datetime.now().strftime('%Y-%m-%dT00:00:00Z'),
                'content': content_with_images
            }
            # Add custom front matter variables
            if front_matter_data:
                template_variables.update(front_matter_data)
            
            # Simple template substitution without additional Gemini processing
            final_content = self._apply_simple_template(
                template_path, template_variables
            )
        else:
            # Standard format: front matter + content
            final_content = f"{front_matter}\n\n{content_with_images}"
        
        # Add modal HTML at the end if images were used
        modal_html = self._get_image_modal_html()
        if modal_html:
            final_content += f"\n\n{modal_html}"
        
        # Write index.md file in bundle directory
        index_path = os.path.join(bundle_dir, 'index.md')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Clean up unused images from the bundle directory
        self._cleanup_unused_images(bundle_dir, final_content)
        
        logger.info(f"Generated Hugo page bundle with pre-formatted content: {bundle_dir}")
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
        
        frames_to_copy = [f for f in frame_data if f.get('should_include', False)]
        logger.info(f"🖼️  IMAGE COPYING: {len(frames_to_copy)} frames marked for copying out of {len(frame_data)} total")
        
        copied_count = 0
        for frame in frames_to_copy:
            source_path = frame.get('optimized_path', frame['path'])
            if not os.path.exists(source_path):
                logger.warning(f"⚠️  Source image not found: {source_path}")
                continue
                
            filename = os.path.basename(source_path)
            dest_path = os.path.join(bundle_dir, filename)
            
            try:
                shutil.copy2(source_path, dest_path)
                
                # Update frame data with new bundle-relative path
                frame['bundle_path'] = filename
                copied_count += 1
                
                logger.info(f"✅ Copied image to bundle: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to copy {source_path}: {e}")
        
        logger.info(f"📊 IMAGE COPYING SUMMARY: {copied_count} images copied to bundle directory")
    
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
        logger.info(f"🖼️  CONTENT GENERATION: Processing {len(frame_data)} frames...")
        
        # Debug logging for should_include filtering
        frames_with_include = [f for f in frame_data if f.get('should_include', False)]
        frames_without_include = [f for f in frame_data if not f.get('should_include', False)]
        
        if frames_without_include:
            logger.warning(f"⚠️  FILTERED OUT {len(frames_without_include)} frames missing should_include=True:")
            for frame in frames_without_include[:5]:  # Show first 5
                timestamp = frame.get('timestamp', 'unknown')
                should_include = frame.get('should_include', 'missing')
                logger.warning(f"    - {timestamp}s: should_include={should_include}")
        
        sorted_frames = sorted(frames_with_include, key=lambda x: x['timestamp'])
        logger.info(f"📍 CONTENT GENERATION: Using {len(sorted_frames)} frames with should_include=True")
        
        # Group transcript segments into paragraphs
        paragraphs = self._group_segments_into_paragraphs(transcript_segments)
        logger.info(f"📝 CONTENT STRUCTURE: {len(paragraphs)} paragraphs from transcript")
        
        total_frames_placed = 0
        for i, paragraph in enumerate(paragraphs):
            paragraph_start = paragraph[0]['start_time']
            paragraph_end = paragraph[-1]['end_time']
            logger.info(f"📝 PARAGRAPH {i+1}: {paragraph_start:.1f}s-{paragraph_end:.1f}s")
            # Add paragraph text
            paragraph_text = self._format_paragraph_text(paragraph)
            content_parts.append(paragraph_text)
            
            # Find all relevant frames for this paragraph (including clustered ones)
            relevant_frames = self._find_relevant_frames_for_paragraph(
                paragraph, sorted_frames, used_frames
            )
            
            if relevant_frames:
                # Display all images in a grid layout if multiple, or single if just one
                if len(relevant_frames) == 1:
                    # Single image - use standard markdown
                    frame = relevant_frames[0]
                    image_markdown = self._generate_single_image_markdown(frame, paragraph)
                    content_parts.append(image_markdown)
                    used_frames.add(frame['timestamp'])
                    total_frames_placed += 1
                    logger.info(f"  🖼️  PLACED single frame {frame['timestamp']:.1f}s in paragraph {i+1}")
                else:
                    # Multiple images - use grid layout
                    grid_markdown = self._generate_image_grid_markdown(relevant_frames, paragraph)
                    content_parts.append(grid_markdown)
                    for frame in relevant_frames:
                        used_frames.add(frame['timestamp'])
                        total_frames_placed += 1
                        logger.info(f"  🖼️  PLACED grid frame {frame['timestamp']:.1f}s in paragraph {i+1}")
                    logger.info(f"  📐 GRID: {len(relevant_frames)} images in grid layout for paragraph {i+1}")
        
        logger.info(f"📊 CONTENT GENERATION SUMMARY: {total_frames_placed} frames placed in {len(paragraphs)} paragraphs")
        unused_frames = len(sorted_frames) - total_frames_placed
        if unused_frames > 0:
            logger.warning(f"⚠️  {unused_frames} frames were not placed (no matching transcript timing)")
        
        return '\n\n'.join(content_parts)
    
    def _generate_content_with_semantic_frames(
        self, 
        transcript_segments: List[Dict], 
        frame_data: List[Dict],
        bundle_dir: Optional[str] = None
    ) -> str:
        """Generate blog content using semantically selected frames with section information."""
        
        logger.info(f"🧠 SEMANTIC CONTENT GENERATION: Processing {len(frame_data)} semantic frames...")
        
        # Group frames by their semantic sections
        section_frames = {}
        for frame in frame_data:
            section_title = frame.get('section_title', 'General Content')
            if section_title not in section_frames:
                section_frames[section_title] = []
            section_frames[section_title].append(frame)
        
        logger.info(f"📋 Found {len(section_frames)} semantic sections with frames")
        
        # Group transcript into traditional paragraphs for text content
        paragraphs = self._group_segments_into_paragraphs(transcript_segments)
        
        content_parts = []
        used_frame_timestamps = set()
        
        # Generate content paragraph by paragraph, inserting semantic frames
        for i, paragraph in enumerate(paragraphs):
            paragraph_start = paragraph[0]['start_time']
            paragraph_end = paragraph[-1]['end_time']
            
            # Add paragraph text
            paragraph_text = self._format_paragraph_text(paragraph)
            content_parts.append(paragraph_text)
            
            # Find semantic frames that best match this paragraph's timeframe
            relevant_frames = []
            for section_title, frames in section_frames.items():
                for frame in frames:
                    timestamp = frame['timestamp']
                    # Check if frame falls within or near this paragraph
                    if (paragraph_start <= timestamp <= paragraph_end or
                        abs(timestamp - paragraph_start) <= 30 or  # 30s window
                        abs(timestamp - paragraph_end) <= 30):
                        
                        if timestamp not in used_frame_timestamps:
                            relevant_frames.append(frame)
                            used_frame_timestamps.add(timestamp)
            
            # Sort relevant frames by timestamp
            relevant_frames.sort(key=lambda x: x['timestamp'])
            
            if relevant_frames:
                # Generate frame content
                if len(relevant_frames) == 1:
                    frame = relevant_frames[0]
                    image_markdown = self._generate_semantic_frame_markdown(frame, paragraph)
                    content_parts.append(image_markdown)
                    logger.info(f"  🖼️  PLACED semantic frame {frame['timestamp']:.1f}s from section '{frame.get('section_title', 'Unknown')}' in paragraph {i+1}")
                else:
                    # Multiple frames - use grid layout
                    grid_markdown = self._generate_semantic_frame_grid_markdown(relevant_frames, paragraph)
                    content_parts.append(grid_markdown)
                    logger.info(f"  📐 PLACED {len(relevant_frames)} semantic frames in grid for paragraph {i+1}")
        
        # Add any remaining frames that weren't placed (orphaned frames)
        orphaned_frames = []
        for section_title, frames in section_frames.items():
            for frame in frames:
                if frame['timestamp'] not in used_frame_timestamps:
                    orphaned_frames.append(frame)
        
        if orphaned_frames:
            logger.info(f"  📎 Adding {len(orphaned_frames)} orphaned semantic frames at end")
            content_parts.append("## Additional Content")
            
            if len(orphaned_frames) == 1:
                frame = orphaned_frames[0]
                image_markdown = self._generate_semantic_frame_markdown(frame, [])
                content_parts.append(image_markdown)
            else:
                grid_markdown = self._generate_semantic_frame_grid_markdown(orphaned_frames, [])
                content_parts.append(grid_markdown)
        
        logger.info(f"📊 SEMANTIC CONTENT SUMMARY: Generated content with frames from {len(section_frames)} sections")
        return '\n\n'.join(content_parts)
    
    def _generate_semantic_frame_markdown(self, frame: Dict, paragraph: List[Dict]) -> str:
        """Generate markdown for a semantic frame with section context."""
        
        # Get frame info
        alt_text = self._generate_semantic_alt_text(frame, paragraph)
        image_path = self._get_hugo_image_path(frame)
        
        # Add section context if available
        section_title = frame.get('section_title', '')
        semantic_score = frame.get('semantic_score', 0)
        
        # Generate enhanced markdown with semantic info
        if self.config.get('use_hugo_shortcodes', False):
            return f'{{{{< figure src="{image_path}" alt="{alt_text}" caption="From: {section_title}" >}}}}'
        else:
            return f'![{alt_text}]({image_path})'
    
    def _generate_semantic_frame_grid_markdown(self, frames: List[Dict], paragraph: List[Dict]) -> str:
        """Generate grid markdown for multiple semantic frames."""
        
        if len(frames) == 1:
            return self._generate_semantic_frame_markdown(frames[0], paragraph)
        
        # Get grid configuration
        gap_size = self.config.get('image_grid', {}).get('gap_size', '10px')
        show_timestamps = self.config.get('image_grid', {}).get('show_timestamps', True)
        
        # Generate grid with semantic context
        image_elements = []
        for frame in frames:
            alt_text = self._generate_semantic_alt_text(frame, paragraph)
            image_path = self._get_hugo_image_path(frame)
            timestamp = frame['timestamp']
            section = frame.get('section_title', 'Unknown')
            
            if self.config.get('use_hugo_shortcodes', False):
                if show_timestamps:
                    caption = f"{section} ({timestamp:.1f}s)"
                    image_elements.append(f'{{{{< figure src="{image_path}" alt="{alt_text}" caption="{caption}" class="grid-image" >}}}}')
                else:
                    image_elements.append(f'{{{{< figure src="{image_path}" alt="{alt_text}" class="grid-image" >}}}}')
            else:
                # Calculate responsive width based on number of images
                num_images = len(frames)
                if num_images == 2:
                    img_width = "calc(48% - 5px)"
                elif num_images == 3:
                    img_width = "calc(32% - 7px)" 
                elif num_images == 4:
                    img_width = "calc(48% - 5px)"  # 2x2 grid
                elif num_images >= 5:
                    img_width = "calc(32% - 7px)"  # 3 per row
                else:
                    img_width = "calc(45% - 10px)"  # fallback
                
                img_style = f"width: {img_width} !important; height: auto !important; object-fit: cover; border-radius: 4px; display: inline-block !important; margin: 3px !important;"
                
                # Use simple img tags without div wrappers for cleaner output
                if show_timestamps:
                    image_elements.append(f'<img src="{image_path}" alt="{alt_text}" title="{section} ({timestamp:.1f}s)" style="{img_style}">')
                else:
                    image_elements.append(f'<img src="{image_path}" alt="{alt_text}" style="{img_style}">')
        
        # Wrap in grid container
        grid_style = f"display: flex; flex-wrap: wrap; gap: {gap_size}; margin: 20px 0; justify-content: space-around;"
        return f'<div class="semantic-image-grid" style="{grid_style}">\n{chr(10).join(image_elements)}\n</div>'
    
    def _generate_semantic_alt_text(self, frame: Dict, paragraph: List[Dict]) -> str:
        """Generate alt text for semantic frames using section context."""
        
        section_title = frame.get('section_title', 'Content')
        timestamp = frame['timestamp']
        
        # Use section title as primary context
        if section_title and section_title != 'Unknown':
            return f"{section_title} demonstration at {timestamp:.1f}s"
        else:
            # Fallback to traditional alt text generation
            return self._generate_alt_text(frame, paragraph)
    
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
        
        logger.debug(f"🔍 FINDING frames for paragraph {paragraph_start:.1f}s-{paragraph_end:.1f}s ({len(sorted_frames)} total frames available)")
        
        # Find frames that overlap with this paragraph timeframe
        candidate_frames = []
        excluded_frames = []
        
        for frame in sorted_frames:
            if frame['timestamp'] in used_frames:
                continue
                
            # Include frames within the paragraph timeframe
            if paragraph_start <= frame['timestamp'] <= paragraph_end:
                candidate_frames.append(frame)
                logger.debug(f"  ✅ INCLUDED {frame['timestamp']:.1f}s (within paragraph)")
            # Also include frames slightly before/after (context) - smaller window
            elif abs(frame['timestamp'] - paragraph_start) <= 5 or abs(frame['timestamp'] - paragraph_end) <= 5:
                candidate_frames.append(frame)
                logger.debug(f"  ✅ INCLUDED {frame['timestamp']:.1f}s (within 5s context)")
            else:
                excluded_frames.append(frame['timestamp'])
        
        if excluded_frames and len(excluded_frames) <= 10:  # Don't spam if too many
            logger.debug(f"  ❌ EXCLUDED frames: {[f'{ts:.1f}s' for ts in excluded_frames[:10]]}")
        elif excluded_frames:
            logger.debug(f"  ❌ EXCLUDED {len(excluded_frames)} frames (timestamps don't match paragraph)")
        
        logger.debug(f"🎯 PARAGRAPH {paragraph_start:.1f}s-{paragraph_end:.1f}s: {len(candidate_frames)} relevant frames found")
        
        if not candidate_frames:
            return []
        
        # For clustered images, return up to 3 from the same time window
        # For regular images, return the best one
        clustered_frames = [f for f in candidate_frames if f.get('is_clustered', False)]
        regular_frames = [f for f in candidate_frames if not f.get('is_clustered', False)]
        
        result = []
        
        # Add regular frames - allow multiple for rapid sequences with good scores
        if regular_frames:
            # Sort by score (highest first)
            regular_frames.sort(key=lambda x: x['score'], reverse=True)
            
            # If there are multiple high-scoring frames, include more of them
            high_score_frames = [f for f in regular_frames if f['score'] >= 500]  # High-quality content
            medium_score_frames = [f for f in regular_frames if 200 <= f['score'] < 500]
            
            if len(high_score_frames) >= 3:
                # Many high-quality frames - take up to 4 best ones
                result.extend(high_score_frames[:4])
                logger.debug(f"  📸 MULTIPLE high-quality frames: added {len(high_score_frames[:4])} frames")
            elif len(high_score_frames) >= 2:
                # Some high-quality frames - take up to 3
                result.extend(high_score_frames[:3])
                logger.debug(f"  📸 MULTIPLE high-quality frames: added {len(high_score_frames[:3])} frames")
            elif len(high_score_frames) == 1 and len(medium_score_frames) >= 2:
                # 1 high + multiple medium - take 1 high + 2 medium
                result.extend(high_score_frames[:1])
                result.extend(medium_score_frames[:2])
                logger.debug(f"  📸 MIXED quality frames: added 1 high + 2 medium frames")
            else:
                # Default: take single best frame (by lowest face ratio)
                best_regular = min(regular_frames, key=lambda x: x['face_ratio'])
                result.append(best_regular)
                logger.debug(f"  📸 SINGLE best frame: {best_regular['timestamp']:.1f}s (score={best_regular['score']:.1f})")
        
        # Add clustered frames (up to 3)
        if clustered_frames:
            # Sort by score and take best 3
            clustered_frames.sort(key=lambda x: x['score'], reverse=True)
            result.extend(clustered_frames[:3])
        
        # Apply similarity filtering to remove duplicate-looking images
        if len(result) > 1:
            similarity_threshold = self.config.get('image_similarity_threshold', 0.15)
            logger.debug(f"  🔍 SIMILARITY CHECK: Filtering {len(result)} frames for duplicates (threshold={similarity_threshold})...")
            result = self._remove_similar_images(result, similarity_threshold)
        
        return result
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate perceptual hash for image similarity detection using difference hash (dHash)."""
        try:
            # Load image and convert to grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return ""
            
            # Resize to 9x8 for difference hash (we need 8x8 differences)
            resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
            
            # Calculate horizontal differences (more sensitive than average-based)
            hash_bits = []
            for row in resized:
                for i in range(len(row) - 1):
                    # Compare adjacent pixels
                    hash_bits.append('1' if row[i] > row[i + 1] else '0')
            
            # Convert to hex string - handle large binary strings properly
            binary_string = ''.join(hash_bits)
            
            # Split into chunks to avoid int overflow
            hex_parts = []
            for i in range(0, len(binary_string), 60):  # Process 60 bits at a time
                chunk = binary_string[i:i+60]
                if chunk:
                    hex_parts.append(format(int(chunk, 2), 'x'))
            
            return ''.join(hex_parts)
            
        except Exception as e:
            logger.warning(f"Could not calculate hash for {image_path}: {e}")
            return ""
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two perceptual hashes (0.0 = completely different, 1.0 = identical)."""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0  # Return 0 (different) if hashes invalid
        
        # Count matching bits
        matching_bits = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        
        # Return normalized similarity (0.0 = completely different, 1.0 = identical)
        return matching_bits / len(hash1)
    
    def _remove_similar_images(self, frames: List[Dict], similarity_threshold: float = 0.15) -> List[Dict]:
        """Remove images that are too similar to other images in the same set."""
        if len(frames) <= 1:
            return frames
        
        # Calculate hashes for all frames
        frame_hashes = []
        for frame in frames:
            image_hash = self._calculate_image_hash(frame['path'])
            frame_hashes.append({
                'frame': frame,
                'hash': image_hash,
                'kept': True
            })
        
        # Compare each frame with all others and mark similar ones for removal
        for i, frame_data in enumerate(frame_hashes):
            if not frame_data['kept']:
                continue
                
            for j, other_frame_data in enumerate(frame_hashes[i+1:], i+1):
                if not other_frame_data['kept']:
                    continue
                
                similarity = self._calculate_hash_similarity(frame_data['hash'], other_frame_data['hash'])
                
                if similarity >= similarity_threshold:
                    # Keep the higher scoring frame, remove the lower scoring one
                    frame1_score = frame_data['frame'].get('score', 0)
                    frame2_score = other_frame_data['frame'].get('score', 0)
                    
                    if frame1_score >= frame2_score:
                        other_frame_data['kept'] = False
                        logger.debug(f"  🗑️  REMOVED similar frame {other_frame_data['frame']['timestamp']:.1f}s (similarity={similarity:.3f} >= {similarity_threshold}, score={frame2_score:.1f} < {frame1_score:.1f})")
                    else:
                        frame_data['kept'] = False
                        logger.debug(f"  🗑️  REMOVED similar frame {frame_data['frame']['timestamp']:.1f}s (similarity={similarity:.3f} >= {similarity_threshold}, score={frame1_score:.1f} < {frame2_score:.1f})")
                        break  # No need to check more if this frame is removed
        
        # Return only the frames that should be kept
        filtered_frames = [fd['frame'] for fd in frame_hashes if fd['kept']]
        
        removed_count = len(frames) - len(filtered_frames)
        if removed_count > 0:
            logger.debug(f"  🎯 SIMILARITY FILTER: Removed {removed_count} similar images, kept {len(filtered_frames)}")
        
        return filtered_frames
    
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
    
    def _generate_single_image_markdown(self, frame: Dict, paragraph: List[Dict]) -> str:
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
                return f'{{{{< figure src="{image_path}" alt="{alt_text}" width="50%" class="inline" >}}}}'
            else:
                return f'{{{{< figure src="{image_path}" alt="{alt_text}" width="50%" >}}}}'
        else:
            # Single image gets 50% width with click-to-enlarge modal
            img_style = "width: 50% !important; height: auto !important; object-fit: cover; border-radius: 4px; display: block !important; margin: 20px auto !important; cursor: pointer !important;"
            
            return f'<img src="{image_path}" alt="{alt_text}" style="{img_style}" onclick="openImageModal(\'{image_path}\', \'{alt_text}\')">'
    
    def _generate_image_grid_markdown(self, frames: List[Dict], paragraph: List[Dict]) -> str:
        """Generate markdown for multiple images displayed in a responsive grid."""
        if not frames:
            return ""
        
        if len(frames) == 1:
            return self._generate_single_image_markdown(frames[0], paragraph)
        
        # Get grid configuration options
        gap_size = self.config.get('image_grid', {}).get('gap_size', '10px')
        show_timestamps = self.config.get('image_grid', {}).get('show_timestamps', True)
        border_radius = self.config.get('image_grid', {}).get('border_radius', '4px')
        max_columns = self.config.get('image_grid', {}).get('max_columns', 3)
        include_css_reset = self.config.get('image_grid', {}).get('include_css_reset', True)
        use_flexbox_fallback = self.config.get('image_grid', {}).get('use_flexbox_fallback', False)
        
        # Determine grid layout based on number of images
        num_images = len(frames)
        
        # New sizing logic: 1 image = 50%, 2 images = 25% each, 3 images = ~16% each
        if use_flexbox_fallback:
            # Flexbox fallback for problematic themes
            grid_class = f"image-flex-{num_images}"
            if num_images == 1:
                grid_style = f"display: flex; justify-content: center; gap: {gap_size}; margin: 20px 0;"
                img_width = "50%"
            elif num_images == 2:
                grid_style = f"display: flex; justify-content: center; gap: {gap_size}; margin: 20px 0;"
                img_width = "25%"
            elif num_images == 3:
                grid_style = f"display: flex; justify-content: center; gap: {gap_size}; margin: 20px 0;"
                img_width = "16.666%"
            else:
                # Fallback for 4+ images (shouldn't happen with 3-image limit)
                grid_style = f"display: flex; flex-wrap: wrap; justify-content: center; gap: {gap_size}; margin: 20px 0;"
                img_width = "calc(25% - 5px)"
        else:
            # Standard CSS with right-float layout for text wrapping
            if num_images == 1:
                grid_class = "image-grid-1"
                grid_style = f"display: flex; justify-content: center; gap: {gap_size}; margin-left: 20px; margin-bottom: 15px; margin-top: 10px; width: 50%; float: right;"
                img_width = "100%"
            elif num_images == 2:
                grid_class = "image-grid-2"
                grid_style = f"display: flex; justify-content: center; gap: {gap_size}; margin-left: 20px; margin-bottom: 15px; margin-top: 10px; width: 50%; float: right;"
                img_width = "calc(50% - 2.5px)"
            elif num_images == 3:
                grid_class = "image-grid-3"
                grid_style = f"display: flex; justify-content: center; gap: {gap_size}; margin-left: 20px; margin-bottom: 15px; margin-top: 10px; width: 50%; float: right;"
                img_width = "calc(33.333% - 3.33px)"
            else:
                # Fallback for 4+ images (shouldn't happen with 3-image limit)
                grid_class = "image-grid-flex"
                grid_style = f"display: flex; flex-wrap: wrap; justify-content: center; gap: {gap_size}; margin-left: 20px; margin-bottom: 15px; margin-top: 10px; width: 50%; float: right;"
                img_width = f"calc({100/num_images:.1f}% - {5*num_images/num_images:.1f}px)"
        
        # Generate individual image elements
        image_elements = []
        for frame in frames:
            alt_text = self._generate_alt_text(frame, paragraph)
            image_path = self._get_hugo_image_path(frame)
            timestamp = frame['timestamp']
            
            if self.config.get('use_hugo_shortcodes', False):
                # Hugo shortcode with responsive sizing
                if show_timestamps:
                    image_elements.append(
                        f'{{{{< figure src="{image_path}" alt="{alt_text}" class="grid-image" caption="{timestamp:.1f}s" >}}}}'
                    )
                else:
                    image_elements.append(
                        f'{{{{< figure src="{image_path}" alt="{alt_text}" class="grid-image" >}}}}'
                    )
            else:
                # HTML img tag with click-to-enlarge modal functionality
                img_style = f"width: {img_width} !important; height: auto !important; object-fit: cover; border-radius: {border_radius}; display: block !important; margin: 0 !important; padding: 0 !important; cursor: pointer !important;"
                caption_style = "font-size: 0.8em !important; color: #666 !important; text-align: center !important; margin-top: 5px !important; margin-bottom: 0 !important; padding: 0 !important;"
                
                if show_timestamps:
                    image_elements.append(f'''<div class="grid-item" style="margin: 0 !important; padding: 0 !important;">
    <img src="{image_path}" alt="{alt_text}" style="{img_style}" onclick="openImageModal('{image_path}', '{alt_text}')">
    <div style="{caption_style}">{timestamp:.1f}s</div>
</div>''')
                else:
                    image_elements.append(f'''<div class="grid-item" style="margin: 0 !important; padding: 0 !important;">
    <img src="{image_path}" alt="{alt_text}" style="{img_style}" onclick="openImageModal('{image_path}', '{alt_text}')">
</div>''')
        
        # Combine into grid container
        if self.config.get('use_hugo_shortcodes', False):
            # For Hugo shortcodes, create a custom grid shortcode
            shortcode_params = f'class="{grid_class}" columns="{num_images}" gap="{gap_size}"'
            
            # Create individual image shortcodes
            image_shortcodes = []
            for frame in frames:
                alt_text = self._generate_alt_text(frame, paragraph)
                image_path = self._get_hugo_image_path(frame)
                timestamp = frame['timestamp']
                
                if show_timestamps:
                    image_shortcodes.append(f'{{{{< grid-image src="{image_path}" alt="{alt_text}" caption="{timestamp:.1f}s" >}}}}')
                else:
                    image_shortcodes.append(f'{{{{< grid-image src="{image_path}" alt="{alt_text}" >}}}}')
            
            return f'''{{{{< image-grid {shortcode_params} >}}}}
{chr(10).join(image_shortcodes)}
{{{{< /image-grid >}}}}'''
        else:
            # For HTML, include all styling inline for maximum compatibility with defensive CSS
            container_style = grid_style + " max-width: 100% !important; overflow: hidden !important; box-sizing: border-box !important; clear: both !important;"
            
            # Optional CSS reset to override theme interference
            css_reset = ""
            if include_css_reset:
                css_reset = f'''<style>
.{grid_class} {{
    display: grid !important;
    box-sizing: border-box !important;
    margin: 20px 0 !important;
    padding: 0 !important;
}}
.{grid_class} .grid-item {{
    box-sizing: border-box !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}
.{grid_class} .grid-item img {{
    width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    object-fit: cover !important;
}}
</style>
'''
            
            return f'''{css_reset}<div class="{grid_class}" style="{container_style}">
{chr(10).join(image_elements)}
</div>'''
    
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
        logger.info(f"🧹 CLEANUP: Found {len(referenced_images)} referenced images in blog content")
        
        # Find all image files in the bundle directory
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']
        all_images = []
        
        for pattern in image_patterns:
            all_images.extend(glob.glob(os.path.join(bundle_dir, pattern)))
        
        # Get just the filenames for comparison
        all_image_files = [os.path.basename(img) for img in all_images]
        logger.info(f"🧹 CLEANUP: Found {len(all_image_files)} image files in bundle directory: {all_image_files}")
        
        # Find unused images
        unused_images = []
        for image_file in all_image_files:
            if not any(image_file in ref for ref in referenced_images):
                unused_images.append(image_file)
                logger.debug(f"  ❌ UNUSED: {image_file} not found in references")
            else:
                logger.debug(f"  ✅ USED: {image_file} found in references")
        
        if unused_images:
            logger.warning(f"🧹 CLEANUP: {len(unused_images)} unused images will be removed: {unused_images}")
        
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
                logger.info(f"🗑️  Removed unused image: {unused_image}")
            except OSError as e:
                logger.warning(f"Could not remove {unused_image}: {e}")
        
        if removed_count > 0:
            size_mb = total_size_saved / (1024 * 1024)
            logger.info(f"🧹 Cleanup complete: Removed {removed_count} unused images, saved {size_mb:.2f} MB")
        else:
            logger.info("✅ No unused images found - all images are referenced in the blog post")
    
    def _extract_referenced_images(self, content: str) -> List[str]:
        """Extract all image filenames referenced in markdown content."""
        import re
        
        referenced_files = []
        
        # Match traditional markdown ![alt text](filename) pattern
        markdown_pattern = r'!\[.*?\]\(([^)]+)\)'
        markdown_matches = re.findall(markdown_pattern, content)
        
        # Match HTML <img src="filename"> pattern (used in grids)
        html_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        html_matches = re.findall(html_pattern, content)
        
        # Match Hugo shortcode {{< figure src="filename" >}} pattern
        hugo_pattern = r'\{\{<\s*figure\s+src=["\']([^"\']+)["\']'
        hugo_matches = re.findall(hugo_pattern, content)
        
        # Combine all matches
        all_matches = markdown_matches + html_matches + hugo_matches
        
        # Extract just the filename from full paths
        for match in all_matches:
            # Handle both relative paths and just filenames
            filename = os.path.basename(match)
            referenced_files.append(filename)
        
        # Remove duplicates
        referenced_files = list(set(referenced_files))
        
        logger.debug(f"🔍 IMAGE REFERENCES: Found {len(referenced_files)} unique image references: {referenced_files}")
        
        return referenced_files
    
    def _apply_simple_template(self, template_path: str, variables: Dict) -> str:
        """Apply template with simple variable substitution, no additional Gemini processing."""
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Simple variable substitution using string replacement
            result = template_content
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                result = result.replace(placeholder, str(value))
            
            logger.info(f"Applied template: {template_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply template {template_path}: {e}")
            # Fallback to default structure
            return f"---\ntitle: {variables.get('title', 'Untitled')}\ndate: {variables.get('date', '')}\n---\n\n{variables.get('content', '')}"
    
    def _insert_frames_into_formatted_content(self, formatted_content: str, frame_data: List[Dict], bundle_dir: str) -> str:
        """Insert frames into pre-formatted blog content at appropriate locations."""
        
        if not frame_data:
            return formatted_content
        
        # Split content into sections based on headers
        import re
        lines = formatted_content.split('\n')
        
        # Group frames by their section titles (from semantic selection)
        section_frames = {}
        for frame in frame_data:
            section_title = frame.get('section_title', 'General')
            if section_title not in section_frames:
                section_frames[section_title] = []
            section_frames[section_title].append(frame)
        
        # Insert frames after corresponding headers
        result_lines = []
        current_header = None
        self._pending_frames = []  # Initialize pending frames list
        
        for line in lines:
            # Check if this line is a header
            header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
            
            # Check if this line is content (not empty, not header)
            is_content_line = line.strip() and not header_match
            
            # If we have pending frames and this is the first content line after a header, insert frames inline
            if is_content_line and self._pending_frames:
                
                # Insert frames inline with the first content line for proper text wrapping
                frames_to_insert = self._pending_frames[:]
                self._pending_frames = []
                
                # Group frames by timestamp proximity to determine if they should be displayed together
                grouped_frames = self._group_frames_by_proximity(frames_to_insert)
                
                # Generate frame markdown for each group
                for frame_group in grouped_frames:
                    if len(frame_group) == 1:
                        # Single frame
                        frame_markdown = self._generate_semantic_frame_markdown_for_formatted_content(frame_group[0])
                        result_lines.append(frame_markdown)
                    else:
                        # Multiple frames - use grid layout
                        grid_markdown = self._generate_semantic_frame_grid_markdown_for_formatted_content(frame_group)
                        result_lines.append(grid_markdown)
                    
                logger.info(f"  🖼️  Inserted {len(frames_to_insert)} frames inline with content for text wrapping")
            
            result_lines.append(line)
            if header_match:
                header_title = header_match.group(2).strip()
                current_header = header_title
                
                # Find frames that belong to this section
                matching_frames = []
                for section_title, frames in section_frames.items():
                    # Use fuzzy matching for section titles
                    if (section_title.lower() in header_title.lower() or 
                        header_title.lower() in section_title.lower() or
                        section_title == header_title):
                        matching_frames.extend(frames)
                
                # Store frames to insert at the beginning of the first paragraph in this section
                if matching_frames:
                    # We'll insert these frames when we encounter the first paragraph content
                    # This allows text to wrap around the floated images properly
                    self._pending_frames.extend(matching_frames)
                    logger.info(f"  🖼️  Queued {len(matching_frames)} frames for section '{header_title}'")
                    
                    # Remove these frames from the pool so they don't get inserted again
                    for section_title in list(section_frames.keys()):
                        section_frames[section_title] = [f for f in section_frames[section_title] if f not in matching_frames]
                        if not section_frames[section_title]:
                            del section_frames[section_title]
        
        # Handle any remaining pending frames that didn't get inserted
        if self._pending_frames:
            logger.warning(f"⚠️  {len(self._pending_frames)} pending frames were not inserted (no content found)")
            for frame in self._pending_frames:
                frame_markdown = self._generate_semantic_frame_markdown_for_formatted_content(frame)
                result_lines.append(frame_markdown)
        
        # Add any remaining frames at the end
        remaining_frames = []
        for frames in section_frames.values():
            remaining_frames.extend(frames)
        
        if remaining_frames:
            result_lines.extend(['', '## Additional Images', ''])
            for frame in remaining_frames:
                image_markdown = self._generate_semantic_frame_markdown_for_formatted_content(frame)
                result_lines.append(image_markdown)
                result_lines.append('')
            logger.info(f"  🖼️  Added {len(remaining_frames)} remaining frames at end")
        
        return '\n'.join(result_lines)
    
    def _group_frames_by_proximity(self, frames: List[Dict]) -> List[List[Dict]]:
        """Group frames by paragraph information or timestamp proximity."""
        if not frames:
            return []
        
        # Sort frames by timestamp first
        sorted_frames = sorted(frames, key=lambda x: x.get('timestamp', 0))
        
        # Check if we have paragraph information
        has_paragraph_info = any(f.get('paragraph_index') is not None for f in sorted_frames)
        
        if has_paragraph_info:
            # Group by paragraph index and paragraph time ranges
            paragraph_groups = {}
            
            for frame in sorted_frames:
                paragraph_key = f"{frame.get('paragraph_index', 0)}_{frame.get('paragraph_start_time', 0)}_{frame.get('paragraph_end_time', 0)}"
                
                if paragraph_key not in paragraph_groups:
                    paragraph_groups[paragraph_key] = []
                
                paragraph_groups[paragraph_key].append(frame)
            
            # Convert to list of groups, respecting the 3-image limit per paragraph
            groups = []
            for paragraph_key, paragraph_frames in paragraph_groups.items():
                # Sort frames within the paragraph by timestamp
                paragraph_frames.sort(key=lambda x: x.get('timestamp', 0))
                
                # Split into groups of max 3 frames each
                for i in range(0, len(paragraph_frames), 3):
                    group = paragraph_frames[i:i+3]
                    groups.append(group)
            
            logger.info(f"🖼️ Grouped {len(frames)} frames by paragraph info into {len(groups)} display groups")
            for i, group in enumerate(groups):
                timestamps = [f"{f.get('timestamp', 0):.1f}s" for f in group]
                paragraph_indices = [str(f.get('paragraph_index', 'unknown')) for f in group]
                logger.info(f"  Group {i+1}: {len(group)} frames at {', '.join(timestamps)} (paragraphs: {', '.join(set(paragraph_indices))})")
            
        else:
            # Fallback to proximity-based grouping
            groups = []
            current_group = [sorted_frames[0]]
            
            # Group frames that are within 10 seconds of each other (same paragraph)
            proximity_threshold = 10.0  # seconds
            
            for i in range(1, len(sorted_frames)):
                current_frame = sorted_frames[i]
                last_frame_in_group = current_group[-1]
                
                time_diff = abs(current_frame.get('timestamp', 0) - last_frame_in_group.get('timestamp', 0))
                
                if time_diff <= proximity_threshold and len(current_group) < 3:  # Max 3 images per group
                    current_group.append(current_frame)
                else:
                    # Start a new group
                    groups.append(current_group)
                    current_group = [current_frame]
            
            # Add the last group
            if current_group:
                groups.append(current_group)
            
            logger.info(f"🖼️ Grouped {len(frames)} frames by proximity into {len(groups)} display groups")
            for i, group in enumerate(groups):
                timestamps = [f"{f.get('timestamp', 0):.1f}s" for f in group]
                logger.info(f"  Group {i+1}: {len(group)} frames at {', '.join(timestamps)}")
        
        return groups
    
    def _generate_semantic_frame_markdown_for_formatted_content(self, frame: Dict) -> str:
        """Generate markdown for a single semantic frame in formatted content with consistent right-float layout."""
        
        image_path = self._get_hugo_image_path(frame)
        timestamp = frame['timestamp']
        section_title = frame.get('section_title', 'Content')
        
        # Generate alt text
        alt_text = f"{section_title} demonstration at {timestamp:.1f}s"
        
        if self.config.get('use_hugo_shortcodes', False):
            return f'{{{{< figure src="{image_path}" alt="{alt_text}" caption="From: {section_title}" class="float-right" width="50%" >}}}}'
        else:
            # Use HTML with inline styles - always float right at 50% width with click-to-enlarge
            # Add some top margin to prevent images from sitting too close to headers
            return f'<img src="{image_path}" alt="{alt_text}" style="width: 50%; float: right; margin-left: 20px; margin-bottom: 15px; margin-top: 10px; border-radius: 4px; cursor: pointer;" title="{section_title} at {timestamp:.1f}s" onclick="openImageModal(\'{image_path}\', \'{alt_text}\')">'
    
    def _generate_semantic_frame_grid_markdown_for_formatted_content(self, frames: List[Dict]) -> str:
        """Generate markdown for multiple frames in formatted content using horizontal layout."""
        
        if not frames:
            return ""
        
        if len(frames) == 1:
            return self._generate_semantic_frame_markdown_for_formatted_content(frames[0])
        
        # Calculate width for each image based on count
        # Since the container is 50% and floated right, images should fill the container
        num_images = len(frames)
        if num_images == 2:
            img_width = "calc(50% - 2.5px)"  # Account for 5px gap between images
        elif num_images == 3:
            img_width = "calc(33.333% - 3.33px)"  # Account for gaps
        else:
            img_width = f"calc({100/num_images:.1f}% - {5*num_images/num_images:.1f}px)"  # Fallback
        
        # Generate individual image elements
        image_elements = []
        for frame in frames:
            image_path = self._get_hugo_image_path(frame)
            timestamp = frame['timestamp']
            section_title = frame.get('section_title', 'Content')
            alt_text = f"{section_title} demonstration at {timestamp:.1f}s"
            
            if self.config.get('use_hugo_shortcodes', False):
                image_elements.append(f'{{{{< figure src="{image_path}" alt="{alt_text}" caption="{timestamp:.1f}s" class="grid-image" width="{img_width}" >}}}}')
            else:
                # HTML img tag with horizontal layout styling
                img_style = f"width: {img_width} !important; height: auto !important; object-fit: cover; border-radius: 4px; display: block !important; margin: 0 !important; cursor: pointer !important;"
                image_elements.append(f'<img src="{image_path}" alt="{alt_text}" style="{img_style}" title="{section_title} at {timestamp:.1f}s" onclick="openImageModal(\'{image_path}\', \'{alt_text}\')">')
        
        # Wrap in container for horizontal layout with right float
        if self.config.get('use_hugo_shortcodes', False):
            return f'''{{{{< image-grid columns="{num_images}" class="float-right" >}}}}
{chr(10).join(image_elements)}
{{{{< /image-grid >}}}}'''
        else:
            # Use flexbox container that floats right and allows text wrapping
            # Total width is still 50% but split among multiple images
            container_style = f"display: flex; justify-content: center; gap: 5px; margin-left: 20px; margin-bottom: 15px; margin-top: 10px; width: 50%; float: right; flex-wrap: wrap;"
            return f'''<div style="{container_style}">
{chr(10).join(image_elements)}
</div>'''
    
    def _generate_frame_grid_markdown_for_formatted_content(self, frames: List[Dict]) -> str:
        """Generate markdown for multiple frames in formatted content using consistent right-float layout."""
        
        # For multiple frames, place them individually with consistent right-float
        # They will stack vertically on the right side
        frame_markdowns = []
        for frame in frames:
            frame_markdown = self._generate_semantic_frame_markdown_for_formatted_content(frame)
            frame_markdowns.append(frame_markdown)
        
        # Add some spacing between frames and a clear div at the end
        result = '\n\n'.join(frame_markdowns)
        result += '\n\n<div style="clear: both;"></div>'  # Clear floats after images
        
        return result