#!/usr/bin/env python3
"""
Hybrid Blog Creator - AI-powered content processing with manual frame selection.

This tool combines the best of both worlds:
- AI: Transcript cleaning, technical term correction, section generation
- Human: Visual frame selection for each section
- Automation: Final blog post generation with selected frames

Workflow:
1. Process video transcript and create sections
2. Extract candidate frames for each section
3. Web interface for manual frame selection
4. Generate final Hugo blog post with selected frames
"""

import os
import json
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime

# Import existing modules
from transcript_extractor import TranscriptExtractor
from blog_formatter import BlogFormatter
from video_processor import VideoProcessor
from hugo_generator import HugoGenerator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class HybridBlogCreator:
    """Integrated tool for AI-assisted blog creation with manual frame selection."""
    
    def __init__(self, config_path: str = "config.local.yaml"):
        from config import Config
        # Load both nested and flattened config for compatibility
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Also load flattened config for template path compatibility
        self.flattened_config = Config.load_local_config()
        
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Session storage
        self.sessions_dir = "blog_sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Current session data
        self.current_session = None
        
    def setup_routes(self):
        """Setup Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            return render_template('hybrid_blog_creator.html')
        
        @self.app.route('/process_video', methods=['POST'])
        def process_video():
            """Process video and prepare sections with frames for selection."""
            try:
                data = request.json
                video_path = data['video_path']
                title = data.get('title', '')
                processing_mode = data.get('processing_mode', 'smart')
                
                session_data = self.create_blog_session(video_path, title, processing_mode)
                return jsonify(session_data)
                
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/images/<path:filename>')
        def serve_image(filename):
            """Serve frame images."""
            if self.current_session:
                session_dir = os.path.join(self.sessions_dir, self.current_session['session_id'])
                frames_dir = os.path.join(session_dir, 'frames')
                if os.path.exists(os.path.join(frames_dir, filename)):
                    return send_from_directory(frames_dir, filename)
            
            # Fallback to temp_frames
            if os.path.exists(os.path.join('temp_frames', filename)):
                return send_from_directory('temp_frames', filename)
            
            return "Image not found", 404
        
        @self.app.route('/save_frame_selections', methods=['POST'])
        def save_frame_selections():
            """Save frame selections for each section."""
            try:
                data = request.json
                self.save_selections(data)
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Error saving selections: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/generate_blog', methods=['POST'])
        def generate_blog():
            """Generate final blog post with selected frames."""
            try:
                blog_data = self.generate_final_blog()
                return jsonify(blog_data)
            except Exception as e:
                logger.error(f"Error generating blog: {e}")
                return jsonify({'error': str(e)}), 500
    
    def create_blog_session(self, video_path: str, title: str, processing_mode: str = 'smart') -> Dict:
        """Create a new blog creation session."""
        logger.info(f"🎬 Creating blog session for: {video_path}")
        
        # Create session
        session_id = f"{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Step 1: Extract and process transcript
        logger.info("📝 Processing transcript...")
        transcript_extractor = TranscriptExtractor(self.config)
        transcript_segments = transcript_extractor.extract_transcript(video_path)
        
        # Step 2: AI-powered content processing
        logger.info("🤖 AI processing: cleaning transcript and creating sections...")
        blog_formatter = BlogFormatter(self.config)
        
        try:
            # Generate structured blog content (this should be independent of frame processing mode!)
            logger.info(f"🤖 Starting AI content processing (processing_mode: {processing_mode})")
            logger.info(f"📊 Input transcript segments: {len(transcript_segments)}")
            
            # Log first few transcript segments for debugging
            if transcript_segments:
                sample_segments = transcript_segments[:3]
                for i, seg in enumerate(sample_segments):
                    logger.debug(f"Transcript sample {i+1}: '{seg.get('text', '')[:100]}...'")
            
            blog_content = blog_formatter.format_transcript_content(
                transcript_segments, title or f"Blog Post: {Path(video_path).stem}"
            )
            
            logger.info(f"📝 AI processing completed, blog_content length: {len(blog_content)} chars")
            
            boundary_map = getattr(blog_formatter, 'boundary_map', {})
            logger.info(f"📍 AI generated {len(boundary_map)} boundaries: {list(boundary_map.keys())}")
            
            # Extract sections from the generated content
            sections = self._extract_sections_from_content(blog_content, boundary_map)
            logger.info(f"✅ Gemini generated {len(sections)} sections with AI content")
            
            # Debug: Check if sections have proper content vs transcript-like content
            for i, section in enumerate(sections):
                content_preview = section.get('content', '')[:150]
                logger.debug(f"Section {i+1} '{section['title']}': '{content_preview}...'")
            
            # Debug: Check if sections have proper paragraphs
            total_paragraphs = sum(len(s.get('paragraphs', [])) for s in sections)
            logger.info(f"📄 Total paragraphs across all sections: {total_paragraphs}")
            
            # Verify content quality
            has_blog_structure = any('##' in section.get('content', '') for section in sections)
            has_transcript_artifacts = any(
                any(word in section.get('content', '').lower() for word in ['right,', 'so,', 'now,', "let's"])
                for section in sections
            )
            
            if has_blog_structure and not has_transcript_artifacts:
                logger.info("🎉 SUCCESS: AI generated properly formatted blog content!")
            elif has_transcript_artifacts:
                logger.warning("⚠️  WARNING: Content still contains transcript-like language")
            else:
                logger.warning("⚠️  WARNING: Content may not have proper blog structure")
            
        except (SystemExit, ValueError) as e:
            logger.error(f"❌ AI processing failed ({type(e).__name__}: {e})")
            if "safety filter" in str(e).lower() or "All prompting strategies failed" in str(e):
                logger.error(f"🚨 GEMINI SAFETY FILTER: The video content triggered Gemini's safety filters")
                logger.error(f"💡 Multiple prompting strategies were attempted but all were blocked")
                logger.error(f"📝 This is usually due to technical content being misidentified as potentially harmful")
                logger.error(f"🔧 The system will create basic sections but content will be transcript-like")
                logger.error(f"⚠️  Consider: 1) Using a different video, 2) Different AI service, 3) Manual editing")
            logger.error(f"⚠️  This failure should NOT be affected by processing_mode: {processing_mode}")
            logger.error(f"🚨 IMPORTANT: User will see transcript-like content instead of blog content!")
            try:
                # Create better basic sections that are more blog-like
                sections = self._create_enhanced_basic_sections_from_transcript(transcript_segments)
                blog_content = self._format_enhanced_basic_content(sections, title)
                logger.warning(f"📄 Created {len(sections)} enhanced basic sections as fallback")
                logger.warning(f"⚠️  These sections contain improved transcript content, but not full AI formatting")
                
                # Add paragraphs to basic sections too
                for section in sections:
                    if 'paragraphs' not in section:
                        section['paragraphs'] = self._break_into_paragraphs(
                            section['content'], section['start_time'], section['end_time']
                        )
                        
            except Exception as fallback_error:
                logger.error(f"Fallback sectioning also failed: {fallback_error}")
                raise ValueError(f"Both AI and fallback sectioning failed: {fallback_error}")
        
        # Step 3: Extract candidate frames for each section
        processing_messages = {
            'smart': "🖼️ Extracting candidate frames with smart processing...",
            'dedupe': "🔄 Extracting frames every 0.5s and removing duplicates...",
            'raw': "⚡ Extracting raw frames every 0.5 seconds..."
        }
        logger.info(processing_messages[processing_mode])
        
        try:
            frames_data = self._extract_frames_by_section(video_path, sections, session_dir, processing_mode)
            total_frames = sum(len(fd['frames']) for fd in frames_data)
            logger.info(f"🖼️ Frame extraction completed: {total_frames} total frames across {len(frames_data)} sections")
        except Exception as frame_error:
            logger.error(f"Frame extraction failed: {frame_error}")
            # Create empty frames data to prevent session failure
            frames_data = []
            for i, section in enumerate(sections):
                frames_data.append({
                    'section_title': section['title'],
                    'section_index': i,
                    'start_time': section['start_time'],
                    'end_time': section['end_time'],
                    'frames': [],
                    'frame_count': 0
                })
            logger.warning("Created empty frames data due to extraction failure")
        
        # Step 4: Create session data
        session_data = {
            'session_id': session_id,
            'video_path': video_path,
            'title': title,
            'transcript_segments': transcript_segments,
            'sections': sections,
            'frames_data': frames_data,
            'blog_content_template': blog_content,
            'created_at': datetime.now().isoformat(),
            'ai_processing_success': len([s for s in sections if 'paragraphs' in s and len(s.get('content', '')) > 100]) > 0
        }
        
        # Save session
        session_file = os.path.join(session_dir, 'session_data.json')
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self.current_session = session_data
        
        # Final summary
        total_frames = sum(len(s['frames']) for s in frames_data)
        total_paragraphs = sum(len(s.get('paragraphs', [])) for s in sections)
        logger.info(f"✅ Session created: {len(sections)} sections, {total_paragraphs} paragraphs, {total_frames} candidate frames")
        logger.info(f"📊 Processing mode '{processing_mode}' completed successfully")
        
        return {
            'session_id': session_id,
            'sections': sections,
            'frames_data': frames_data,
            'transcript_segments': transcript_segments,  # Include for transcript toggle
            'title': title
        }
    
    def _extract_sections_from_content(self, blog_content: str, boundary_map: Dict) -> List[Dict]:
        """Extract section information from generated blog content, broken down by paragraphs."""
        sections = []
        
        # First, collect all section titles and their timestamps
        section_timestamps = []
        lines = blog_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('# ') or line.startswith('## '):
                title = line.lstrip('# ').strip()
                
                # Find timing from boundary map
                timestamp = 0
                found_timing = False
                for boundary_title, timing in boundary_map.items():
                    if title.lower() in boundary_title.lower() or boundary_title.lower() in title.lower():
                        if isinstance(timing, (int, float)):
                            timestamp = float(timing)
                        elif isinstance(timing, dict):
                            timestamp = timing.get('start_time', 0)
                        found_timing = True
                        logger.debug(f"📍 Matched section '{title}' to boundary '{boundary_title}' at {timestamp:.1f}s")
                        break
                
                if not found_timing:
                    logger.warning(f"⚠️  No timing found for section '{title}' in boundary map")
                    logger.warning(f"   Available boundaries: {list(boundary_map.keys())}")
                
                section_timestamps.append((title, timestamp))
        
        # Sort by timestamp to ensure proper ordering
        section_timestamps.sort(key=lambda x: x[1])
        
        # Fix timing issues: if multiple sections have timestamp 0, distribute them evenly
        zero_timestamp_sections = [i for i, (title, ts) in enumerate(section_timestamps) if ts == 0]
        if len(zero_timestamp_sections) > 1:
            logger.warning(f"⚠️  {len(zero_timestamp_sections)} sections have timestamp 0, redistributing...")
            # Estimate video duration or use a reasonable default
            video_duration = 600  # 10 minutes default, could be improved by getting actual duration
            
            # Find the last section with a real timestamp
            last_real_timestamp = 0
            for title, ts in section_timestamps:
                if ts > 0:
                    last_real_timestamp = max(last_real_timestamp, ts)
            
            # If we have real timestamps, use those to estimate remaining duration
            if last_real_timestamp > 0:
                remaining_duration = max(60, video_duration - last_real_timestamp)
            else:
                remaining_duration = video_duration
            
            # Redistribute zero-timestamp sections evenly
            section_duration = remaining_duration / len(zero_timestamp_sections)
            for i, zero_idx in enumerate(zero_timestamp_sections):
                if i == 0:
                    # Ensure first section always starts at 0
                    new_timestamp = 0.0
                    old_title, old_ts = section_timestamps[zero_idx]
                    section_timestamps[zero_idx] = (old_title, new_timestamp)
                    logger.info(f"📍 First section '{old_title}' kept at 0s")
                else:
                    # Distribute remaining sections starting from the appropriate offset
                    if last_real_timestamp > 0:
                        # If we have real timestamps, start after them
                        new_timestamp = last_real_timestamp + (i * section_duration)
                    else:
                        # If all sections were at 0, distribute evenly from the start
                        new_timestamp = i * section_duration
                    old_title, old_ts = section_timestamps[zero_idx]
                    section_timestamps[zero_idx] = (old_title, new_timestamp)
                    logger.info(f"📍 Redistributed '{old_title}': 0s -> {new_timestamp:.1f}s")
            
            # Re-sort after redistribution
            section_timestamps.sort(key=lambda x: x[1])
        
        # Now extract sections with paragraphs
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('# ') or line.startswith('## '):
                # New section header
                if current_section:
                    # Break current section into paragraphs before adding
                    current_section['paragraphs'] = self._break_into_paragraphs(
                        current_section['content'], 
                        current_section['start_time'], 
                        current_section['end_time']
                    )
                    sections.append(current_section)
                
                title = line.lstrip('# ').strip()
                
                # Find this section in our sorted list
                start_time = 0
                end_time = 0
                for i, (sec_title, sec_time) in enumerate(section_timestamps):
                    if sec_title == title:
                        start_time = sec_time
                        # End time is the start of the next section, or +60 seconds for the last
                        if i + 1 < len(section_timestamps):
                            end_time = section_timestamps[i + 1][1]
                        else:
                            end_time = start_time + 60
                        logger.debug(f"📍 Section '{title}': {start_time:.1f}s to {end_time:.1f}s")
                        break
                
                current_section = {
                    'title': title,
                    'start_time': start_time,
                    'end_time': end_time,
                    'content': ''
                }
            elif current_section and line:
                current_section['content'] += line + '\n'
        
        if current_section:
            # Break final section into paragraphs
            current_section['paragraphs'] = self._break_into_paragraphs(
                current_section['content'], 
                current_section['start_time'], 
                current_section['end_time']
            )
            sections.append(current_section)
        
        # Debug: log section and paragraph content
        for i, section in enumerate(sections):
            content_length = len(section.get('content', ''))
            paragraph_count = len(section.get('paragraphs', []))
            logger.debug(f"Section {i+1}: '{section['title']}' - {paragraph_count} paragraphs, {content_length} chars total")
            
            if not section.get('content', '').strip():
                logger.warning(f"Section '{section['title']}' has no content! Adding fallback...")
                section['content'] = f"This section covers content from {section['start_time']:.1f}s to {section['end_time']:.1f}s in the video."
                section['paragraphs'] = [{'content': section['content'], 'start_time': section['start_time'], 'end_time': section['end_time']}]
        
        return sections
    
    def _break_into_paragraphs(self, content: str, section_start: float, section_end: float) -> List[Dict]:
        """Break section content into paragraphs with estimated timing."""
        paragraphs = []
        
        # Split by double newlines to get paragraphs, then split large paragraphs further
        raw_paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not raw_paragraphs:
            return [{'content': content, 'start_time': section_start, 'end_time': section_end}]
        
        # Further split paragraphs that are too long (over ~600 characters or 4+ sentences)
        refined_paragraphs = []
        for paragraph in raw_paragraphs:
            if len(paragraph) > 600 or paragraph.count('.') > 4:
                # Split by sentences, grouping into larger chunks (2-3 sentences per paragraph)
                sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                
                current_chunk = []
                current_length = 0
                sentence_count = 0
                
                for sentence in sentences:
                    if (sentence_count >= 2 and current_length + len(sentence) > 400) and current_chunk:
                        # Start new chunk after 2+ sentences if getting long
                        refined_paragraphs.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                        sentence_count = 1
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                        sentence_count += 1
                
                if current_chunk:
                    refined_paragraphs.append(' '.join(current_chunk))
            else:
                refined_paragraphs.append(paragraph)
        
        # Distribute time evenly across refined paragraphs
        section_duration = section_end - section_start
        paragraph_duration = section_duration / len(refined_paragraphs)
        
        for i, paragraph_content in enumerate(refined_paragraphs):
            paragraph_start = section_start + (i * paragraph_duration)
            paragraph_end = section_start + ((i + 1) * paragraph_duration)
            
            paragraphs.append({
                'content': paragraph_content,
                'start_time': paragraph_start,
                'end_time': paragraph_end
            })
        
        return paragraphs
    
    def _create_basic_sections_from_transcript(self, transcript_segments: List[Dict]) -> List[Dict]:
        """Create basic sections when AI processing fails."""
        sections = []
        
        if transcript_segments:
            # Debug: log the structure of transcript segments
            logger.debug(f"Transcript segment keys: {list(transcript_segments[0].keys())}")
            logger.debug(f"Total transcript segments: {len(transcript_segments)}")
            
            # Safely get total duration
            last_segment = transcript_segments[-1]
            total_duration = last_segment.get('end_time', last_segment.get('end', 300))
        else:
            total_duration = 300
            
        section_duration = total_duration / 5  # Create 5 sections
        
        for i in range(5):
            start_time = i * section_duration
            end_time = min((i + 1) * section_duration, total_duration)
            
            # Get actual transcript content for this time range (with overlap tolerance)
            section_content = []
            for segment in transcript_segments:
                # Include segments that overlap with this section's time range
                if (segment['start_time'] < end_time and segment['end_time'] > start_time):
                    section_content.append(segment['text'])
            
            content = '\n\n'.join(section_content) if section_content else f"No transcript content found for this time range ({start_time:.1f}s - {end_time:.1f}s)"
            
            sections.append({
                'title': f"Section {i + 1}",
                'start_time': start_time,
                'end_time': end_time,
                'content': content
            })
        
        return sections
    
    def _format_basic_content(self, sections: List[Dict], title: str) -> str:
        """Format basic blog content template."""
        content = f"# {title}\n\n"
        
        for section in sections:
            content += f"## {section['title']}\n\n"
            content += f"{section['content']}\n\n"
            content += "{{< image-placeholder >}}\n\n"
        
        return content
    
    def _create_enhanced_basic_sections_from_transcript(self, transcript_segments: List[Dict]) -> List[Dict]:
        """Create more blog-like sections from transcript when AI processing fails."""
        if not transcript_segments:
            return []
        
        # Group transcript segments into logical sections (longer than basic)
        sections = []
        current_section_segments = []
        current_section_duration = 0
        target_section_duration = 90  # 1.5 minutes per section for better content
        
        for segment in transcript_segments:
            segment_duration = segment['end_time'] - segment['start_time']
            
            # Add segment to current section
            current_section_segments.append(segment)
            current_section_duration += segment_duration
            
            # Check if we should finalize this section
            if current_section_duration >= target_section_duration and len(current_section_segments) >= 3:
                sections.append(self._create_enhanced_section_from_segments(current_section_segments, len(sections) + 1))
                current_section_segments = []
                current_section_duration = 0
        
        # Handle remaining segments
        if current_section_segments:
            sections.append(self._create_enhanced_section_from_segments(current_section_segments, len(sections) + 1))
        
        return sections
    
    def _create_enhanced_section_from_segments(self, segments: List[Dict], section_number: int) -> Dict:
        """Create an enhanced section from transcript segments with better formatting."""
        if not segments:
            return {}
        
        start_time = segments[0]['start_time']
        end_time = segments[-1]['end_time']
        
        # Combine and clean transcript text
        combined_text = ' '.join([seg['text'] for seg in segments])
        
        # Basic cleaning and formatting
        lines = combined_text.split('.')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Skip very short fragments
                # Capitalize first letter and add period if needed
                line = line[0].upper() + line[1:] if line else line
                if not line.endswith('.'):
                    line += '.'
                cleaned_lines.append(line)
        
        # Group sentences into paragraphs (3-4 sentences each)
        paragraphs = []
        current_paragraph = []
        
        for line in cleaned_lines:
            current_paragraph.append(line)
            if len(current_paragraph) >= 3:  # Create paragraph every 3 sentences
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Create content with paragraph breaks
        content = '\n\n'.join(paragraphs)
        
        # Generate a meaningful title based on content keywords
        title = self._generate_section_title(content, section_number)
        
        return {
            'title': title,
            'start_time': start_time,
            'end_time': end_time,
            'content': content,
            'paragraphs': self._split_into_paragraphs(content, start_time, end_time)
        }
    
    def _generate_section_title(self, content: str, section_number: int) -> str:
        """Generate a meaningful title from section content."""
        # Simple keyword extraction for title generation
        words = content.lower().split()
        
        # Common meaningful words that could indicate topic
        topic_indicators = [
            'setup', 'configuration', 'install', 'create', 'build', 'deploy', 
            'tutorial', 'guide', 'example', 'demo', 'introduction', 'overview',
            'implementation', 'development', 'testing', 'debugging', 'optimization',
            'security', 'performance', 'database', 'server', 'client', 'api',
            'frontend', 'backend', 'framework', 'library', 'application'
        ]
        
        # Look for topic indicators in content
        found_topics = [word for word in words if word in topic_indicators]
        
        if found_topics:
            # Use the first meaningful topic found
            topic = found_topics[0].capitalize()
            return f"{topic} Overview"
        else:
            # Fallback to generic section naming
            return f"Section {section_number}"
    
    def _format_enhanced_basic_content(self, sections: List[Dict], title: str) -> str:
        """Format enhanced basic content with better structure than raw transcript."""
        content = f"# {title}\n\n"
        content += "_This content was automatically generated from video transcript. Some AI processing features may have been limited._\n\n"
        
        for section in sections:
            content += f"## {section['title']}\n\n"
            content += f"{section['content']}\n\n"
            content += "{{< image-placeholder >}}\n\n"
        
        return content
    
    def _extract_frames_by_section(self, video_path: str, sections: List[Dict], session_dir: str, processing_mode: str = 'smart') -> List[Dict]:
        """Extract candidate frames for each section."""
        frames_dir = os.path.join(session_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        temp_frames_dir = "temp_frames"
        
        # Clear and extract all frames
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        if processing_mode == 'smart':
            # Use existing video processor with smart analysis
            video_processor = VideoProcessor(self.config)
            transcript_extractor = TranscriptExtractor(self.config)
            transcript_segments = transcript_extractor.extract_transcript(video_path)
            smart_frames = video_processor.extract_frames(video_path, temp_frames_dir, transcript_segments)
            
            # Convert dict format to namedtuple format for compatibility
            from collections import namedtuple
            Frame = namedtuple('Frame', ['filename', 'timestamp', 'path'])
            candidate_frames = []
            
            for frame_dict in smart_frames:
                # Extract filename from path if not provided
                filename = frame_dict.get('filename') or os.path.basename(frame_dict['path'])
                candidate_frames.append(Frame(
                    filename=filename,
                    timestamp=frame_dict['timestamp'],
                    path=frame_dict['path']
                ))
        elif processing_mode == 'dedupe':
            # Extract every 0.5s, then remove duplicates
            candidate_frames = self._extract_raw_frames(video_path, temp_frames_dir)
            candidate_frames = self._remove_duplicate_frames(candidate_frames, temp_frames_dir)
        else:  # raw mode
            # Fast extraction: just grab frames every 0.5 seconds
            candidate_frames = self._extract_raw_frames(video_path, temp_frames_dir)
        
        # Group frames by section
        frames_data = []
        
        # Debug: Log frame and section timing information
        if candidate_frames:
            logger.info(f"🖼️  Frame distribution debug:")
            logger.info(f"   Total candidate frames: {len(candidate_frames)}")
            frame_times = [f.timestamp for f in candidate_frames[:10]]  # First 10 frames
            logger.info(f"   First 10 frame timestamps: {frame_times}")
            
            logger.info(f"   Section timing:")
            for i, section in enumerate(sections):
                logger.info(f"   Section {i+1}: '{section['title']}' - {section['start_time']:.1f}s to {section['end_time']:.1f}s")
        
        for section_idx, section in enumerate(sections):
            section_frames = []
            matched_frames = 0
            
            for frame in candidate_frames:
                if section['start_time'] <= frame.timestamp <= section['end_time']:
                    matched_frames += 1
                    # Copy frame to session directory
                    src_path = os.path.join(temp_frames_dir, frame.filename)
                    dst_path = os.path.join(frames_dir, frame.filename)
                    
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                        
                        section_frames.append({
                            'filename': frame.filename,
                            'timestamp': frame.timestamp,
                            'path': dst_path
                        })
            
            logger.info(f"🎯 Section {section_idx+1} '{section['title']}': {len(section_frames)} frames ({section['start_time']:.1f}s-{section['end_time']:.1f}s)")
            
            if len(section_frames) == 0 and section_idx > 0:
                logger.warning(f"⚠️  No frames found for section {section_idx+1}! This suggests a timing mismatch.")
                logger.warning(f"   Section range: {section['start_time']:.1f}s to {section['end_time']:.1f}s")
                if candidate_frames:
                    nearby_frames = [f for f in candidate_frames if abs(f.timestamp - section['start_time']) < 30]
                    if nearby_frames:
                        logger.warning(f"   Nearby frames: {[f'{f.timestamp:.1f}s' for f in nearby_frames[:5]]}")
            
            frames_data.append({
                'section_title': section['title'],
                'section_index': len(frames_data),
                'start_time': section['start_time'],
                'end_time': section['end_time'],
                'frames': section_frames,
                'frame_count': len(section_frames)
            })
        
        return frames_data
    
    def _extract_raw_frames(self, video_path: str, output_dir: str) -> List:
        """Extract raw frames every 0.5 seconds using ffmpeg."""
        import subprocess
        import cv2
        from collections import namedtuple
        
        # Create a simple frame object structure
        Frame = namedtuple('Frame', ['filename', 'timestamp', 'path'])
        
        try:
            # Get video duration first
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'csv=p=0', 
                '-select_streams', 'v:0', '-show_entries', 'format=duration', 
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            
            # Extract frames every 1 second using manual timestamp method for better reliability
            # This approach extracts frames at exact timestamps rather than relying on fps filter
            frame_times = []
            current_time = 0.0
            while current_time <= duration:
                frame_times.append(current_time)
                current_time += 1.0
            
            logger.info(f"📊 Will extract {len(frame_times)} frames from 0s to {frame_times[-1]:.1f}s (video duration: {duration:.1f}s)")
            logger.info(f"⚡ Extracting frames at specific timestamps with ffmpeg...")
            
            # Extract frames at specific timestamps (more reliable than fps filter)
            frames = []
            for i, timestamp in enumerate(frame_times):
                filename = f"frame_{i:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Extract frame at specific timestamp
                cmd = [
                    'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                    '-vframes', '1',  # Extract exactly 1 frame
                    '-q:v', '2',      # High quality
                    '-an',            # No audio
                    filepath
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    # Verify the frame was actually created
                    if os.path.exists(filepath):
                        frames.append(Frame(
                            filename=filename,
                            timestamp=timestamp,
                            path=filepath
                        ))
                        if i % 10 == 0:  # Log progress every 10 frames (since we have fewer frames now)
                            logger.info(f"  📸 Extracted frame {i+1}/{len(frame_times)} at {timestamp:.1f}s")
                    else:
                        logger.warning(f"⚠️  Frame not created at {timestamp:.1f}s")
                        
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️  Failed to extract frame at {timestamp:.1f}s: {e}")
                    
            logger.info(f"✅ Successfully extracted {len(frames)} frames using timestamp method")
            return frames
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e}")
            # Fallback to cv2 extraction
            return self._extract_raw_frames_cv2(video_path, output_dir)
        except Exception as e:
            logger.error(f"Raw frame extraction failed: {e}")
            return []
    
    def _extract_raw_frames_cv2(self, video_path: str, output_dir: str) -> List:
        """Fallback: extract frames using OpenCV."""
        import cv2
        from collections import namedtuple
        
        Frame = namedtuple('Frame', ['filename', 'timestamp', 'path'])
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📹 CV2: Video has {total_frames} frames at {fps:.2f} fps ({duration:.2f}s)")
            
            frame_interval = int(fps * 1.0)  # Every 1 second
            if frame_interval == 0:
                frame_interval = 1  # Minimum interval
            
            frame_count = 0
            extracted_count = 0
            
            logger.info(f"📊 CV2: Extracting every {frame_interval} frames (every {frame_interval/fps:.1f}s - targeting 1s intervals)")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    filename = f"frame_{extracted_count:05d}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(filepath, frame)
                    frames.append(Frame(
                        filename=filename,
                        timestamp=timestamp,
                        path=filepath
                    ))
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:  # Progress every 10 frames (since we have fewer frames now)
                        logger.info(f"  📸 CV2: Extracted {extracted_count} frames, current time: {timestamp:.1f}s")
                
                frame_count += 1
            
            cap.release()
            
            if frames:
                last_timestamp = frames[-1].timestamp
                coverage = (last_timestamp / duration) * 100 if duration > 0 else 0
                logger.info(f"✅ CV2 extracted {len(frames)} frames, coverage: {last_timestamp:.1f}s/{duration:.1f}s ({coverage:.1f}%)")
            else:
                logger.warning("⚠️  CV2 extracted no frames!")
                
            return frames
            
        except Exception as e:
            logger.error(f"CV2 frame extraction failed: {e}")
            return []
    
    def _remove_duplicate_frames(self, frames: List, frames_dir: str) -> List:
        """Remove duplicate frames using improved sliding window approach with caching."""
        import cv2
        import numpy as np
        
        if not frames:
            return frames
        
        logger.info(f"🔄 Removing duplicates from {len(frames)} frames...")
        
        # Debug: Show frame distribution before processing
        if frames:
            timestamps = [f.timestamp for f in frames]
            logger.info(f"📊 Frame timestamp range: {min(timestamps):.1f}s to {max(timestamps):.1f}s")
            logger.info(f"📊 Sample timestamps: {timestamps[:10]} ... {timestamps[-10:] if len(timestamps) > 10 else []}")
        
        unique_frames = []
        # Much less aggressive settings - allow more variation
        similarity_threshold = 0.88  # 88% similar = duplicate (was 0.92, reducing further for less aggressive filtering)
        window_size = 5  # Compare against fewer frames for speed
        min_time_gap = 2.0  # Minimum 2 seconds between frames for better distribution 
        frames_processed = 0
        frames_skipped_due_to_errors = 0
        image_cache = {}  # Cache processed images
        
        for i, frame in enumerate(frames):
            is_duplicate = False
            frames_processed += 1
            
            # Progress logging every 200 frames
            if frames_processed % 200 == 0:
                logger.info(f"🔄 Processing frame {frames_processed}/{len(frames)} (timestamp: {frame.timestamp:.1f}s)")
            
            try:
                # Use cached image if available
                if frame.path not in image_cache:
                    current_img = cv2.imread(frame.path)
                    if current_img is None:
                        logger.debug(f"Could not read frame file: {frame.path}")
                        frames_skipped_due_to_errors += 1
                        continue
                    
                    # Convert to grayscale and resize for faster comparison
                    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                    current_small = cv2.resize(current_gray, (32, 32))  # Slightly larger for better accuracy
                    image_cache[frame.path] = current_small
                else:
                    current_small = image_cache[frame.path]
                
                # Enforce minimum time gap between frames (simple temporal filtering)
                if unique_frames:
                    last_frame_time = unique_frames[-1].timestamp
                    if frame.timestamp - last_frame_time < min_time_gap:
                        logger.debug(f"Skipping frame {frame.filename} at {frame.timestamp:.1f}s (too close to {last_frame_time:.1f}s)")
                        is_duplicate = True  # Mark as duplicate but continue processing
                        continue
                
                # Compare with recent unique frames (sliding window)
                comparison_frames = unique_frames[-window_size:] if len(unique_frames) > window_size else unique_frames
                
                for unique_frame in comparison_frames:
                    try:
                        # Use cached image if available
                        if unique_frame.path not in image_cache:
                            unique_img = cv2.imread(unique_frame.path)
                            if unique_img is None:
                                continue
                            unique_gray = cv2.cvtColor(unique_img, cv2.COLOR_BGR2GRAY)
                            unique_small = cv2.resize(unique_gray, (32, 32))
                            image_cache[unique_frame.path] = unique_small
                        else:
                            unique_small = image_cache[unique_frame.path]
                        
                        # Calculate normalized cross correlation for similarity detection
                        correlation = cv2.matchTemplate(current_small, unique_small, cv2.TM_CCOEFF_NORMED)[0][0]
                        
                        if correlation > similarity_threshold:
                            is_duplicate = True
                            logger.debug(f"Frame {frame.filename} ({frame.timestamp:.1f}s) is duplicate of {unique_frame.filename} ({unique_frame.timestamp:.1f}s) (correlation: {correlation:.3f})")
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error comparing frames: {e}")
                        continue
                
                if not is_duplicate:
                    unique_frames.append(frame)
                    logger.debug(f"✅ Keeping frame {frame.filename} at {frame.timestamp:.1f}s (unique #{len(unique_frames)})")
                    
            except Exception as e:
                logger.warning(f"Error processing frame {frame.filename} at {frame.timestamp:.1f}s: {e}")
                frames_skipped_due_to_errors += 1
                # Include frame if we can't process it
                unique_frames.append(frame)
        
        logger.info(f"✅ Kept {len(unique_frames)} unique frames (removed {len(frames) - len(unique_frames)} duplicates)")
        logger.info(f"📊 Processing summary: {frames_processed} frames processed, {frames_skipped_due_to_errors} skipped due to errors")
        
        # Debug: Show final frame distribution
        if unique_frames:
            final_timestamps = [f.timestamp for f in unique_frames]
            logger.info(f"📊 Final frame timestamp range: {min(final_timestamps):.1f}s to {max(final_timestamps):.1f}s")
            if len(final_timestamps) <= 20:
                logger.info(f"📊 All final timestamps: {final_timestamps}")
            else:
                logger.info(f"📊 Final timestamps: {final_timestamps[:10]} ... {final_timestamps[-10:]}")
        
        # Safety check: ensure we have frames covering the full video duration
        if unique_frames and len(frames) > 0:
            original_duration = max(f.timestamp for f in frames)
            final_duration = max(f.timestamp for f in unique_frames)
            coverage_ratio = final_duration / original_duration if original_duration > 0 else 0
            
            logger.info(f"📊 Video coverage: {final_duration:.1f}s / {original_duration:.1f}s ({coverage_ratio:.1%})")
            
            # If we're only covering less than 50% of the video, fall back to sampling
            if coverage_ratio < 0.5 and len(unique_frames) < 50:
                logger.warning(f"⚠️ Poor video coverage ({coverage_ratio:.1%}), adding sampling fallback...")
                unique_frames = self._add_sampling_fallback(frames, unique_frames, target_count=100)
        
        return unique_frames
    
    def _add_sampling_fallback(self, original_frames: List, current_frames: List, target_count: int = 100) -> List:
        """Add evenly distributed frames as fallback when deduplication is too aggressive."""
        if not original_frames:
            return current_frames
        
        # Calculate sampling interval to get target number of frames
        total_duration = max(f.timestamp for f in original_frames)
        sampling_interval = total_duration / target_count
        
        sampled_frames = []
        existing_timestamps = {f.timestamp for f in current_frames}
        
        # Add evenly spaced frames that don't already exist
        for i in range(target_count):
            target_timestamp = i * sampling_interval
            
            # Find the closest frame to this timestamp
            closest_frame = min(original_frames, key=lambda f: abs(f.timestamp - target_timestamp))
            
            # Only add if we don't already have this timestamp (within 1 second tolerance)
            if not any(abs(closest_frame.timestamp - ts) < 1.0 for ts in existing_timestamps):
                sampled_frames.append(closest_frame)
                existing_timestamps.add(closest_frame.timestamp)
        
        # Combine and sort by timestamp
        combined_frames = current_frames + sampled_frames
        combined_frames.sort(key=lambda f: f.timestamp)
        
        logger.info(f"📊 Sampling fallback: added {len(sampled_frames)} frames, total now {len(combined_frames)}")
        return combined_frames
    
    def save_selections(self, selections_data: Dict):
        """Save frame selections for each section."""
        if not self.current_session:
            raise ValueError("No active session")
        
        session_dir = os.path.join(self.sessions_dir, self.current_session['session_id'])
        selections_file = os.path.join(session_dir, 'frame_selections.json')
        
        with open(selections_file, 'w') as f:
            json.dump(selections_data, f, indent=2)
        
        logger.info(f"💾 Saved frame selections: {len(selections_data.get('sections', []))} sections")
    
    def generate_final_blog(self) -> Dict:
        """Generate the final blog post with selected frames."""
        if not self.current_session:
            raise ValueError("No active session")
        
        session_dir = os.path.join(self.sessions_dir, self.current_session['session_id']) 
        selections_file = os.path.join(session_dir, 'frame_selections.json')
        
        if not os.path.exists(selections_file):
            raise ValueError("No frame selections found")
        
        # Load selections
        with open(selections_file, 'r') as f:
            selections = json.load(f)
        
        logger.info("🏗️ Generating final blog post with selected frames...")
        
        # Prepare frame data for Hugo generator with paragraph grouping
        selected_frames = []
        for section_data in selections['sections']:
            frames_with_paragraphs = section_data.get('frames_with_paragraphs', [])
            
            if frames_with_paragraphs:
                # Use the new paragraph-aware data structure
                for frame_info in frames_with_paragraphs:
                    frame_filename = frame_info['filename']
                    src_path = os.path.join(session_dir, 'frames', frame_filename)
                    
                    if os.path.exists(src_path):
                        selected_frames.append({
                            'filename': frame_filename,
                            'path': src_path,
                            'section': section_data['section_title'],
                            'section_title': section_data['section_title'],
                            'timestamp': frame_info['timestamp'],
                            'paragraph_index': frame_info['paragraph_index'],
                            'paragraph_start_time': frame_info['paragraph_start_time'],
                            'paragraph_end_time': frame_info['paragraph_end_time'],
                            'should_include': True  # Mark as selected for processing
                        })
            else:
                # Fallback to old method for backward compatibility
                for frame_path in section_data['selected_frames']:
                    frame_filename = os.path.basename(frame_path)
                    src_path = os.path.join(session_dir, 'frames', frame_filename)
                    
                    if os.path.exists(src_path):
                        # Extract timestamp from filename (assumes format like "frame_123.5s.jpg")
                        timestamp = 0.0
                        try:
                            # Try to extract timestamp from filename
                            import re
                            match = re.search(r'(\d+\.?\d*)s?\.jpg', frame_filename)
                            if match:
                                timestamp = float(match.group(1))
                            else:
                                # Fallback: try to get from frame index (frame_00123.jpg -> 123 * 0.5)
                                match = re.search(r'frame_(\d+)\.jpg', frame_filename)
                                if match:
                                    frame_index = int(match.group(1))
                                    timestamp = frame_index * 0.5
                        except:
                            timestamp = 0.0
                        
                        selected_frames.append({
                            'filename': frame_filename,
                            'path': src_path,
                            'section': section_data['section_title'],
                            'section_title': section_data['section_title'],
                            'timestamp': timestamp,
                            'should_include': True  # Mark as selected for processing
                        })
        
        # Generate blog using Hugo generator
        hugo_generator = HugoGenerator(self.config)
        
        try:
            # Use original blog content template - let Hugo generator handle frame insertion
            blog_content = self.current_session['blog_content_template']
            
            # Generate final blog post using pre-formatted content (no Gemini needed!)
            # Create output path
            video_name = Path(self.current_session['video_path']).stem
            output_dir = os.path.join(
                self.config['output']['base_folder'], 
                self.config['output']['posts_folder'],
                video_name
            )
            
            # Create video info dict
            video_info = {
                'path': self.current_session['video_path'],
                'duration': 0,  # Not needed for pre-formatted content
                'title': self.current_session['title']
            }
            
            # Get template path from config - try both nested and flattened formats
            template_path = (self.flattened_config.get('template_path') or 
                           self.config.get('template', {}).get('path'))
            
            result_path = hugo_generator.generate_blog_post_with_formatted_content(
                self.current_session['title'],
                blog_content,
                selected_frames,
                video_info,
                output_dir,
                front_matter_data=None,
                template_path=template_path,
                transcript_segments=self.current_session['transcript_segments']
            )
            
            result = {'output_file': result_path}
            
            logger.info(f"✅ Blog generated successfully: {result['output_file']}")
            
            return {
                'success': True,
                'output_file': result['output_file'],
                'frames_used': len(selected_frames),
                'blog_content': blog_content
            }
            
        except Exception as e:
            logger.error(f"Failed to generate blog: {e}")
            raise
    
    
    def run(self, host='127.0.0.1', port=5001, debug=True):
        """Start the web interface."""
        logger.info(f"🚀 Starting Hybrid Blog Creator at http://{host}:{port}")
        logger.info("💡 This tool combines AI content processing with manual frame selection")
        logger.info("📝 Workflow: Process video → Select frames → Generate blog")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    creator = HybridBlogCreator()
    creator.run()