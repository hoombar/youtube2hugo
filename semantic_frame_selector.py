"""
Semantic-driven frame selection using Gemini for content analysis.

This module analyzes transcript content to identify logical sections and visual cues,
then selects frames based on semantic relevance rather than just temporal distribution.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional OCR support
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract not available - OCR-based text matching will be disabled")

class SemanticSectionAnalyzer:
    """Analyzes transcript content to identify semantic sections with visual cues."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gemini_client = None
        
        # Initialize Gemini client
        api_key = config.get('gemini_api_key') or config.get('gemini', {}).get('api_key')
        if not api_key:
            raise ValueError("Gemini API key required for semantic analysis")
            
        genai.configure(api_key=api_key)
        model_name = config.get('gemini_model', 'gemini-2.5-flash')
        self.gemini_client = genai.GenerativeModel(model_name)
        logger.info(f"Semantic analyzer initialized with model: {model_name}")
    
    def analyze_transcript_sections(self, transcript_segments: List[Dict]) -> List[Dict]:
        """Analyze transcript to identify semantic sections with visual cues."""
        
        # Convert transcript segments to text with timestamps
        transcript_text = self._format_transcript_for_analysis(transcript_segments)
        
        # Create comprehensive Gemini prompt
        prompt = self._create_section_analysis_prompt(transcript_text)
        
        # Get Gemini analysis
        logger.info("🧠 Analyzing transcript with Gemini for semantic sections...")
        response = self._query_gemini_with_retry(prompt)
        
        # Parse response into structured sections
        sections = self._parse_gemini_sections_response(response)
        
        logger.info(f"📋 Identified {len(sections)} semantic sections")
        for i, section in enumerate(sections):
            logger.info(f"   Section {i+1}: {section['start_time']:.1f}s-{section['end_time']:.1f}s - {section['title']}")
        
        return sections
    
    def _format_transcript_for_analysis(self, transcript_segments: List[Dict]) -> str:
        """Format transcript segments into timestamped text for Gemini analysis."""
        formatted_lines = []
        
        for segment in transcript_segments:
            timestamp = segment['start_time']
            text = segment['text'].strip()
            if text:
                formatted_lines.append(f"[{timestamp:.1f}s] {text}")
        
        return '\n'.join(formatted_lines)
    
    def _create_section_analysis_prompt(self, transcript_text: str) -> str:
        """Create comprehensive prompt for Gemini section analysis."""
        
        prompt = f"""
Analyze this video transcript and identify logical content sections for optimal frame selection in a technical blog post.

TRANSCRIPT:
{transcript_text}

ANALYSIS REQUIREMENTS:

1. **Identify 4-8 logical content sections** based on topic changes, demonstrations, or conceptual shifts
2. **Determine precise timestamps** for each section start/end 
3. **Extract visual cues** - specific UI elements, screens, devices, or visual content mentioned
4. **Assess importance** for blog illustration purposes

RESPONSE FORMAT (valid JSON only):
```json
{{
  "sections": [
    {{
      "title": "Descriptive section title",
      "start_time": 10.5,
      "end_time": 45.2,
      "importance": "high|medium|low",
      "main_topic": "Brief description of what's being demonstrated",
      "visual_cues": [
        "settings screen",
        "configuration panel", 
        "device list",
        "automation interface"
      ],
      "mentioned_text": [
        "Home Assistant",
        "Configuration",
        "Devices & Services"
      ],
      "ui_elements": [
        "buttons",
        "dropdown menus",
        "input fields",
        "navigation bars"
      ],
      "key_moments": [
        {{"time": 15.3, "action": "opens settings menu"}},
        {{"time": 28.7, "action": "configures device"}}
      ]
    }}
  ]
}}
```

VISUAL CUE CATEGORIES:
- **Screens**: "settings screen", "dashboard", "configuration page", "mobile app"
- **UI Elements**: "buttons", "menus", "forms", "navigation", "dialogs"
- **Devices**: "smart speakers", "sensors", "hubs", "mobile device", "computer screen"
- **Home Assistant**: "Lovelace dashboard", "configuration yaml", "integrations page", "automation editor"
- **Code/Config**: "yaml files", "configuration code", "automation scripts", "entity cards"
- **Actions**: "clicking", "typing", "selecting", "scrolling", "navigating"

IMPORTANCE LEVELS:
- **high**: Core functionality, main demonstrations, key setup steps
- **medium**: Supporting content, secondary features, explanations  
- **low**: Transitions, introductions, tangential content

Focus on sections where specific visual elements are mentioned or demonstrated. Prioritize technical content that would benefit from visual illustration in a blog post.

Return ONLY the JSON response, no additional text.
"""
        return prompt
    
    def _query_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Query Gemini with retry logic for rate limiting."""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Querying Gemini (attempt {attempt + 1}/{max_retries})...")
                response = self.gemini_client.generate_content(prompt)
                
                if hasattr(response, 'text') and response.text:
                    logger.info(f"✅ Gemini response received ({len(response.text)} characters)")
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini")
                
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("❌ All Gemini retry attempts failed - using fallback sections")
                    raise
                
                # Exponential backoff with maximum wait time
                import time
                wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    def _parse_gemini_sections_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's JSON response into structured sections."""
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            if json_text.startswith('```json'):
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif json_text.startswith('```'):
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            parsed = json.loads(json_text)
            
            if 'sections' not in parsed:
                raise ValueError("Response missing 'sections' key")
            
            sections = parsed['sections']
            
            # Validate and clean sections
            validated_sections = []
            for section in sections:
                if self._validate_section(section):
                    validated_sections.append(section)
                else:
                    logger.warning(f"Skipping invalid section: {section.get('title', 'Unknown')}")
            
            return validated_sections
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            
            # Fallback: create default sections
            return self._create_fallback_sections()
    
    def _validate_section(self, section: Dict) -> bool:
        """Validate that a section has required fields."""
        required_fields = ['title', 'start_time', 'end_time', 'visual_cues']
        
        for field in required_fields:
            if field not in section:
                return False
        
        # Validate timestamp logic
        if section['start_time'] >= section['end_time']:
            return False
            
        return True
    
    def _create_fallback_sections(self) -> List[Dict]:
        """Create fallback sections if Gemini analysis fails."""
        logger.warning("Creating fallback sections due to Gemini parsing failure")
        
        return [
            {
                "title": "Introduction and Setup",
                "start_time": 0,
                "end_time": 120,
                "importance": "high",
                "visual_cues": ["settings screen", "configuration panel", "initial setup"],
                "mentioned_text": ["setup", "configuration", "getting started"],
                "ui_elements": ["buttons", "menus", "forms"]
            },
            {
                "title": "Main Demonstration",
                "start_time": 120,
                "end_time": 300,
                "importance": "high", 
                "visual_cues": ["dashboard", "main interface", "device controls"],
                "mentioned_text": ["dashboard", "devices", "automation"],
                "ui_elements": ["cards", "switches", "controls"]
            },
            {
                "title": "Advanced Configuration",
                "start_time": 300,
                "end_time": 500,
                "importance": "medium",
                "visual_cues": ["configuration files", "advanced settings", "yaml editor"],
                "mentioned_text": ["configuration", "yaml", "advanced"],
                "ui_elements": ["text editors", "code blocks", "file browser"]
            }
        ]


class SemanticFrameScorer:
    """Scores frames based on semantic relevance to section content."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.video_title = ""  # Will be set by select_frames_semantically
        
        # Configurable scoring weights
        semantic_config = config.get('semantic_frame_selection', {})
        self.base_score_weight = semantic_config.get('base_score_weight', 0.3)
        self.text_score_weight = semantic_config.get('text_score_weight', 0.4)
        self.visual_score_weight = semantic_config.get('visual_score_weight', 0.3)
        self.score_threshold = semantic_config.get('score_threshold', 50.0)
    
    def score_frame_for_section(self, frame_path: str, section: Dict) -> float:
        """Score a frame's relevance to a semantic section."""
        
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return 0.0
            
            total_score = 0.0
            
            # Base visual quality score (from existing algorithm)
            base_score = self._get_base_visual_score(frame)
            total_score += base_score * self.base_score_weight
            
            # OCR-based text matching
            text_score = self._score_text_relevance(frame, section)
            total_score += text_score * self.text_score_weight
            
            # Visual element matching
            visual_score = self._score_visual_elements(frame, section)
            total_score += visual_score * self.visual_score_weight
            
            # Importance multiplier
            importance_multiplier = {'high': 1.2, 'medium': 1.0, 'low': 0.8}
            multiplier = importance_multiplier.get(section.get('importance', 'medium'), 1.0)
            total_score *= multiplier
            
            logger.debug(f"Frame {frame_path}: base={base_score:.1f}, text={text_score:.1f}, visual={visual_score:.1f}, total={total_score:.1f}")
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error scoring frame {frame_path}: {e}")
            return 0.0
    
    def _get_base_visual_score(self, frame: np.ndarray) -> float:
        """Get base visual quality score using existing algorithm metrics."""
        
        # This would integrate with the existing scoring system
        # For now, implement basic quality metrics
        
        height, width = frame.shape[:2]
        
        # Edge density
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Contrast
        contrast = np.std(gray)
        
        # Brightness
        brightness = np.mean(gray)
        
        # Simple scoring
        score = 0
        if edge_density > 0.05:
            score += 30
        if 50 < brightness < 200:
            score += 20
        if contrast > 30:
            score += 25
        
        # Training-derived brightness preference: bonus for bright frames
        if brightness > 160:
            score += 10
            
        return min(score, 100)
    
    def _score_text_relevance(self, frame: np.ndarray, section: Dict) -> float:
        """Score frame based on OCR text matching section's mentioned text."""
        
        if not OCR_AVAILABLE:
            logger.debug("OCR not available - skipping text relevance scoring")
            return 0.0
        
        try:
            # Extract text using OCR
            text = pytesseract.image_to_string(frame, config='--psm 6')
            text_lower = text.lower()
            
            # Title sequence detection - heavily penalize frames with title/intro indicators
            generic_title_indicators = [
                'subscribe', 'like and subscribe', 'channel',
                'intro', 'introduction', 'welcome to', 'today we', 'today i',
                'logo', 'brand', 'episode', 'part 1', 'part 2', 
                'tutorial series', 'coming up'
            ]
            
            # Check for generic title sequence indicators
            for indicator in generic_title_indicators:
                if indicator in text_lower:
                    logger.debug(f"Title sequence detected: '{indicator}' found in frame text")
                    return -50.0  # Heavy penalty for title sequence frames
            
            # Dynamic title sequence detection based on video title
            if self.video_title:
                title_words = self.video_title.lower().split()
                # Check if significant portion of video title appears in frame text
                title_matches = sum(1 for word in title_words if len(word) > 3 and word in text_lower)
                if title_matches >= len(title_words) * 0.6:  # 60% of title words found
                    logger.debug(f"Dynamic title sequence detected: {title_matches}/{len(title_words)} title words found")
                    return -50.0  # Heavy penalty for title sequence frames
            
            score = 0.0
            mentioned_texts = section.get('mentioned_text', [])
            
            for mentioned in mentioned_texts:
                if mentioned.lower() in text_lower:
                    score += 50  # High bonus for exact text matches
                    logger.debug(f"Text match found: '{mentioned}' in frame")
            
            # Partial matches
            for mentioned in mentioned_texts:
                words = mentioned.lower().split()
                matches = sum(1 for word in words if word in text_lower)
                if matches > 0:
                    score += (matches / len(words)) * 25
            
            return min(score, 100)
            
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return 0.0
    
    def _score_visual_elements(self, frame: np.ndarray, section: Dict) -> float:
        """Score frame based on visual elements mentioned in section."""
        
        visual_cues = section.get('visual_cues', [])
        ui_elements = section.get('ui_elements', [])
        
        score = 0.0
        
        # Color scheme detection for Home Assistant
        if any('home assistant' in cue.lower() for cue in visual_cues):
            if self._detect_ha_colors(frame):
                score += 40
        
        # UI element detection
        if any(elem in ['buttons', 'menus', 'forms'] for elem in ui_elements):
            if self._detect_ui_elements(frame):
                score += 30
        
        # Screen/interface detection  
        if any('screen' in cue or 'interface' in cue for cue in visual_cues):
            if self._detect_screen_content(frame):
                score += 35
        
        return min(score, 100)
    
    def calculate_semantic_score(self, frame_path: str, section: Dict) -> float:
        """Calculate semantic score for a frame relative to a section (alias for score_frame_for_section)."""
        return self.score_frame_for_section(frame_path, section)
    
    def _detect_ha_colors(self, frame: np.ndarray) -> bool:
        """Detect Home Assistant's characteristic orange/blue color scheme."""
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Home Assistant blue (hue ~210-240, high saturation)
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Home Assistant orange (hue ~10-25, high saturation)
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([15, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        blue_ratio = np.sum(blue_mask > 0) / (frame.shape[0] * frame.shape[1])
        orange_ratio = np.sum(orange_mask > 0) / (frame.shape[0] * frame.shape[1])
        
        return blue_ratio > 0.01 or orange_ratio > 0.01
    
    def _detect_ui_elements(self, frame: np.ndarray) -> bool:
        """Detect common UI elements like buttons, forms, etc."""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for rectangular UI elements
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ui_rectangles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Reasonable button/form size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:  # Reasonable UI element proportions
                    ui_rectangles += 1
        
        return ui_rectangles > 3
    
    def _detect_screen_content(self, frame: np.ndarray) -> bool:
        """Detect if frame shows computer screen or interface content."""
        
        # Look for high contrast regions typical of screen content
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local standard deviation (texture analysis)
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_diff = (gray.astype(np.float32) - local_mean) ** 2
        variance = cv2.filter2D(sqr_diff, -1, kernel)
        # Ensure no negative values due to floating point precision
        variance = np.maximum(variance, 0.0)
        local_std = np.sqrt(variance)
        
        # High standard deviation indicates text/UI elements
        high_contrast_ratio = np.sum(local_std > 20) / (frame.shape[0] * frame.shape[1])
        
        return high_contrast_ratio > 0.1


class SemanticFrameSelector:
    """Main class for semantic-driven frame selection."""
    
    def __init__(self, config: Dict, video_processor=None):
        self.config = config
        self.video_processor = video_processor
        self.section_analyzer = SemanticSectionAnalyzer(config)
        self.frame_scorer = SemanticFrameScorer(config)
    
    def select_frames_semantically(
        self, 
        video_path: str, 
        transcript_segments: List[Dict], 
        temp_dir: str,
        video_title: str = None
    ) -> List[Dict]:
        """Select frames using semantic analysis of transcript content."""
        
        logger.info("🧠 Starting semantic frame selection...")
        
        # Store video title for title sequence detection
        self.frame_scorer.video_title = video_title or ""
        
        # Step 1: Analyze transcript for semantic sections
        sections = self.section_analyzer.analyze_transcript_sections(transcript_segments)
        
        if not sections:
            logger.error("❌ No semantic sections created - falling back to simple extraction")
            return []
        
        logger.info(f"📋 Created {len(sections)} semantic sections:")
        for i, section in enumerate(sections):
            logger.info(f"   Section {i+1}: '{section['title']}' ({section['start_time']:.1f}s-{section['end_time']:.1f}s, {section.get('importance', 'medium')} importance)")
        
        # Step 2: Extract candidate frames ONCE for all sections (OPTIMIZATION!)
        logger.info("🎬 Extracting candidate frames for all sections...")
        all_candidate_frames = self._extract_all_candidate_frames(
            video_path, sections, temp_dir
        )
        
        if not all_candidate_frames:
            logger.error("❌ No candidate frames extracted - check video processing")
            return []
        
        logger.info(f"📸 Total candidate frames available: {len(all_candidate_frames)}")
        timestamps = [f"{f['timestamp']:.1f}s" for f in all_candidate_frames[:10]]
        logger.info(f"   Sample timestamps: {timestamps}{'...' if len(all_candidate_frames) > 10 else ''}")
        
        # Step 3: For each section, score and select from pre-extracted frames
        all_selected_frames = []
        
        for i, section in enumerate(sections):
            logger.info(f"🎯 Processing section {i+1}: {section['title']}")
            section_frames = self._select_frames_for_section_optimized(
                section, all_candidate_frames
            )
            all_selected_frames.extend(section_frames)
        
        # Step 4: Apply cross-section diversity filtering
        logger.info("🎨 Applying cross-section diversity filtering...")
        diversity_filtered_frames = self._apply_cross_section_diversity(all_selected_frames)
        
        # Step 5: Final deduplication and quality filtering
        final_frames = self._deduplicate_and_filter_frames(diversity_filtered_frames)
        
        logger.info(f"✅ Semantic selection complete: {len(final_frames)} frames selected")
        return final_frames
    
    def select_frames_from_blog_content(
        self,
        video_path: str,
        transcript_segments: List[Dict],
        formatted_blog_content: str,
        temp_dir: str,
        video_title: str = None,
        blog_formatter = None
    ) -> List[Dict]:
        """Select frames using semantic analysis of formatted blog content instead of raw transcript."""
        
        logger.info("🧠 Starting semantic frame selection from formatted blog content...")
        
        # Store video title for title sequence detection
        self.frame_scorer.video_title = video_title or ""
        
        # Step 1: Analyze formatted blog content for semantic sections
        sections = self._analyze_blog_content_sections(formatted_blog_content, transcript_segments, blog_formatter)
        
        if not sections:
            logger.error("❌ No semantic sections created from blog content - timestamp mapping failed")
            raise ValueError("Failed to map blog sections to timestamps. The improved flow requires accurate section-to-timestamp mapping.")
        
        logger.info(f"📋 Created {len(sections)} semantic sections from blog content:")
        for i, section in enumerate(sections):
            logger.info(f"   Section {i+1}: '{section['title']}' ({section['start_time']:.1f}s-{section['end_time']:.1f}s, {section.get('importance', 'medium')} importance)")
        
        # Step 2: Extract candidate frames ONCE for all sections (OPTIMIZATION!)
        logger.info("🎬 Extracting candidate frames for all sections...")
        all_candidate_frames = self._extract_all_candidate_frames(
            video_path, sections, temp_dir
        )
        
        if not all_candidate_frames:
            logger.error("❌ No candidate frames extracted - check video processing")
            return []
        
        logger.info(f"📸 Total candidate frames available: {len(all_candidate_frames)}")
        timestamps = [f"{f['timestamp']:.1f}s" for f in all_candidate_frames[:10]]
        logger.info(f"   Sample timestamps: {timestamps}{'...' if len(all_candidate_frames) > 10 else ''}")
        
        # Step 3: For each section, score and select from pre-extracted frames
        all_selected_frames = []
        for section in sections:
            logger.info(f"🎯 Selecting frames for section: '{section['title']}'")
            section_frames = self._select_frames_for_section_from_candidates(
                all_candidate_frames, section
            )
            all_selected_frames.extend(section_frames)
        
        logger.info(f"📋 Initial selection: {len(all_selected_frames)} frames from {len(sections)} sections")
        
        # Step 4: Apply cross-section diversity filtering
        logger.info("🎨 Applying cross-section diversity filtering...")
        diversity_filtered_frames = self._apply_cross_section_diversity(all_selected_frames)
        
        # Step 5: Final deduplication and quality filtering
        final_frames = self._deduplicate_and_filter_frames(diversity_filtered_frames)
        
        logger.info(f"✅ Blog-content semantic selection complete: {len(final_frames)} frames selected")
        return final_frames
    
    def _select_frames_for_section(
        self, 
        video_path: str, 
        section: Dict, 
        temp_dir: str
    ) -> List[Dict]:
        """Select the best frames for a specific semantic section."""
        
        start_time = section['start_time']
        end_time = section['end_time']
        importance = section.get('importance', 'medium')
        
        logger.info(f"   📍 Section: {start_time:.1f}s-{end_time:.1f}s ({importance} importance)")
        
        # Extract candidate frames in time window
        candidate_frames = self._extract_candidate_frames(
            video_path, start_time, end_time, temp_dir
        )
        
        if not candidate_frames:
            logger.warning(f"   ⚠️  No candidate frames found for section")
            return []
        
        # Score each candidate frame for semantic relevance
        scored_frames = []
        for frame_info in candidate_frames:
            score = self.frame_scorer.score_frame_for_section(
                frame_info['path'], section
            )
            frame_info['semantic_score'] = score
            frame_info['section_title'] = section['title']
            frame_info['section_importance'] = importance
            scored_frames.append((score, frame_info))
        
        # Sort by score and select top frames
        scored_frames.sort(key=lambda x: x[0], reverse=True)
        
        # Determine how many frames to select based on importance
        frame_counts = {'high': 3, 'medium': 2, 'low': 1}
        max_frames = frame_counts.get(importance, 2)
        
        selected_frames = []
        for score, frame_info in scored_frames[:max_frames]:
            if score > self.frame_scorer.score_threshold:
                selected_frames.append(frame_info)
                logger.info(f"   ✅ Selected frame {frame_info['timestamp']:.1f}s (score: {score:.1f})")
            else:
                logger.info(f"   ❌ Rejected frame {frame_info['timestamp']:.1f}s (score: {score:.1f} < {self.frame_scorer.score_threshold})")
        
        return selected_frames
    
    def _extract_all_candidate_frames(
        self, 
        video_path: str, 
        sections: List[Dict], 
        temp_dir: str
    ) -> List[Dict]:
        """Extract candidate frames once for all sections (OPTIMIZED)."""
        
        if not self.video_processor:
            logger.error("VideoProcessor required for frame extraction")
            return []
        
        # Find the overall time range covering all sections
        min_time = min(section['start_time'] for section in sections)
        max_time = max(section['end_time'] for section in sections)
        
        logger.info(f"   📍 Extracting frames from {min_time:.1f}s to {max_time:.1f}s (covers all sections)")
        
        # Create fake transcript segments for the combined time range
        fake_segments = [
            {
                'start_time': min_time,
                'end_time': max_time,
                'text': 'Content for semantic analysis across all sections'
            }
        ]
        
        # Override config for denser sampling (since we're doing this once)
        original_interval = self.config.get('frame_sample_interval', 15)
        self.config['frame_sample_interval'] = 3  # Very dense sampling - every 3 seconds
        
        try:
            # Extract frames using existing processor
            frames = self.video_processor.extract_frames(
                video_path, temp_dir, fake_segments
            )
            
            # Filter to combined time range and ensure should_include=True
            candidate_frames = []
            for frame in frames:
                if (min_time <= frame['timestamp'] <= max_time and 
                    frame.get('should_include', False)):
                    candidate_frames.append(frame)
            
            logger.info(f"   📸 Extracted {len(candidate_frames)} candidate frames for all sections")
            return candidate_frames
            
        finally:
            # Restore original config
            self.config['frame_sample_interval'] = original_interval
    
    def _select_frames_for_section_optimized(
        self, 
        section: Dict, 
        all_candidate_frames: List[Dict]
    ) -> List[Dict]:
        """Select the best frames for a section from pre-extracted candidates (OPTIMIZED)."""
        
        start_time = section['start_time']
        end_time = section['end_time']
        importance = section.get('importance', 'medium')
        
        logger.info(f"   📍 Section: {start_time:.1f}s-{end_time:.1f}s ({importance} importance)")
        
        # Filter pre-extracted frames to this section's time window
        section_candidates = []
        for frame in all_candidate_frames:
            if start_time <= frame['timestamp'] <= end_time:
                section_candidates.append(frame)
        
        if not section_candidates:
            logger.warning(f"   ⚠️  No candidate frames found in time window")
            return []
        
        logger.info(f"   📸 Found {len(section_candidates)} candidate frames in section window")
        
        # Score each candidate frame for semantic relevance
        scored_frames = []
        for frame_info in section_candidates:
            score = self.frame_scorer.score_frame_for_section(
                frame_info['path'], section
            )
            frame_info['semantic_score'] = score
            frame_info['section_title'] = section['title']
            frame_info['section_importance'] = importance
            scored_frames.append((score, frame_info))
        
        # Sort by score and select top frames
        scored_frames.sort(key=lambda x: x[0], reverse=True)
        
        # Determine how many frames to select based on importance
        frame_counts = {'high': 3, 'medium': 2, 'low': 1}
        max_frames = frame_counts.get(importance, 2)
        
        selected_frames = []
        min_threshold = 30.0  # Lower threshold - was 50, too high
        
        if scored_frames:
            logger.info(f"   📊 Top frame scores: {[f'{s:.1f}' for s, f in scored_frames[:5]]}")
        
        for score, frame_info in scored_frames[:max_frames]:
            if score > self.frame_scorer.score_threshold:
                selected_frames.append(frame_info)
                logger.info(f"   ✅ Selected frame {frame_info['timestamp']:.1f}s (score: {score:.1f})")
            else:
                logger.info(f"   ❌ Rejected frame {frame_info['timestamp']:.1f}s (score: {score:.1f} < {self.frame_scorer.score_threshold})")
        
        # If no frames meet threshold, take the best one anyway (fallback)
        if not selected_frames and scored_frames:
            best_score, best_frame = scored_frames[0]
            selected_frames.append(best_frame)
            logger.warning(f"   🔄 Fallback: Selected best frame {best_frame['timestamp']:.1f}s (score: {best_score:.1f}) despite low score")
        
        return selected_frames
    
    def _apply_cross_section_diversity(self, selected_frames: List[Dict]) -> List[Dict]:
        """Apply cross-section diversity filtering to avoid visually similar frames."""
        
        if len(selected_frames) <= 1:
            return selected_frames
        
        logger.info(f"   🔍 Checking diversity across {len(selected_frames)} selected frames")
        
        # Group frames by section for better analysis
        sections_frames = {}
        for frame in selected_frames:
            section = frame.get('section_title', 'Unknown')
            if section not in sections_frames:
                sections_frames[section] = []
            sections_frames[section].append(frame)
        
        # Calculate similarity penalties and adjust scores
        diversity_adjusted_frames = []
        
        for section, frames in sections_frames.items():
            logger.info(f"   🎯 Processing section '{section}' ({len(frames)} frames)")
            
            for frame in frames:
                # Calculate similarity penalty against all OTHER selected frames
                other_frames = [f for f in selected_frames if f != frame]
                similarity_penalty = self._calculate_similarity_penalty(frame, other_frames)
                
                # Adjust semantic score with diversity penalty
                original_score = frame.get('semantic_score', 0)
                adjusted_score = original_score - similarity_penalty
                frame['diversity_adjusted_score'] = adjusted_score
                frame['similarity_penalty'] = similarity_penalty
                
                logger.debug(f"     Frame {frame['timestamp']:.1f}s: {original_score:.1f} -> {adjusted_score:.1f} (penalty: {similarity_penalty:.1f})")
        
        # Re-select frames based on diversity-adjusted scores, maintaining section limits
        final_frames = []
        
        for section, frames in sections_frames.items():
            # Sort by diversity-adjusted score
            frames.sort(key=lambda x: x['diversity_adjusted_score'], reverse=True)
            
            # Determine how many frames this section should get (aligned with selection logic)
            importance = frames[0].get('section_importance', 'medium') if frames else 'medium'
            frame_counts = {'high': 5, 'medium': 4, 'low': 3}  # Aligned with updated selection logic
            max_frames = frame_counts.get(importance, 4)
            
            selected_count = 0
            for frame in frames:
                # Take best frames up to section limit, with minimum quality threshold
                # Use a much lower diversity threshold since penalties can be aggressive
                diversity_threshold = self.config.get('semantic_frame_selection', {}).get('diversity_threshold', 10.0)
                if selected_count < max_frames and frame['diversity_adjusted_score'] > diversity_threshold:
                    final_frames.append(frame)
                    selected_count += 1
                    logger.info(f"   ✅ Kept {frame['timestamp']:.1f}s from '{section}' (diversity score: {frame['diversity_adjusted_score']:.1f})")
                else:
                    reason = "limit reached" if selected_count >= max_frames else "score too low"
                    logger.info(f"   ❌ Dropped {frame['timestamp']:.1f}s from '{section}' ({reason})")
        
        logger.info(f"   📊 Diversity filtering: {len(selected_frames)} -> {len(final_frames)} frames")
        return final_frames
    
    def _calculate_similarity_penalty(self, frame: Dict, other_frames: List[Dict]) -> float:
        """Calculate similarity penalty for a frame against other selected frames."""
        
        if not other_frames:
            return 0.0
        
        frame_path = frame['path']
        max_penalty = 0.0
        similar_count = 0
        
        for other_frame in other_frames:
            other_path = other_frame['path']
            
            try:
                # Calculate visual similarity using existing image hash method
                frame_hash = self._calculate_image_hash_for_diversity(frame_path)
                other_hash = self._calculate_image_hash_for_diversity(other_path)
                
                if frame_hash and other_hash:
                    similarity = self._calculate_hash_similarity_for_diversity(frame_hash, other_hash)
                    
                    # Convert similarity to penalty (higher similarity = higher penalty)
                    if similarity > 0.8:  # Very similar images
                        penalty = 40.0
                        similar_count += 1
                    elif similarity > 0.6:  # Somewhat similar
                        penalty = 25.0
                    elif similarity > 0.4:  # Slightly similar
                        penalty = 10.0
                    else:
                        penalty = 0.0
                    
                    max_penalty = max(max_penalty, penalty)
                    
                    if penalty > 0:
                        logger.debug(f"       Similarity to {other_frame['timestamp']:.1f}s: {similarity:.3f} (penalty: {penalty:.1f})")
                
            except Exception as e:
                logger.debug(f"       Could not compare with {other_frame['timestamp']:.1f}s: {e}")
                continue
        
        # Additional penalty for multiple similar frames
        if similar_count > 1:
            max_penalty += (similar_count - 1) * 10.0
            logger.debug(f"       Multiple similarity penalty: +{(similar_count - 1) * 10.0:.1f}")
        
        return min(max_penalty, 60.0)  # Cap penalty at 60 points
    
    def _calculate_image_hash_for_diversity(self, image_path: str) -> str:
        """Calculate perceptual hash for diversity comparison (reuse existing logic)."""
        try:
            # Load image and convert to grayscale
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return ""
            
            # Resize to 9x8 for difference hash
            resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
            
            # Calculate horizontal differences
            hash_bits = []
            for row in resized:
                for i in range(len(row) - 1):
                    hash_bits.append('1' if row[i] > row[i + 1] else '0')
            
            # Convert to hex string
            binary_string = ''.join(hash_bits)
            hex_parts = []
            for i in range(0, len(binary_string), 60):
                chunk = binary_string[i:i+60]
                if chunk:
                    hex_parts.append(format(int(chunk, 2), 'x'))
            
            return ''.join(hex_parts)
            
        except Exception as e:
            logger.debug(f"Hash calculation failed for {image_path}: {e}")
            return ""
    
    def _calculate_hash_similarity_for_diversity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between hashes for diversity filtering."""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        
        # Count matching characters
        matching_chars = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matching_chars / len(hash1)
    
    def _extract_candidate_frames(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: float, 
        temp_dir: str
    ) -> List[Dict]:
        """Extract candidate frames from a time window."""
        
        if not self.video_processor:
            logger.error("VideoProcessor required for frame extraction")
            return []
        
        # Create fake transcript segments for this time window to use existing extraction
        fake_segments = [
            {
                'start_time': start_time,
                'end_time': end_time,
                'text': 'Content for semantic analysis'
            }
        ]
        
        # Use existing video processor but with dense sampling in time window
        logger.info(f"   🎬 Extracting candidate frames from {start_time:.1f}s-{end_time:.1f}s...")
        
        # Override config for denser sampling in semantic windows
        original_interval = self.config.get('frame_sample_interval', 15)
        self.config['frame_sample_interval'] = 5  # Sample every 5 seconds
        
        try:
            # Extract frames using existing processor
            frames = self.video_processor.extract_frames(
                video_path, temp_dir, fake_segments
            )
            
            # Filter to time window and ensure should_include=True
            window_frames = []
            for frame in frames:
                if (start_time <= frame['timestamp'] <= end_time and 
                    frame.get('should_include', False)):
                    window_frames.append(frame)
            
            logger.info(f"   📸 Found {len(window_frames)} candidate frames in window")
            return window_frames
            
        finally:
            # Restore original config
            self.config['frame_sample_interval'] = original_interval
    
    def _deduplicate_and_filter_frames(self, frames: List[Dict]) -> List[Dict]:
        """Remove duplicate and low-quality frames from final selection."""
        
        if not frames:
            return []
        
        # Sort by semantic score
        frames.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
        
        # Remove frames that are too close in time (prevent duplicates)
        filtered_frames = []
        # Much smaller gap for technical content - configurable
        min_time_gap = self.config.get('semantic_frame_selection', {}).get('min_time_gap_seconds', 0.5)
        
        for frame in frames:
            timestamp = frame['timestamp']
            
            # Check if too close to any already selected frame
            too_close = any(
                abs(timestamp - existing['timestamp']) < min_time_gap
                for existing in filtered_frames
            )
            
            if not too_close:
                filtered_frames.append(frame)
            else:
                logger.debug(f"   🗑️  Deduplicating frame at {timestamp:.1f}s (too close to existing frame)")
        
        # Apply similarity filtering (reuse existing similarity logic)
        if len(filtered_frames) > 1:
            similarity_threshold = self.config.get('image_similarity_threshold', 0.15)
            logger.info(f"   🔍 Applying similarity filtering (threshold={similarity_threshold})")
            
            # Use existing similarity filtering from hugo_generator
            # This requires importing HugoGenerator - we'll do this when integrating
            pass
        
        logger.info(f"   📊 Final selection: {len(filtered_frames)} frames after deduplication")
        return filtered_frames
    
    def compare_with_temporal_selection(
        self, 
        video_path: str, 
        transcript_segments: List[Dict], 
        temp_dir: str
    ) -> Dict:
        """Compare semantic selection with traditional temporal selection."""
        
        # Run semantic selection
        semantic_frames = self.select_frames_semantically(
            video_path, transcript_segments, temp_dir
        )
        
        # Run traditional temporal selection (using existing method)
        if self.video_processor:
            temporal_frames = self.video_processor.extract_frames(
                video_path, temp_dir, transcript_segments
            )
            # Filter to should_include=True
            temporal_frames = [f for f in temporal_frames if f.get('should_include', False)]
        else:
            temporal_frames = []
        
        # Compare results
        comparison = {
            'semantic_count': len(semantic_frames),
            'temporal_count': len(temporal_frames),
            'semantic_timestamps': [f['timestamp'] for f in semantic_frames],
            'temporal_timestamps': [f['timestamp'] for f in temporal_frames],
            'semantic_sections': len(set(f.get('section_title', '') for f in semantic_frames)),
            'improvement_ratio': len(semantic_frames) / max(len(temporal_frames), 1)
        }
        
        logger.info("📊 SELECTION COMPARISON:")
        logger.info(f"   Semantic: {comparison['semantic_count']} frames from {comparison['semantic_sections']} sections")
        logger.info(f"   Temporal: {comparison['temporal_count']} frames")
        logger.info(f"   Ratio: {comparison['improvement_ratio']:.2f}")
        
        return comparison
    
    def _analyze_blog_content_sections(self, formatted_blog_content: str, transcript_segments: List[Dict], blog_formatter=None) -> List[Dict]:
        """Analyze formatted blog content to extract semantic sections with timestamps."""
        
        logger.info("🧠 Analyzing formatted blog content for semantic sections...")
        
        # Try to get boundary map from blog formatter if available
        boundary_map = getattr(blog_formatter, 'boundary_map', {}) if blog_formatter else {}
        
        # Extract headers from blog content (these are the semantic sections)
        import re
        header_pattern = r'^(#{1,3})\s+(.+)$'
        lines = formatted_blog_content.split('\n')
        
        sections = []
        current_section = None
        section_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            if header_match:
                # Save previous section if exists
                if current_section:
                    current_section['content'] = '\n'.join(section_content)
                    current_section['end_time'] = self._estimate_section_end_time(
                        current_section, transcript_segments
                    )
                    sections.append(current_section)
                
                # Start new section
                header_level = len(header_match.group(1))
                section_title = header_match.group(2).strip()
                
                # Try to get start time from boundary map first, then fallback to estimation
                if section_title in boundary_map:
                    start_time = boundary_map[section_title]
                    logger.info(f"   🎯 Using boundary marker for '{section_title}' at {start_time:.1f}s")
                else:
                    start_time = self._estimate_section_start_time(section_title, transcript_segments)
                
                current_section = {
                    'title': section_title,
                    'start_time': start_time,
                    'importance': self._determine_section_importance(header_level, section_title),
                    'visual_cues': self._extract_visual_cues_from_title(section_title),
                    'mentioned_text': self._extract_key_terms_from_title(section_title),
                    'ui_elements': []
                }
                section_content = []
            else:
                if current_section:
                    section_content.append(line)
        
        # Add final section
        if current_section:
            current_section['content'] = '\n'.join(section_content)
            current_section['end_time'] = self._estimate_section_end_time(
                current_section, transcript_segments
            )
            sections.append(current_section)
        
        # Filter out sections that are too short or couldn't be mapped to timestamps
        valid_sections = []
        for section in sections:
            if (section['start_time'] is not None and 
                section['end_time'] is not None and
                section['end_time'] > section['start_time']):
                valid_sections.append(section)
                logger.info(f"   ✅ Section: '{section['title']}' ({section['start_time']:.1f}s-{section['end_time']:.1f}s)")
            else:
                logger.warning(f"   ⚠️  Skipped section: '{section['title']}' (couldn't map to timestamps)")
        
        logger.info(f"📋 Extracted {len(valid_sections)} valid sections from blog content")
        return valid_sections
    
    def _estimate_section_start_time(self, section_title: str, transcript_segments: List[Dict]) -> Optional[float]:
        """Estimate when a section starts based on transcript content (fallback only)."""
        
        # This should only be used as fallback when boundary markers fail
        logger.warning(f"   ⚠️  Using fallback mapping for '{section_title}' - boundary markers failed")
        
        # If no match found, return None (section will be filtered out)
        return None
    
    def _estimate_section_end_time(self, section: Dict, transcript_segments: List[Dict]) -> Optional[float]:
        """Estimate when a section ends based on content length and transcript."""
        
        start_time = section['start_time']
        if start_time is None:
            return None
        
        # Estimate duration based on content length (rough heuristic)
        content_length = len(section.get('content', ''))
        estimated_duration = max(30, min(180, content_length / 20))  # 30-180 seconds
        
        # Find actual end based on transcript segments
        end_time = start_time + estimated_duration
        
        # Make sure we don't exceed total video duration
        if transcript_segments:
            # Handle both key formats for compatibility
            max_time = max(seg.get('end_time', seg.get('end', 0)) for seg in transcript_segments)
            end_time = min(end_time, max_time)
        
        return end_time
    
    def _determine_section_importance(self, header_level: int, section_title: str) -> str:
        """Determine section importance based on header level and title."""
        
        # H1 = high, H2 = medium, H3+ = low
        if header_level == 1:
            return 'high'
        elif header_level == 2:
            return 'medium'
        else:
            return 'low'
    
    def _extract_visual_cues_from_title(self, section_title: str) -> List[str]:
        """Extract visual cues from section title."""
        
        title_lower = section_title.lower()
        visual_cues = []
        
        # Common visual cue patterns
        if 'setup' in title_lower or 'configuration' in title_lower:
            visual_cues.extend(['settings', 'configuration', 'setup screens'])
        if 'dashboard' in title_lower or 'interface' in title_lower:
            visual_cues.extend(['dashboard', 'main interface', 'ui'])
        if 'device' in title_lower or 'automation' in title_lower:
            visual_cues.extend(['devices', 'automation', 'controls'])
        if 'install' in title_lower or 'add' in title_lower:
            visual_cues.extend(['installation', 'add screens', 'dialogs'])
        
        return visual_cues or ['interface']  # Default fallback
    
    def _extract_key_terms_from_title(self, section_title: str) -> List[str]:
        """Extract key searchable terms from section title."""
        
        # Split title into words and filter out common words
        import re
        words = re.findall(r'\b\w+\b', section_title.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'why', 'when', 'where'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Limit to top 5 terms
    
    def _select_frames_for_section_from_candidates(self, candidate_frames: List[Dict], section: Dict) -> List[Dict]:
        """Select best frames for a section from pre-extracted candidates."""
        
        start_time = section['start_time']
        end_time = section['end_time']
        
        # Filter candidates to section timeframe
        section_candidates = [
            frame for frame in candidate_frames
            if start_time <= frame['timestamp'] <= end_time
        ]
        
        if not section_candidates:
            logger.warning(f"   ⚠️  No candidate frames found for section '{section['title']}' ({start_time:.1f}s-{end_time:.1f}s)")
            return []
        
        # Score each candidate frame for this section
        scored_frames = []
        for frame_info in section_candidates:
            score = self.frame_scorer.calculate_semantic_score(
                frame_info['path'], section
            )
            frame_info['semantic_score'] = score
            frame_info['section_title'] = section['title']
            frame_info['section_importance'] = section.get('importance', 'medium')
            scored_frames.append((score, frame_info))
        
        # Sort by score and select top frames
        scored_frames.sort(key=lambda x: x[0], reverse=True)
        
        # Determine how many frames to select based on section importance and duration
        section_duration = end_time - start_time
        importance = section.get('importance', 'medium')
        
        if importance == 'high':
            target_frames = max(3, min(5, int(section_duration / 45)))  # More generous
        elif importance == 'medium':
            target_frames = max(2, min(4, int(section_duration / 60)))  # At least 2 frames, up to 4
        else:  # low importance
            target_frames = max(1, min(3, int(section_duration / 90)))   # At least 1, up to 3
        
        # Apply score threshold filtering
        score_threshold = self.config.get('semantic_frame_selection', {}).get('score_threshold', 35.0)
        selected_frames = []
        
        for score, frame_info in scored_frames[:target_frames * 2]:  # Consider more candidates
            if score >= score_threshold:
                selected_frames.append(frame_info)
                if len(selected_frames) >= target_frames:
                    break
        
        scores = [f['semantic_score'] for f in selected_frames]
        formatted_scores = [f"{score:.1f}" for score in scores]
        logger.info(f"   🎯 Selected {len(selected_frames)} frames for '{section['title']}' (scores: {formatted_scores})")
        return selected_frames