#!/usr/bin/env python3
"""
Frame Selection Trainer - Manual frame selection tool for algorithm training.

This tool creates a web interface for manually selecting the best frames from extracted
candidates, then analyzes the selections to improve the automatic frame selection algorithm.
"""

import os
import json
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np

# Import existing modules
from transcript_extractor import TranscriptExtractor
from blog_formatter import BlogFormatter
from video_processor import VideoProcessor
from semantic_frame_selector import SemanticFrameSelector, SemanticFrameScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameSelectionTrainer:
    """Tool for manually selecting frames and training the algorithm from ground truth data."""
    
    def __init__(self, config_path: str = "config.local.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Storage for training data
        self.training_data_dir = "training_data"
        os.makedirs(self.training_data_dir, exist_ok=True)
        
    def setup_routes(self):
        """Setup Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            return render_template('frame_selector.html')
        
        @self.app.route('/process_video', methods=['POST'])
        def process_video():
            """Process video and prepare frames for selection."""
            try:
                data = request.json
                video_path = data['video_path']
                
                # Handle video file path - check if it exists and is a valid video file
                if not os.path.exists(video_path):
                    # If no extension, try common video extensions
                    if not os.path.splitext(video_path)[1]:
                        # First try adding extensions directly
                        for ext in ['.mp4', '.mov', '.mkv', '.avi', '.m4v']:
                            test_path = video_path + ext
                            if os.path.exists(test_path):
                                video_path = test_path
                                logger.info(f"Found video file: {video_path}")
                                break
                        else:
                            # If that fails, check if it's a directory and look for video files inside
                            if os.path.isdir(video_path):
                                basename = os.path.basename(video_path)
                                for ext in ['.mp4', '.mov', '.mkv', '.avi', '.m4v']:
                                    test_path = os.path.join(video_path, basename + ext)
                                    if os.path.exists(test_path):
                                        video_path = test_path
                                        logger.info(f"Found video file in directory: {video_path}")
                                        break
                                else:
                                    return jsonify({'error': f'No video file found in directory: {video_path}'}), 400
                            else:
                                return jsonify({'error': f'Video file not found: {video_path} (tried common extensions)'}), 400
                    else:
                        return jsonify({'error': f'Video file not found: {video_path}'}), 400
                
                result = self.prepare_frame_selection(video_path)
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/images/<path:filename>')
        def serve_image(filename):
            """Serve frame images."""
            # Try multiple possible directories
            for directory in ['temp_frames', '.']:
                image_path = os.path.join(directory, filename)
                if os.path.exists(image_path):
                    return send_from_directory(directory, filename)
            
            # If not found, return a 404 error
            logger.error(f"Image not found: {filename}")
            # Debug: list what files are actually available
            if os.path.exists('temp_frames'):
                available_files = os.listdir('temp_frames')
                logger.error(f"Available files in temp_frames: {available_files}")
            return "Image not found", 404
        
        @self.app.route('/save_selections', methods=['POST'])
        def save_selections():
            """Save manual frame selections."""
            try:
                data = request.json
                self.save_training_data(data)
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Error saving selections: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/analyze_selections', methods=['POST'])
        def analyze_selections():
            """Analyze saved selections to improve algorithm."""
            try:
                data = request.json
                video_id = data['video_id']
                analysis = self.analyze_training_data(video_id)
                return jsonify(analysis)
            except Exception as e:
                logger.error(f"Error analyzing selections: {e}")
                return jsonify({'error': str(e)}), 500
    
    def prepare_frame_selection(self, video_path: str) -> Dict:
        """Process video and prepare frames for manual selection."""
        logger.info(f"Processing video: {video_path}")
        
        # Create unique ID for this training session
        video_id = Path(video_path).stem
        session_dir = os.path.join(self.training_data_dir, video_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Step 1: Extract transcript
        logger.info("Extracting transcript...")
        transcript_extractor = TranscriptExtractor(self.config)
        transcript_segments = transcript_extractor.extract_transcript(video_path)
        
        # Step 2: Format into sectioned blog post with retry logic
        logger.info("Formatting blog content...")
        blog_formatter = BlogFormatter(self.config)
        
        # Try up to 2 times with Gemini before falling back
        for attempt in range(2):
            try:
                logger.info(f"Attempting Gemini formatting (attempt {attempt + 1}/2)...")
                blog_content = blog_formatter.format_transcript_content(
                    transcript_segments, 
                    f"Training Video: {video_id}"
                )
                boundary_map = getattr(blog_formatter, 'boundary_map', {})
                logger.info("✅ Gemini formatting succeeded")
                break
            except (SystemExit, ValueError) as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed ({type(e).__name__}: {e})")
                if attempt == 1:  # Last attempt failed
                    logger.warning("All Gemini attempts failed - using basic sectioning")
                    blog_content = self._create_basic_sections(transcript_segments)
                    boundary_map = self._create_basic_boundary_map(transcript_segments)
                    break
                else:
                    logger.info("Retrying with fresh BlogFormatter instance...")
                    blog_formatter = BlogFormatter(self.config)  # Fresh instance
        
        # Step 3: Extract ALL candidate frames
        logger.info("Extracting candidate frames...")
        video_processor = VideoProcessor(self.config)
        temp_frames_dir = "temp_frames"
        
        # Robust cleanup: completely clear temp_frames directory
        self._clear_temp_frames(temp_frames_dir)
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        candidate_frames = video_processor.extract_frames(video_path, temp_frames_dir, transcript_segments)
        
        # Step 4: Group frames by sections
        sections_with_frames = self._group_frames_by_sections(
            candidate_frames, boundary_map, blog_content
        )
        
        # Step 5: Save session data
        session_data = {
            'video_id': video_id,
            'video_path': video_path,
            'blog_content': blog_content,
            'boundary_map': boundary_map,
            'sections_with_frames': sections_with_frames,
            'total_frames': len(candidate_frames)
        }
        
        with open(os.path.join(session_dir, 'session_data.json'), 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return session_data
    
    def _create_basic_sections(self, transcript_segments: List[Dict]) -> str:
        """Create basic sections when Gemini is unavailable."""
        # Simple time-based sectioning
        total_duration = max(seg.get('end_time', seg.get('end', 0)) for seg in transcript_segments)
        section_duration = total_duration / 5  # 5 sections
        
        blog_content = "## Introduction\n\nContent from video transcript.\n\n"
        
        for i in range(1, 5):
            start_time = i * section_duration
            blog_content += f"## Section {i}\n\nContent from {start_time:.1f}s onwards.\n\n"
        
        blog_content += "## Conclusion\n\nSummary of the content.\n\n"
        return blog_content
    
    def _create_basic_boundary_map(self, transcript_segments: List[Dict]) -> Dict:
        """Create basic boundary map when Gemini is unavailable."""
        total_duration = max(seg.get('end_time', seg.get('end', 0)) for seg in transcript_segments)
        section_duration = total_duration / 5
        
        return {
            'Introduction': 0.0,
            'Section 1': section_duration,
            'Section 2': section_duration * 2,
            'Section 3': section_duration * 3,
            'Section 4': section_duration * 4,
            'Conclusion': section_duration * 4.5
        }
    
    def _group_frames_by_sections(self, candidate_frames: List[Dict], boundary_map: Dict, blog_content: str) -> List[Dict]:
        """Group candidate frames by blog sections."""
        # Extract section headers from blog content
        import re
        headers = re.findall(r'^## (.+)$', blog_content, re.MULTILINE)
        
        sections = []
        section_boundaries = list(boundary_map.items()) if boundary_map else []
        
        # If no boundary map, create time-based sections
        if not section_boundaries and candidate_frames:
            max_time = max(frame['timestamp'] for frame in candidate_frames)
            section_duration = max_time / len(headers) if headers else max_time / 5
            section_boundaries = [(header, i * section_duration) for i, header in enumerate(headers)]
        
        for i, (section_title, start_time) in enumerate(section_boundaries):
            # Determine end time for this section
            if i + 1 < len(section_boundaries):
                end_time = section_boundaries[i + 1][1]
            else:
                end_time = float('inf')
            
            # Find frames in this section's time range
            section_frames = []
            for frame in candidate_frames:
                if start_time <= frame['timestamp'] < end_time:
                    # Add filename field for web interface
                    frame_copy = frame.copy()
                    if 'path' in frame_copy:
                        frame_copy['filename'] = os.path.basename(frame_copy['path'])
                    section_frames.append(frame_copy)
            
            # Sort frames by timestamp
            section_frames.sort(key=lambda x: x['timestamp'])
            
            sections.append({
                'title': section_title,
                'start_time': start_time,
                'end_time': end_time if end_time != float('inf') else None,
                'frames': section_frames,
                'frame_count': len(section_frames)
            })
        
        return sections
    
    def save_training_data(self, selection_data: Dict):
        """Save manual frame selections as training data."""
        video_id = selection_data['video_id']
        session_dir = os.path.join(self.training_data_dir, video_id)
        
        # Save selections
        selections_file = os.path.join(session_dir, 'manual_selections.json')
        with open(selections_file, 'w') as f:
            json.dump(selection_data, f, indent=2)
        
        # Copy selected frames to permanent storage
        selected_frames_dir = os.path.join(session_dir, 'selected_frames')
        os.makedirs(selected_frames_dir, exist_ok=True)
        
        for section in selection_data['sections']:
            for frame_path in section['selected_frames']:
                if os.path.exists(frame_path):
                    dest_path = os.path.join(selected_frames_dir, os.path.basename(frame_path))
                    shutil.copy2(frame_path, dest_path)
        
        logger.info(f"Saved training data for {video_id}")
    
    def analyze_training_data(self, video_id: str) -> Dict:
        """Analyze manual selections to identify patterns."""
        session_dir = os.path.join(self.training_data_dir, video_id)
        selections_file = os.path.join(session_dir, 'manual_selections.json')
        
        if not os.path.exists(selections_file):
            raise ValueError(f"No training data found for {video_id}")
        
        with open(selections_file, 'r') as f:
            selection_data = json.load(f)
        
        # Initialize analyzer
        frame_scorer = SemanticFrameScorer(self.config)
        
        analysis = {
            'selected_frames_analysis': [],
            'rejected_frames_analysis': [],
            'patterns': {},
            'recommendations': []
        }
        
        # Analyze selected vs rejected frames
        for section in selection_data['sections']:
            selected_paths = set(section['selected_frames'])
            
            for frame in section['available_frames']:
                frame_path = frame['path']
                is_selected = frame_path in selected_paths
                
                # Analyze frame characteristics
                frame_analysis = self._analyze_single_frame(frame_path, frame_scorer)
                frame_analysis['timestamp'] = frame['timestamp']
                frame_analysis['section'] = section['title']
                frame_analysis['selected'] = is_selected
                
                if is_selected:
                    analysis['selected_frames_analysis'].append(frame_analysis)
                else:
                    analysis['rejected_frames_analysis'].append(frame_analysis)
        
        # Identify patterns
        analysis['patterns'] = self._identify_patterns(
            analysis['selected_frames_analysis'],
            analysis['rejected_frames_analysis']
        )
        
        # Generate detailed analysis
        analysis['detailed_analysis'] = self._generate_detailed_analysis(analysis['patterns'])
        
        # Ensure all values are JSON serializable
        analysis = self._make_json_safe(analysis)
        
        return analysis
    
    def _make_json_safe(self, obj):
        """Recursively convert all values to JSON-safe types."""
        if isinstance(obj, dict):
            return {key: self._make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # Convert unknown types to string
            return str(obj)
    
    def _analyze_single_frame(self, frame_path: str, scorer: SemanticFrameScorer) -> Dict:
        """Analyze characteristics of a single frame."""
        try:
            # Check if file exists first to avoid OpenCV warnings
            if not os.path.exists(frame_path):
                # Try to find the frame in the training data directory
                filename = os.path.basename(frame_path)
                video_id = frame_path.split('/')[-3] if 'training_data' in frame_path else None
                if video_id:
                    alt_path = os.path.join(self.training_data_dir, video_id, 'selected_frames', filename)
                    if os.path.exists(alt_path):
                        frame_path = alt_path
                    else:
                        return {'error': f'Frame file not found: {filename}'}
                else:
                    return {'error': f'Frame file not found: {frame_path}'}
            
            frame = cv2.imread(frame_path)
            if frame is None:
                return {'error': f'Could not load frame: {os.path.basename(frame_path)}'}
            
            analysis = {}
            
            # Visual quality metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysis['brightness'] = float(np.mean(gray))
            analysis['contrast'] = float(np.std(gray))
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            analysis['edge_density'] = float(np.sum(edges > 0) / (frame.shape[0] * frame.shape[1]))
            
            # Color analysis - convert to standard Python booleans
            analysis['has_ha_colors'] = bool(scorer._detect_ha_colors(frame))
            analysis['has_ui_elements'] = bool(scorer._detect_ui_elements(frame))
            analysis['has_screen_content'] = bool(scorer._detect_screen_content(frame))
            
            # OCR analysis (if available)
            try:
                import pytesseract
                text = pytesseract.image_to_string(frame, config='--psm 6')
                analysis['ocr_text'] = text.strip()
                analysis['text_length'] = len(text.strip())
                analysis['has_text'] = len(text.strip()) > 10
            except:
                analysis['ocr_text'] = ""
                analysis['text_length'] = 0
                analysis['has_text'] = False
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _identify_patterns(self, selected_analysis: List[Dict], rejected_analysis: List[Dict]) -> Dict:
        """Identify patterns that distinguish selected from rejected frames."""
        patterns = {}
        
        if not selected_analysis or not rejected_analysis:
            return patterns
        
        # Calculate means for selected frames
        selected_means = {}
        for key in ['brightness', 'contrast', 'edge_density', 'text_length']:
            values = [frame.get(key, 0) for frame in selected_analysis if key in frame]
            selected_means[key] = float(np.mean(values)) if values else 0.0
        
        # Calculate means for rejected frames  
        rejected_means = {}
        for key in ['brightness', 'contrast', 'edge_density', 'text_length']:
            values = [frame.get(key, 0) for frame in rejected_analysis if key in frame]
            rejected_means[key] = float(np.mean(values)) if values else 0.0
        
        # Calculate boolean feature frequencies
        for key in ['has_ha_colors', 'has_ui_elements', 'has_screen_content', 'has_text']:
            selected_freq = float(np.mean([frame.get(key, False) for frame in selected_analysis]))
            rejected_freq = float(np.mean([frame.get(key, False) for frame in rejected_analysis]))
            patterns[f'{key}_preference'] = float(selected_freq - rejected_freq)
        
        patterns['selected_means'] = selected_means
        patterns['rejected_means'] = rejected_means
        patterns['differences'] = {
            key: float(selected_means[key] - rejected_means.get(key, 0))
            for key in selected_means
        }
        
        return patterns
    
    def _generate_detailed_analysis(self, patterns: Dict) -> Dict:
        """Generate detailed, actionable analysis with specific config recommendations."""
        analysis = {
            'detailed_metrics': {},
            'config_recommendations': {},
            'comparative_analysis': {},
            'actionable_changes': []
        }
        
        if not patterns:
            return analysis
        
        selected_means = patterns.get('selected_means', {})
        rejected_means = patterns.get('rejected_means', {})
        differences = patterns.get('differences', {})
        
        # Detailed metric analysis
        for metric in ['brightness', 'contrast', 'edge_density', 'text_length']:
            if metric in selected_means and metric in rejected_means:
                selected_val = selected_means[metric]
                rejected_val = rejected_means[metric]
                diff = differences.get(metric, 0)
                
                analysis['detailed_metrics'][metric] = {
                    'selected_avg': round(selected_val, 2),
                    'rejected_avg': round(rejected_val, 2),
                    'difference': round(diff, 2),
                    'preference_strength': self._categorize_preference(diff, metric)
                }
        
        # Feature preference analysis
        feature_analysis = {}
        for feature in ['has_ha_colors', 'has_ui_elements', 'has_screen_content', 'has_text']:
            pref_key = f'{feature}_preference'
            if pref_key in patterns:
                pref_value = patterns[pref_key]
                feature_analysis[feature] = {
                    'preference_score': round(pref_value, 3),
                    'selected_frequency': round(pref_value + rejected_means.get(feature, 0), 3),
                    'rejected_frequency': round(rejected_means.get(feature, 0), 3),
                    'strength': self._categorize_feature_preference(pref_value)
                }
        
        analysis['feature_preferences'] = feature_analysis
        
        # Generate specific config recommendations
        analysis['config_recommendations'] = self._generate_config_recommendations(patterns)
        
        # Generate actionable changes
        analysis['actionable_changes'] = self._generate_actionable_changes(patterns)
        
        return analysis
    
    def _categorize_preference(self, diff: float, metric: str) -> str:
        """Categorize the strength of preference for a metric."""
        thresholds = {
            'brightness': [5, 15, 30],
            'contrast': [3, 8, 15], 
            'edge_density': [0.005, 0.015, 0.03],
            'text_length': [5, 15, 30]
        }
        
        thresh = thresholds.get(metric, [1, 5, 10])
        abs_diff = abs(diff)
        
        if abs_diff < thresh[0]:
            return "No clear preference"
        elif abs_diff < thresh[1]:
            return "Weak preference" + (" for higher" if diff > 0 else " for lower")
        elif abs_diff < thresh[2]:
            return "Strong preference" + (" for higher" if diff > 0 else " for lower")
        else:
            return "Very strong preference" + (" for higher" if diff > 0 else " for lower")
    
    def _categorize_feature_preference(self, pref: float) -> str:
        """Categorize feature preference strength."""
        if abs(pref) < 0.1:
            return "No clear preference"
        elif abs(pref) < 0.3:
            return "Weak preference" + (" for" if pref > 0 else " against")
        elif abs(pref) < 0.6:
            return "Strong preference" + (" for" if pref > 0 else " against")
        else:
            return "Very strong preference" + (" for" if pref > 0 else " against")
    
    def _generate_config_recommendations(self, patterns: Dict) -> Dict:
        """Generate specific config file value recommendations."""
        config_recs = {}
        differences = patterns.get('differences', {})
        
        # Current config values (from self.config)
        current_config = self.config.get('semantic_frame_selection', {})
        
        # Scoring weight recommendations
        base_weight = current_config.get('base_score_weight', 0.5)
        text_weight = current_config.get('text_score_weight', 0.2)
        visual_weight = current_config.get('visual_score_weight', 0.3)
        
        # Adjust weights based on what user prefers
        brightness_diff = differences.get('brightness', 0)
        contrast_diff = differences.get('contrast', 0)
        edge_diff = differences.get('edge_density', 0)
        text_diff = differences.get('text_length', 0)
        
        # Calculate new base score weight (brightness + contrast + edges are part of base score)
        base_importance = (abs(brightness_diff) / 30 + abs(contrast_diff) / 15 + abs(edge_diff) / 0.03) / 3
        new_base_weight = min(0.8, max(0.2, base_weight + (base_importance - 0.5) * 0.3))
        
        # Calculate new text weight
        text_importance = abs(text_diff) / 30
        new_text_weight = min(0.5, max(0.1, text_weight + (text_importance - 0.5) * 0.2))
        
        # Visual weight is remainder
        new_visual_weight = max(0.1, 1.0 - new_base_weight - new_text_weight)
        
        # Normalize to sum to 1.0
        total = new_base_weight + new_text_weight + new_visual_weight
        new_base_weight /= total
        new_text_weight /= total 
        new_visual_weight /= total
        
        config_recs['scoring_weights'] = {
            'base_score_weight': round(new_base_weight, 2),
            'text_score_weight': round(new_text_weight, 2), 
            'visual_score_weight': round(new_visual_weight, 2),
            'current_base': base_weight,
            'current_text': text_weight,
            'current_visual': visual_weight
        }
        
        # Score threshold recommendation
        current_threshold = current_config.get('score_threshold', 35.0)
        # If user is selecting very few frames, lower threshold
        # If user is selecting many frames, raise threshold
        selection_ratio = len(patterns.get('selected_frames_analysis', [])) / (
            len(patterns.get('selected_frames_analysis', [])) + len(patterns.get('rejected_frames_analysis', []))
        ) if patterns.get('rejected_frames_analysis') else 0.5
        
        if selection_ratio < 0.1:  # Very selective
            new_threshold = max(25.0, current_threshold - 5.0)
        elif selection_ratio < 0.2:  # Somewhat selective  
            new_threshold = max(30.0, current_threshold - 2.0)
        elif selection_ratio > 0.4:  # Not very selective
            new_threshold = min(50.0, current_threshold + 5.0)
        else:
            new_threshold = current_threshold
        
        config_recs['score_threshold'] = {
            'recommended': round(new_threshold, 1),
            'current': current_threshold,
            'selection_ratio': round(selection_ratio, 3)
        }
        
        return config_recs
    
    def _generate_actionable_changes(self, patterns: Dict) -> List[Dict]:
        """Generate specific, actionable changes to improve the algorithm."""
        changes = []
        differences = patterns.get('differences', {})
        
        # Specific config file changes
        config_recs = self._generate_config_recommendations(patterns)
        
        if 'scoring_weights' in config_recs:
            weights = config_recs['scoring_weights']
            changes.append({
                'type': 'config_update',
                'file': 'config.local.yaml',
                'section': 'semantic_frame_selection',
                'changes': {
                    'base_score_weight': weights['base_score_weight'],
                    'text_score_weight': weights['text_score_weight'],
                    'visual_score_weight': weights['visual_score_weight']
                },
                'reason': f"Rebalance weights based on your preferences. Base: {weights['current_base']}→{weights['base_score_weight']}, Text: {weights['current_text']}→{weights['text_score_weight']}, Visual: {weights['current_visual']}→{weights['visual_score_weight']}"
            })
        
        if 'score_threshold' in config_recs:
            threshold = config_recs['score_threshold']
            if threshold['recommended'] != threshold['current']:
                changes.append({
                    'type': 'config_update',
                    'file': 'config.local.yaml', 
                    'section': 'semantic_frame_selection',
                    'changes': {
                        'score_threshold': threshold['recommended']
                    },
                    'reason': f"Adjust threshold from {threshold['current']} to {threshold['recommended']} based on your selection ratio ({threshold['selection_ratio']:.1%})"
                })
        
        # Algorithm improvements
        brightness_diff = differences.get('brightness', 0)
        if abs(brightness_diff) > 15:
            changes.append({
                'type': 'algorithm_improvement',
                'component': 'base_visual_score',
                'modification': f"Add brightness preference: {'bonus' if brightness_diff > 0 else 'penalty'} for {'bright' if brightness_diff > 0 else 'dark'} frames",
                'code_change': f"if brightness {'>' if brightness_diff > 0 else '<'} {160 if brightness_diff > 0 else 100}: score += {10 if brightness_diff > 0 else -10}"
            })
        
        contrast_diff = differences.get('contrast', 0)
        if abs(contrast_diff) > 10:
            changes.append({
                'type': 'algorithm_improvement',
                'component': 'base_visual_score',
                'modification': f"Increase contrast importance (current preference: {'+' if contrast_diff > 0 else '-'}{abs(contrast_diff):.1f})",
                'code_change': f"if contrast > {max(30, 30 + contrast_diff/2)}: score += {min(30, 25 + abs(contrast_diff)/2)}"
            })
        
        return changes
    
    def _clear_temp_frames(self, temp_frames_dir: str):
        """Robustly clear all files from temp_frames directory."""
        if os.path.exists(temp_frames_dir):
            try:
                # Remove entire directory and recreate
                shutil.rmtree(temp_frames_dir)
                logger.info(f"Cleared temp frames directory: {temp_frames_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temp_frames directory: {e}")
                # Fallback: try to remove individual files
                try:
                    for filename in os.listdir(temp_frames_dir):
                        file_path = os.path.join(temp_frames_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.debug(f"Removed old frame: {filename}")
                except Exception as e2:
                    logger.error(f"Failed to clear temp frames: {e2}")
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Start the web interface."""
        logger.info(f"Starting Frame Selection Trainer at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    trainer = FrameSelectionTrainer()
    trainer.run()