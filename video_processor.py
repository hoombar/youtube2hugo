"""Video processing module for frame extraction and analysis."""

import cv2
import ffmpeg
import os
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video analysis and frame extraction."""
    
    def __init__(self, config: Dict):
        self.config = config
        # Use OpenCV's Haar cascade for face detection as fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_frames(self, video_path: str, output_dir: str, transcript_segments: List[Dict] = None) -> List[Dict]:
        """Extract frames from video using intelligent content-aware selection."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get video info
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(video_info['duration'])
        fps = eval(video_info['r_frame_rate'])
        
        logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}")
        
        # Get candidate timestamps based on content and intervals
        candidate_timestamps = self._get_intelligent_timestamps(duration, transcript_segments)
        
        # Add focused dense sampling only where needed (first 30s for rapid sequences)
        dense_sampling_end = min(30, duration * 0.1)  # Only first 30s
        dense_timestamps = list(np.arange(5, dense_sampling_end, 2.0))  # Every 2 seconds starting at 5s
        
        # Add smart content-aware dense sampling around key moments
        if transcript_segments:
            # Look for segments with visual keywords and add nearby timestamps
            visual_keywords = ['show', 'see', 'look', 'here', 'screen', 'this', 'device', 'interface']
            for segment in transcript_segments:
                text = segment.get('text', '').lower()
                if any(keyword in text for keyword in visual_keywords):
                    start_time = segment.get('start', 0)
                    if start_time < 60:  # Only first minute for speed
                        dense_timestamps.extend([start_time - 1, start_time, start_time + 1])
        
        # Combine with existing timestamps and remove duplicates
        all_candidates = sorted(set(candidate_timestamps + dense_timestamps))
        candidate_timestamps = all_candidates
        
        extracted_frames = []
        last_selected_timestamp = None
        
        for i, timestamp in enumerate(candidate_timestamps):
            # Optimized spacing: reasonable for early periods, wider for later content
            if timestamp < 30:  # First 30 seconds - allow closer spacing for rapid sequences
                min_spacing = 2.0
            elif timestamp < 60:  # Rest of first minute
                min_spacing = 4.0  
            else:
                min_spacing = 12.0
            
            # Skip if too close to last selected frame
            if (last_selected_timestamp is not None and 
                abs(timestamp - last_selected_timestamp) < min_spacing):
                continue
            
            # Get single best frame near this timestamp
            best_frame = self._extract_single_frame_near_timestamp(
                video_path, output_dir, timestamp, fps, extracted_frames
            )
            
            if best_frame:
                # Check for duplicates before adding
                is_duplicate = any(abs(best_frame['timestamp'] - existing['timestamp']) < 0.1 
                                 for existing in extracted_frames)
                
                if not is_duplicate:
                    # Add required fields for compatibility
                    best_frame['should_include'] = True
                    best_frame['face_ratio'] = self._analyze_frame_composition(best_frame['path'])
                    extracted_frames.append(best_frame)
                    last_selected_timestamp = timestamp
                    
                    logger.info(f"Selected frame at {best_frame['timestamp']:.1f}s: score={best_frame.get('score', 0):.3f}")
                else:
                    logger.debug(f"Skipped duplicate frame at {best_frame['timestamp']:.1f}s")
            else:
                logger.info(f"Rejected frame at {timestamp:.1f}s: no suitable frames found")
                
        return extracted_frames
    
    def _get_intelligent_timestamps(self, duration: float, transcript_segments: List[Dict] = None) -> List[float]:
        """Generate intelligent timestamps based on content analysis and intervals."""
        timestamps = []
        
        # Base interval sampling (reduced frequency, avoid intro/outro)
        base_interval = self.config.get('frame_sample_interval', 20)  # Increased from 15 to 20
        intro_skip = min(30, duration * 0.1)  # Skip first 30s or 10% of video, whichever is smaller
        outro_skip = min(20, duration * 0.05)  # Skip last 20s or 5% of video
        base_timestamps = list(np.arange(intro_skip + base_interval, duration - outro_skip, base_interval))
        
        # Content-aware timestamps from transcript
        content_timestamps = []
        if transcript_segments:
            content_timestamps = self._get_content_aware_timestamps(transcript_segments)
        
        # Combine and deduplicate
        all_timestamps = set(base_timestamps + content_timestamps)
        
        # Remove timestamps too close to each other (minimum 10 seconds apart)
        sorted_timestamps = sorted(all_timestamps)
        filtered_timestamps = []
        
        for ts in sorted_timestamps:
            if not filtered_timestamps or ts - filtered_timestamps[-1] >= 10:
                filtered_timestamps.append(ts)
        
        return filtered_timestamps
    
    def _get_content_aware_timestamps(self, transcript_segments: List[Dict]) -> List[float]:
        """Extract timestamps based on content keywords and topic changes."""
        high_priority_keywords = [
            # Direct visual references
            'show', 'see', 'look', 'here', 'screen', 'interface', 'example',
            'configuration', 'settings', 'setup', 'config', 'database', 'log',
            'error', 'diagram', 'code', 'script', 'file', 'folder', 'directory',
            # Demonstration phrases
            'going to do', 'made earlier', 'take a look', 'you can see',
            'here we have', 'if we look', 'what we have', 'this is',
            # Technical content indicators
            'device', 'network', 'coordinator', 'analysis', 'mapping',
            'instances', 'address', 'friendly name'
        ]
        
        medium_priority_keywords = [
            # Topic transitions
            'first', 'next', 'then', 'second', 'third', 'finally',
            'important', 'key', 'critical', 'main', 'primary',
            # Content structuring  
            'problem', 'solution', 'issue', 'feature', 'option',
            'method', 'approach', 'way', 'technique'
        ]
        
        timestamps = []
        
        for i, segment in enumerate(transcript_segments):
            text = segment.get('text', '').lower()
            start_time = segment.get('start_time', 0)
            
            segment_score = 0
            
            # High priority content (visual demonstrations)
            for keyword in high_priority_keywords:
                if keyword in text:
                    segment_score += 3
                    break
            
            # Medium priority content (topic transitions)
            for keyword in medium_priority_keywords:
                if keyword in text:
                    segment_score += 1
                    break
            
            # Look for specific patterns that indicate visual content
            visual_patterns = [
                'can see', 'you see', 'shows', 'displays', 'appears',
                'pop up', 'click', 'select', 'choose', 'pick',
                'window', 'dialog', 'menu', 'button', 'form',
                'table', 'list', 'grid', 'chart', 'graph'
            ]
            
            for pattern in visual_patterns:
                if pattern in text:
                    segment_score += 2
                    break
            
            # Bonus for segments that mention specific tools/software
            tool_keywords = [
                'claude', 'zigbee', 'mqtt', 'home assistant', 'nabu casa',
                'analyzer', 'coordinator', 'router', 'device'
            ]
            
            for tool in tool_keywords:
                if tool in text:
                    segment_score += 1
                    break
            
            # Include timestamps with high enough scores
            if segment_score >= 2:  # At least medium relevance
                timestamps.append(start_time)
                
            # Also include segments with dramatic topic changes
            if i > 0:
                prev_text = transcript_segments[i-1].get('text', '').lower()
                current_words = set(text.split())
                prev_words = set(prev_text.split())
                
                # If very few words overlap, it might be a topic change
                overlap = len(current_words.intersection(prev_words))
                if overlap < 3 and len(current_words) > 5:
                    timestamps.append(start_time)
        
        return timestamps
    
    def _detect_content_density_hotspots(self, video_path: str, candidate_timestamps: List[float]) -> List[float]:
        """DISABLED: Content density hotspot detection removed for performance."""
        return []  # Hotspot detection disabled for speed
    
    def _extract_single_frame_near_timestamp(self, video_path: str, output_dir: str, target_timestamp: float, fps: float, existing_frames: List[Dict]) -> Dict:
        """Extract the best single frame near a target timestamp."""
        candidate_frames = []
        window_size = 3.0  # Check 3 seconds around target
        
        # Test just 2 frames around the target timestamp for speed
        for offset in [0, 1.0]:  # Check frames at target and +1s
            test_timestamp = max(0, target_timestamp + offset)
            frame_path = os.path.join(output_dir, f"candidate_{test_timestamp:.1f}s.jpg")
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=test_timestamp)
                    .output(frame_path, vframes=1, format='image2', vcodec='mjpeg', s='640x360')  # Smaller resolution for speed
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)  # Suppress output
                )
                
                # Score this candidate
                score = self._score_frame_quality(frame_path, 
                    existing_frames[-1]['path'] if existing_frames else None)
                
                candidate_frames.append({
                    'path': frame_path,
                    'timestamp': test_timestamp,
                    'score': score
                })
                
            except Exception as e:
                logger.warning(f"Could not extract frame at {test_timestamp}s: {e}")
                continue
        
        # Select best frame only if it meets quality threshold
        if candidate_frames:
            best_frame = max(candidate_frames, key=lambda x: x['score'])
            
            # Quality threshold - prefer no image over bad image (tuned from reverse engineering)
            min_quality_score = 61.2  # Minimum acceptable score
            
            if best_frame['score'] >= 85.0:  # Balanced threshold
                # Clean up non-selected frames
                for frame in candidate_frames:
                    if frame != best_frame and os.path.exists(frame['path']):
                        os.remove(frame['path'])
                
                # Rename best frame to final name
                final_path = os.path.join(os.path.dirname(best_frame['path']), 
                                        f"frame_{best_frame['timestamp']:.1f}s.jpg")
                os.rename(best_frame['path'], final_path)
                best_frame['path'] = final_path
                
                return best_frame
            else:
                # Clean up all frames - none meet quality threshold
                logger.info(f"No suitable frame found near {target_timestamp:.1f}s (best score: {best_frame['score']:.1f})")
                for frame in candidate_frames:
                    if os.path.exists(frame['path']):
                        os.remove(frame['path'])
        
        return None
        
        # Sample more densely around candidates to find rapid changes
        for timestamp in candidate_timestamps:
            # Test 20-second window around each candidate with 1-second intervals
            test_start = max(0, timestamp - 10)
            test_end = timestamp + 10
            test_times = list(np.arange(test_start, test_end, 1.0))
            
            if len(test_times) < 3:
                continue
            
            frame_scores = []
            rapid_changes = []
            previous_frame = None
            
            for test_time in test_times:
                try:
                    # Extract small test frame for speed
                    test_frame_path = f"/tmp/density_test_{test_time:.1f}.jpg"
                    (
                        ffmpeg
                        .input(video_path, ss=test_time)
                        .output(test_frame_path, vframes=1, format='image2', vcodec='mjpeg', s='160x120')
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    
                    current_frame = cv2.imread(test_frame_path, cv2.IMREAD_GRAYSCALE)
                    if current_frame is not None:
                        # Quick quality score (simplified)
                        edges = cv2.Canny(current_frame, 50, 150)
                        edge_density = np.sum(edges > 0) / (current_frame.shape[0] * current_frame.shape[1])
                        
                        # Face detection (quick)
                        faces = self.face_cascade.detectMultiScale(current_frame, scaleFactor=1.3, minNeighbors=3)
                        face_penalty = len(faces) * 50
                        
                        quick_score = edge_density * 100 - face_penalty
                        frame_scores.append((test_time, quick_score))
                        
                        # Detect rapid changes
                        if previous_frame is not None:
                            change = np.mean(np.abs(current_frame.astype(float) - previous_frame.astype(float)))
                            rapid_changes.append(change)
                        
                        previous_frame = current_frame
                    
                    # Cleanup
                    if os.path.exists(test_frame_path):
                        os.remove(test_frame_path)
                        
                except Exception as e:
                    logger.debug(f"Error testing density at {test_time}s: {e}")
                    continue
            
            # Analyze results for this window
            if len(frame_scores) >= 5 and len(rapid_changes) >= 3:
                # Count high-quality frames
                good_frames = sum(1 for _, score in frame_scores if score > 20)
                
                # Check for rapid changes (jump cuts)
                avg_change = np.mean(rapid_changes)
                max_change = np.max(rapid_changes)
                change_peaks = sum(1 for change in rapid_changes if change > avg_change * 1.5)
                
                # Determine if this is a hotspot
                is_hotspot = False
                
                # Multiple good frames in small window (reduced threshold)
                if good_frames >= 2:
                    is_hotspot = True
                    logger.info(f"Content density hotspot at {timestamp:.1f}s: {good_frames} good frames")
                
                # Rapid content switching (jump cuts)
                if change_peaks >= 2 and avg_change > 15:
                    is_hotspot = True
                    logger.info(f"Jump cut hotspot at {timestamp:.1f}s: {change_peaks} peaks, avg change {avg_change:.1f}")
                
                if is_hotspot:
                    hotspots.append(timestamp)
        
        logger.info(f"Found {len(hotspots)} content density hotspots")
        return hotspots
    
    def _get_multiple_candidates_near_timestamp(self, video_path: str, output_dir: str, target_timestamp: float, fps: float, existing_frames: List[Dict], window_size: float) -> List[Dict]:
        """Get multiple candidate frames near timestamp for dense content areas."""
        candidates = []
        
        # Smaller steps in hotspots to catch rapid changes
        frame_step = 0.5  # Check every 0.5 seconds in hotspots
        start_time = max(0, target_timestamp - window_size)
        end_time = target_timestamp + window_size
        
        for offset in np.arange(start_time, end_time, frame_step):
            frame_path = os.path.join(output_dir, f"candidate_{offset:.1f}s.jpg")
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=offset)
                    .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                # Score this candidate
                score = self._score_frame_quality(frame_path, 
                    existing_frames[-1]['path'] if existing_frames else None)
                
                candidates.append({
                    'timestamp': offset,
                    'path': frame_path,
                    'score': score
                })
                
            except ffmpeg.Error as e:
                logger.warning(f"Could not extract candidate at {offset}s: {e}")
                continue
        
        return candidates
    
    def _select_best_candidates(self, candidates: List[Dict], existing_frames: List[Dict], in_hotspot: bool) -> List[Dict]:
        """Select best candidate(s) - multiple if in hotspot with diverse content."""
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter by minimum quality (balanced to catch targets while reducing unwanted)
        good_candidates = [c for c in candidates if c['score'] >= 85.0]
        
        if not good_candidates:
            # Clean up rejected candidates
            for candidate in candidates:
                if os.path.exists(candidate['path']):
                    os.remove(candidate['path'])
            return []
        
        selected = []
        
        # Always take the best candidate
        best = good_candidates[0]
        selected.append(best)
        
        # In hotspots, consider additional candidates if significantly different
        if in_hotspot and len(good_candidates) > 1:
            for candidate in good_candidates[1:]:
                # Check if significantly different from already selected
                is_different = True
                for selected_candidate in selected:
                    diversity = self._calculate_diversity_bonus(
                        cv2.imread(candidate['path']), 
                        selected_candidate['path']
                    )
                    if diversity < 60:  # Not different enough
                        is_different = False
                        break
                
                # Also check difference from recent existing frames
                if is_different and len(existing_frames) > 0:
                    recent_diversity = self._calculate_diversity_bonus(
                        cv2.imread(candidate['path']),
                        existing_frames[-1]['path']
                    )
                    if recent_diversity < 60:
                        is_different = False
                
                # Add if different and high quality
                if is_different and candidate['score'] >= 61.2:  # Use same threshold as main selection
                    candidate['is_clustered'] = True  # Mark for smaller display
                    selected.append(candidate)
                    if len(selected) >= 3:  # Max 3 frames per hotspot
                        break
        
        # Clean up non-selected candidates
        selected_paths = {c['path'] for c in selected}
        for candidate in candidates:
            if candidate['path'] not in selected_paths and os.path.exists(candidate['path']):
                os.remove(candidate['path'])
        
        # Rename selected candidates to final names
        final_selected = []
        for candidate in selected:
            final_path = os.path.join(os.path.dirname(candidate['path']), 
                                    f"frame_{candidate['timestamp']:.1f}s.jpg")
            os.rename(candidate['path'], final_path)
            candidate['path'] = final_path
            final_selected.append(candidate)
        
        return final_selected
    
    def _find_best_frame_near_timestamp(self, video_path: str, output_dir: str, target_timestamp: float, fps: float, previous_frame_path: str = None) -> Dict:
        """Find the best frame within a window around the target timestamp."""
        window_size = 3.0  # seconds
        frame_step = 1.0   # seconds
        
        candidate_frames = []
        
        # Sample frames around the target timestamp
        start_time = max(0, target_timestamp - window_size)
        end_time = target_timestamp + window_size
        
        for offset in np.arange(start_time, end_time, frame_step):
            frame_path = os.path.join(output_dir, f"candidate_{offset:.1f}s.jpg")
            
            try:
                # Extract frame
                (
                    ffmpeg
                    .input(video_path, ss=offset)
                    .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                # Analyze frame quality
                score = self._score_frame_quality(frame_path, previous_frame_path)
                
                candidate_frames.append({
                    'timestamp': offset,
                    'path': frame_path,
                    'score': score
                })
                
            except ffmpeg.Error as e:
                logger.warning(f"Could not extract frame at {offset}s: {e}")
                continue
        
        # Select best frame only if it meets quality threshold
        if candidate_frames:
            best_frame = max(candidate_frames, key=lambda x: x['score'])
            
            # Quality threshold - prefer no image over bad image (tuned from reverse engineering)
            min_quality_score = 61.2  # Minimum acceptable score
            
            if best_frame['score'] >= 85.0:  # Balanced threshold
                # Clean up non-selected frames
                for frame in candidate_frames:
                    if frame != best_frame and os.path.exists(frame['path']):
                        os.remove(frame['path'])
                
                # Rename best frame to final name
                final_path = os.path.join(output_dir, f"frame_{best_frame['timestamp']:.1f}s.jpg")
                os.rename(best_frame['path'], final_path)
                best_frame['path'] = final_path
                
                return best_frame
            else:
                # Clean up all frames - none meet quality threshold
                logger.info(f"No suitable frame found near {target_timestamp:.1f}s (best score: {best_frame['score']:.1f})")
                for frame in candidate_frames:
                    if os.path.exists(frame['path']):
                        os.remove(frame['path'])
        
        return None
    
    def _analyze_frame_composition(self, frame_path: str) -> float:
        """Analyze frame to determine face-to-screen ratio using OpenCV."""
        image = cv2.imread(frame_path)
        if image is None:
            return 0.0
            
        height, width, _ = image.shape
        total_area = height * width
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return 0.0
            
        total_face_area = 0
        for (x, y, w, h) in faces:
            face_area = w * h
            total_face_area += face_area
            
        face_ratio = total_face_area / total_area
        return face_ratio
    
    def _score_frame_quality(self, frame_path: str, previous_frame_path: str = None) -> float:
        """Score frame quality based on multiple factors to avoid talking heads and prefer interesting content."""
        image = cv2.imread(frame_path)
        if image is None:
            return 0.0
            
        height, width, _ = image.shape
        total_area = height * width
        score = 0.0
        
        # 1. Advanced talking head detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        talking_head_penalty = 0.0
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_area = w * h
                face_ratio = face_area / total_area
                
                # Calculate face position
                face_center_x = x + w/2
                face_center_y = y + h/2
                center_x = width / 2
                center_y = height / 2
                
                # Talking head detection criteria:
                # 1. Face takes up significant portion (>8% for close-up shots)
                # 2. Face is in upper-center region (typical talking head position)
                # 3. Check for "portrait-like" composition
                
                is_talking_head = False
                
                # More aggressive face detection - any prominent face is problematic
                
                # 1. Large face is almost always talking head
                if face_ratio > 0.12:  # Lowered from 0.15
                    is_talking_head = True
                
                # 2. Face in center region (very strict)
                if face_ratio > 0.05:  # Even smaller faces
                    # Check if face is anywhere near center (tighter bounds)
                    if (face_center_y < height * 0.7 and  # Upper 70% of frame  
                        abs(face_center_x - center_x) < width * 0.25):  # Center 50% horizontally
                        is_talking_head = True
                
                # 3. Face dominates visual attention
                if face_ratio > 0.04:  # Lowered threshold
                    # Check surrounding content
                    non_face_region = np.copy(gray)
                    non_face_region[y:y+h, x:x+w] = 0  # Zero out face region
                    
                    # Calculate edge density outside face region
                    edges_non_face = cv2.Canny(non_face_region, 50, 150)
                    non_face_edge_density = np.sum(edges_non_face > 0) / total_area
                    
                    # If there's little interesting content besides the face, likely talking head
                    if non_face_edge_density < 0.08:  # Increased threshold for required content
                        is_talking_head = True
                
                # 4. Face takes up significant vertical space (portrait-like)
                face_height_ratio = h / height
                if face_height_ratio > 0.25 and face_ratio > 0.06:
                    is_talking_head = True
                
                if is_talking_head:
                    talking_head_penalty += 200  # Heavy penalty
                    logger.debug(f"Detected talking head: face_ratio={face_ratio:.3f}, position=({face_center_x:.0f},{face_center_y:.0f})")
                elif face_ratio > 0.03:
                    # Small penalty for any noticeable face (prefer content over people)
                    talking_head_penalty += face_ratio * 20
        
        score -= talking_head_penalty
        
        # 2. Screen content detection (multiple approaches)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_area
        
        # Detect screen/monitor characteristics
        screen_content_score = 0
        
        # A. High edge density suggests text, code, or UI elements
        if edge_density > 0.12:  # Very rich content
            screen_content_score += 60
        elif edge_density > 0.08:  # Good content
            screen_content_score += 40
        elif edge_density < 0.02:  # Too bland
            screen_content_score -= 20
        
        # B. Look for regular patterns (text lines, UI grids)
        # Horizontal line detection (common in text and UI)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / total_area
        
        if horizontal_score > 0.02:  # Strong horizontal patterns
            screen_content_score += 40
        
        # C. Vertical line detection (sidebars, menus, code indentation)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        vertical_score = np.sum(vertical_lines > 0) / total_area
        
        if vertical_score > 0.01:  # Vertical structures
            screen_content_score += 30
        
        # D. Rectangle detection for UI elements, windows, boxes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_shapes = 0
        large_rectangles = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Reasonable size
                # Check if it's roughly rectangular
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangular shape
                    rectangular_shapes += 1
                    if area > 2000:  # Large UI elements
                        large_rectangles += 1
        
        if rectangular_shapes > 5:  # Many UI elements
            screen_content_score += 50
        elif rectangular_shapes > 2:
            screen_content_score += 25
        
        if large_rectangles > 0:  # Windows, dialogs, major UI components
            screen_content_score += 30
        
        # E. Color analysis for screen content
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check for typical screen colors (blues, whites, greys)
        # Many UIs use blue/white/grey color schemes
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        grey_mask = cv2.inRange(hsv, (0, 0, 50), (180, 30, 200))
        
        screen_color_ratio = (np.sum(blue_mask > 0) + np.sum(white_mask > 0) + np.sum(grey_mask > 0)) / total_area
        
        if screen_color_ratio > 0.4:  # Predominantly screen-like colors
            screen_content_score += 25
        
        # F. Dark mode detection (common in development/technical content)
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:  # Dark background
            # Check for bright text on dark background
            bright_pixels = np.sum(gray > 180)
            bright_ratio = bright_pixels / total_area
            if bright_ratio > 0.1:  # Significant bright text
                screen_content_score += 35  # Dark mode interfaces are great content
        
        score += screen_content_score
        
        # 3. Enhanced title sequence detection (further reduced penalty)
        title_penalty = self._detect_specific_intro_sequence(image, gray)
        # Apply very reduced intro penalty to avoid rejecting good content
        if title_penalty > 0:
            title_penalty = min(title_penalty * 0.1, 20.0)  # Much smaller penalty
        score -= title_penalty
        
        # 4. Text detection bonus
        # Look for high-contrast regions that might be text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_regions = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 10), np.uint8))
        text_score = np.sum(text_regions > 0) / total_area
        
        if text_score > 0.2:  # Lots of potential text
            score += 40
        
        # 5. Brightness check (avoid too dark or too bright frames)
        brightness = np.mean(gray)
        if 50 < brightness < 200:  # Good brightness range
            score += 10
        else:
            score -= 15
        
        # 6. Diversity bonus - reward frames significantly different from previous
        diversity_bonus = 0
        if previous_frame_path and os.path.exists(previous_frame_path):
            diversity_bonus = self._calculate_diversity_bonus(image, previous_frame_path)
            score += diversity_bonus
        
        # 7. Face-free bonus - massive reward for frames with no faces
        if len(faces) == 0:
            face_free_bonus = 100  # Large bonus for completely face-free frames
            score += face_free_bonus
            logger.debug(f"Face-free bonus: +{face_free_bonus}")
        
        # 8. Device/gadget detection bonus (tuned from reverse engineering)
        device_bonus = self._detect_devices_and_gadgets(image, gray)
        # Apply recommended device bonus scaling: 86.7 points for detected devices
        if device_bonus > 0:
            device_bonus = min(device_bonus * 0.72, 86.7)  # Scale to recommended max
        score += device_bonus
        
        return max(0, score)  # Ensure non-negative score
    
    def _calculate_diversity_bonus(self, current_image: np.ndarray, previous_frame_path: str) -> float:
        """Calculate bonus score for visual diversity compared to previous frame."""
        try:
            previous_image = cv2.imread(previous_frame_path)
            if previous_image is None:
                return 0
            
            # Resize both images to same size for comparison
            height, width = current_image.shape[:2]
            previous_resized = cv2.resize(previous_image, (width, height))
            
            # Convert to grayscale for comparison
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate multiple difference metrics
            
            # 1. Mean Squared Error (MSE) - higher is more different
            mse = np.mean((current_gray.astype(float) - previous_gray.astype(float)) ** 2)
            diversity_from_mse = min(mse / 100, 80)  # Normalize and cap
            
            # 2. Histogram difference (color distribution)
            current_hist = cv2.calcHist([current_image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            previous_hist = cv2.calcHist([previous_resized], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            hist_diff = cv2.compareHist(current_hist, previous_hist, cv2.HISTCMP_BHATTACHARYYA)
            diversity_from_hist = hist_diff * 60  # Scale up
            
            # 3. Edge pattern difference (structural content)
            current_edges = cv2.Canny(current_gray, 50, 150)
            previous_edges = cv2.Canny(previous_gray, 50, 150)
            edge_diff = np.mean(np.abs(current_edges.astype(float) - previous_edges.astype(float))) / 255
            diversity_from_edges = edge_diff * 50
            
            # Combine diversity metrics
            total_diversity = diversity_from_mse + diversity_from_hist + diversity_from_edges
            
            # Cap the bonus to prevent extreme scores
            diversity_bonus = min(total_diversity, 120)
            
            logger.debug(f"Diversity analysis: MSE={mse:.1f}, Hist={hist_diff:.3f}, Edge={edge_diff:.3f}, Total bonus={diversity_bonus:.1f}")
            
            return diversity_bonus
            
        except Exception as e:
            logger.warning(f"Error calculating diversity: {e}")
            return 0
    
    def _detect_devices_and_gadgets(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect devices, gadgets, and hardware in the frame for bonus scoring."""
        device_score = 0
        height, width = gray.shape
        
        # 1. Circular/rounded objects (common in many devices)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 0:
                device_score += min(len(circles) * 15, 60)  # Up to 60 points for circular objects
        
        # 2. Small rectangular objects (buttons, switches, ports, displays)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_rectangles = 0
        medium_rectangles = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Small to medium sized objects
                # Check if roughly rectangular
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    if area < 1000:
                        small_rectangles += 1  # Buttons, switches, small displays
                    else:
                        medium_rectangles += 1  # Larger devices, panels
        
        if small_rectangles > 3:
            device_score += 40  # Many small components suggest device
        if medium_rectangles > 0:
            device_score += 30  # Medium components
        
        # 3. LED/indicator detection (small bright spots)
        # Look for very bright, small circular regions
        _, bright_thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        led_indicators = 0
        for contour in bright_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 200:  # Small bright spots (LEDs)
                led_indicators += 1
        
        if led_indicators > 2:
            device_score += 35  # Multiple LEDs suggest electronic device
        
        # 4. Color-based device detection
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Common device colors: black/grey electronics, blue screens, green circuit boards
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        blue_screen_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        green_pcb_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        
        black_ratio = np.sum(black_mask > 0) / (height * width)
        blue_ratio = np.sum(blue_screen_mask > 0) / (height * width)
        green_ratio = np.sum(green_pcb_mask > 0) / (height * width)
        
        if black_ratio > 0.3:  # Significant black/grey (electronics)
            device_score += 25
        if blue_ratio > 0.2:  # Blue screens/displays
            device_score += 30
        if green_ratio > 0.1:  # Green circuit boards
            device_score += 40
        
        # 5. Text/label detection (device labels, model numbers)
        # Look for small text regions that might be device labels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_regions = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _, text_binary = cv2.threshold(text_regions, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        text_contours, _ = cv2.findContours(text_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_text_regions = 0
        
        for contour in text_contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Small text (labels, model numbers)
                small_text_regions += 1
        
        if small_text_regions > 5:
            device_score += 20  # Multiple small text regions suggest device labels
        
        # Cap the device bonus
        device_score = min(device_score, 120)
        
        if device_score > 30:
            logger.debug(f"Device detection bonus: +{device_score}")
        
        return device_score
    
    def _detect_specific_intro_sequence(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect your specific intro style: lightbulb+house logo on blue/green/black background."""
        intro_penalty = 0
        height, width = gray.shape
        
        # 1. Color scheme detection (blue/green/black background)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Your brand colors
        blue_bg_mask = cv2.inRange(hsv, (100, 30, 30), (130, 255, 255))  # Blue backgrounds
        green_bg_mask = cv2.inRange(hsv, (40, 30, 30), (80, 255, 200))   # Green backgrounds  
        black_bg_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))      # Dark/black backgrounds
        
        brand_color_ratio = (np.sum(blue_bg_mask > 0) + np.sum(green_bg_mask > 0) + np.sum(black_bg_mask > 0)) / (height * width)
        
        if brand_color_ratio > 0.6:  # Predominantly brand colors
            intro_penalty += 80
            logger.debug("Detected brand color scheme")
        
        # 2. Logo shape detection (lightbulb + house combination)
        # Look for circular shapes (lightbulb) near triangular/rectangular shapes (house)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                 param1=50, param2=30, minRadius=15, maxRadius=80)
        
        # Find triangular/house-like shapes
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        triangular_shapes = 0
        house_like_shapes = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Reasonable logo size
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 3:  # Triangle (roof)
                    triangular_shapes += 1
                elif len(approx) == 4:  # Rectangle (house body)
                    house_like_shapes += 1
        
        # Logo combination detection
        has_circles = circles is not None and len(circles[0]) > 0
        has_house_shapes = triangular_shapes > 0 or house_like_shapes > 0
        
        if has_circles and has_house_shapes:
            intro_penalty += 100
            logger.debug("Detected lightbulb + house logo combination")
        elif has_circles and brand_color_ratio > 0.4:
            intro_penalty += 60
            logger.debug("Detected circular logo on brand background")
        
        # 3. Centered composition detection (typical of title cards)
        center_region = gray[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.75)]
        edge_density_center = np.sum(cv2.Canny(center_region, 50, 150) > 0) / (center_region.shape[0] * center_region.shape[1])
        
        # Peripheral regions
        top_region = gray[:int(height*0.25), :]
        bottom_region = gray[int(height*0.75):, :]
        left_region = gray[:, :int(width*0.25)]
        right_region = gray[:, int(width*0.75):]
        
        peripheral_density = 0
        for region in [top_region, bottom_region, left_region, right_region]:
            if region.size > 0:
                peripheral_density += np.sum(cv2.Canny(region, 50, 150) > 0) / region.size
        peripheral_density /= 4
        
        # If center has much more content than periphery (typical logo layout)
        if edge_density_center > peripheral_density * 3 and edge_density_center > 0.1:
            intro_penalty += 50
            logger.debug("Detected centered logo composition")
        
        # 4. Solid/gradient background detection
        color_variance = np.var(image)
        if color_variance < 800:  # Very uniform (solid colors/gradients)
            intro_penalty += 40
            logger.debug("Detected uniform background")
        
        # 5. Text detection in intro style (channel name, title text)
        # Look for medium-sized text regions in center
        _, text_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_contours, _ = cv2.findContours(text_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center_text_regions = 0
        for contour in text_contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Medium text size
                # Check if in center region
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    if (width*0.2 < cx < width*0.8 and height*0.3 < cy < height*0.7):
                        center_text_regions += 1
        
        if center_text_regions >= 2:  # Multiple text elements in center
            intro_penalty += 40
            logger.debug("Detected centered text elements")
        
        # Cap the penalty
        intro_penalty = min(intro_penalty, 200)
        
        if intro_penalty > 50:
            logger.debug(f"Intro sequence detection penalty: -{intro_penalty}")
        
        return intro_penalty
    
    def _should_include_frame(self, face_ratio: float) -> bool:
        """Legacy method - now using _score_frame_quality instead."""
        return face_ratio < 0.15  # More strict face detection
    
    def optimize_images(self, frame_info_list: List[Dict]) -> List[Dict]:
        """Optimize extracted images for web use."""
        optimized_frames = []
        
        for frame_info in frame_info_list:
            if not frame_info['should_include']:
                continue
                
            original_path = frame_info['path']
            optimized_path = original_path.replace('.jpg', '_optimized.jpg')
            
            try:
                with Image.open(original_path) as img:
                    # Resize if needed
                    max_width = self.config.get('image_max_width', 1920)
                    max_height = self.config.get('image_max_height', 1080)
                    
                    if img.width > max_width or img.height > max_height:
                        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    # Save optimized version
                    quality = self.config.get('image_quality', 95)
                    img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
                    
                    # Update frame info
                    frame_info['optimized_path'] = optimized_path
                    optimized_frames.append(frame_info)
                    
                    logger.info(f"Optimized frame: {optimized_path}")
                    
            except Exception as e:
                logger.error(f"Error optimizing image {original_path}: {e}")
                continue
                
        return optimized_frames