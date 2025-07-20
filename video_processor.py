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
        
        # 1. ULTRA-AGGRESSIVE talking head detection with multiple detection passes
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple detection passes with different parameters to catch more faces
        faces_pass1 = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25), maxSize=(400, 400))
        faces_pass2 = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20), maxSize=(300, 300))
        faces_pass3 = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30), maxSize=(500, 500))
        
        # Combine all detected faces and remove duplicates
        all_faces = []
        if len(faces_pass1) > 0:
            all_faces.extend(faces_pass1)
        if len(faces_pass2) > 0:
            all_faces.extend(faces_pass2)
        if len(faces_pass3) > 0:
            all_faces.extend(faces_pass3)
        
        # Remove duplicate/overlapping faces
        faces = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            for existing_face in faces:
                ex, ey, ew, eh = existing_face
                # Check if faces overlap significantly
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                face_area = w * h
                existing_area = ew * eh
                smaller_area = min(face_area, existing_area)
                
                if overlap_area > smaller_area * 0.5:  # 50% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                faces.append(face)
        
        logger.debug(f"Face detection: Pass1={len(faces_pass1)}, Pass2={len(faces_pass2)}, Pass3={len(faces_pass3)}, Final={len(faces)}")
        
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
                
                # ULTRA-AGGRESSIVE talking head detection - we NEVER want solo talking heads
                is_talking_head = False
                
                # 1. ANY moderately large face is a talking head (even more aggressive)
                if face_ratio > 0.06:  # Lowered from 0.08 - catch smaller faces
                    is_talking_head = True
                    logger.debug(f"Talking head: large face ratio {face_ratio:.3f}")
                
                # 2. ANY face in center region - EXTREMELY strict (most important fix)
                if face_ratio > 0.02:  # Lowered from 0.03 - catch even tiny centered faces
                    # MUCH more aggressive center detection
                    if (face_center_y < height * 0.8 and  # Upper 80% of frame (was 75%)
                        abs(face_center_x - center_x) < width * 0.4):  # Center 80% horizontally (was 70%)
                        is_talking_head = True
                        logger.debug(f"Talking head: centered face at ({face_center_x:.0f},{face_center_y:.0f}), ratio={face_ratio:.3f}")
                
                # 3. ULTRA-STRICT center detection for ANY face
                if face_ratio > 0.015:  # Even tinier faces
                    # Dead center detection - very strict
                    if (face_center_y < height * 0.7 and  # Upper 70% 
                        abs(face_center_x - center_x) < width * 0.3):  # Center 60% horizontally
                        is_talking_head = True
                        logger.debug(f"Talking head: dead center face at ({face_center_x:.0f},{face_center_y:.0f})")
                
                # 4. Face dominates attention (much more aggressive)
                if face_ratio > 0.02:  # Lowered threshold
                    non_face_region = np.copy(gray)
                    non_face_region[y:y+h, x:x+w] = 0  # Zero out face region
                    
                    edges_non_face = cv2.Canny(non_face_region, 50, 150)
                    non_face_edge_density = np.sum(edges_non_face > 0) / total_area
                    
                    # Much stricter - need significant content outside face
                    if non_face_edge_density < 0.15:  # Increased from 0.12
                        is_talking_head = True
                        logger.debug(f"Talking head: low content density {non_face_edge_density:.3f}")
                
                # 5. Portrait-like composition (more aggressive)
                face_height_ratio = h / height
                face_width_ratio = w / width
                if (face_height_ratio > 0.18 and face_ratio > 0.03) or \
                   (face_width_ratio > 0.18 and face_ratio > 0.03):
                    is_talking_head = True
                    logger.debug(f"Talking head: portrait composition h_ratio={face_height_ratio:.3f}, w_ratio={face_width_ratio:.3f}")
                
                # 6. Single face penalty - very aggressive for any solo face
                if len(faces) == 1:  # Single face is much more likely to be talking head
                    if face_ratio > 0.015:  # Any visible solo face (lowered from 0.02)
                        is_talking_head = True
                        logger.debug(f"Talking head: solo face detected, ratio={face_ratio:.3f}")
                
                # 7. Upper body detection (torso + face = definitely talking head)
                # Check for skin-colored regions below the face
                if face_ratio > 0.03:
                    # Look for skin-like colors in region below face
                    torso_region = image[y+h:min(height, y+h+80), max(0, x-20):min(width, x+w+20)]
                    if torso_region.size > 0:
                        hsv_torso = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
                        skin_mask = cv2.inRange(hsv_torso, (0, 30, 60), (20, 150, 255))
                        skin_ratio = np.sum(skin_mask > 0) / torso_region[:,:,0].size
                        if skin_ratio > 0.2:  # Significant skin-colored area below face
                            is_talking_head = True
                            logger.debug(f"Talking head: upper body detected, skin_ratio={skin_ratio:.3f}")
                
                # 6. Screen sharing check - detect if significant screen content exists
                screen_content_detected = False
                screen_content_score = 0
                
                # Check multiple regions for screen content
                # Right side (common layout)
                right_region = gray[:, int(width * 0.3):]  # Right 70% of screen
                right_edges = cv2.Canny(right_region, 50, 150)
                right_edge_density = np.sum(right_edges > 0) / right_region.size
                
                # Main central area (full screen content)
                center_region = gray[int(height * 0.1):int(height * 0.9), int(width * 0.2):int(width * 0.8)]
                if center_region.size > 0:
                    center_edges = cv2.Canny(center_region, 50, 150)
                    center_edge_density = np.sum(center_edges > 0) / center_region.size
                else:
                    center_edge_density = 0
                
                # Bottom region (common for code editors, terminals)
                bottom_region = gray[int(height * 0.4):, :]
                bottom_edges = cv2.Canny(bottom_region, 50, 150)
                bottom_edge_density = np.sum(bottom_edges > 0) / bottom_region.size
                
                # Calculate overall screen content score
                if right_edge_density > 0.12:
                    screen_content_score += 30
                if center_edge_density > 0.15:
                    screen_content_score += 40
                if bottom_edge_density > 0.1:
                    screen_content_score += 20
                
                # Additional checks for browser/code patterns
                # Look for rectangular patterns (windows, text blocks)
                contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_rectangles = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 2000:  # Large UI elements
                        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                        if len(approx) == 4:  # Rectangle
                            large_rectangles += 1
                
                if large_rectangles > 2:
                    screen_content_score += 25
                
                # Determine if this is legitimate screen sharing
                if screen_content_score > 50:
                    screen_content_detected = True
                    logger.debug(f"Strong screen content detected, score: {screen_content_score}")
                elif screen_content_score > 30:
                    screen_content_detected = True
                    logger.debug(f"Moderate screen content detected, score: {screen_content_score}")
                
                # Apply penalties based on context
                if is_talking_head:
                    if screen_content_detected and face_ratio < 0.12:
                        # Significantly reduced penalty for legitimate screen sharing with small face
                        if screen_content_score > 60:
                            talking_head_penalty += 150  # Light penalty for strong screen content
                        else:
                            talking_head_penalty += 300  # Medium penalty for moderate screen content
                        logger.debug(f"Reduced talking head penalty: screen_score={screen_content_score}, face_ratio={face_ratio:.3f}")
                    else:
                        # MASSIVE penalty for clear talking heads without screen content
                        talking_head_penalty += 800  # Unchanged massive penalty
                        logger.debug(f"MASSIVE talking head penalty: face_ratio={face_ratio:.3f}")
                elif face_ratio > 0.02:
                    # Small penalty for any face, but much reduced if screen sharing
                    base_penalty = face_ratio * 50
                    if screen_content_detected and face_ratio < 0.08:
                        base_penalty *= 0.3  # Reduce penalty for small faces with screen content
                    talking_head_penalty += base_penalty
                    logger.debug(f"Small face penalty: {base_penalty:.1f}, screen_detected={screen_content_detected}")
        
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
        
        # 9. Web browser detection bonus
        browser_bonus = self._detect_web_browser(image, gray)
        score += browser_bonus
        
        # 10. Code editor detection bonus
        code_editor_bonus = self._detect_code_editor(image, gray)
        score += code_editor_bonus
        
        # 11. Mouse cursor detection bonus
        cursor_bonus = self._detect_mouse_cursor(image, gray)
        score += cursor_bonus
        
        # 12. Selected text detection bonus
        selected_text_bonus = self._detect_selected_text(image, gray)
        score += selected_text_bonus
        
        # 13. Document/table detection bonus
        document_table_bonus = self._detect_document_table(image, gray)
        score += document_table_bonus
        
        # 14. Smart home device detection bonus
        smart_device_bonus = self._detect_smart_home_devices(image, gray)
        score += smart_device_bonus
        
        # 15. Small computer/tech device detection bonus
        tech_device_bonus = self._detect_tech_devices(image, gray)
        score += tech_device_bonus
        
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
    
    def _detect_web_browser(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect web browser interfaces and award bonus points."""
        browser_score = 0
        height, width = gray.shape
        
        # 1. Browser UI elements detection
        # Look for top navigation bars (address bars, tabs)
        top_strip = gray[:int(height * 0.15), :]  # Top 15% of screen
        
        # Look for horizontal lines in top area (typical of browser tabs/navigation)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(cv2.Canny(top_strip, 50, 150), cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_density = np.sum(horizontal_lines > 0) / top_strip.size
        
        if horizontal_density > 0.01:  # Strong horizontal structure in top area
            browser_score += 40
            logger.debug("Detected browser navigation bar structure")
        
        # 2. URL bar detection - look for long rectangular regions in top area
        contours, _ = cv2.findContours(cv2.Canny(top_strip, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # URL bars are typically long and horizontal
            if aspect_ratio > 8 and area > 1000:  # Long horizontal rectangle
                browser_score += 35
                logger.debug("Detected URL bar shape")
                break
        
        # 3. Browser button detection (back, forward, refresh, etc.)
        # Look for small square/circular regions in top-left
        top_left = gray[:int(height * 0.1), :int(width * 0.2)]
        circles = cv2.HoughCircles(top_left, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                                 param1=50, param2=20, minRadius=8, maxRadius=25)
        
        if circles is not None and len(circles[0]) >= 2:  # Multiple small circular buttons
            browser_score += 30
            logger.debug("Detected browser navigation buttons")
        
        # 4. Scrollbar detection (right edge vertical lines)
        right_edge = gray[:, int(width * 0.95):]  # Rightmost 5%
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(cv2.Canny(right_edge, 50, 150), cv2.MORPH_OPEN, vertical_kernel)
        
        if np.sum(vertical_lines > 0) > 50:  # Significant vertical structure
            browser_score += 25
            logger.debug("Detected browser scrollbar")
        
        # 5. Web content indicators
        # Look for typical web colors and layouts
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Common web colors: blues (links), whites (backgrounds), standard web palette
        blue_links = cv2.inRange(hsv, (100, 100, 50), (130, 255, 255))  # Blue links
        white_bg = cv2.inRange(hsv, (0, 0, 220), (180, 30, 255))       # White backgrounds
        
        blue_ratio = np.sum(blue_links > 0) / (height * width)
        white_ratio = np.sum(white_bg > 0) / (height * width)
        
        if blue_ratio > 0.05:  # Significant blue (links)
            browser_score += 20
        if white_ratio > 0.3:  # Significant white background
            browser_score += 15
        
        # 6. Text density suggesting web content
        # Web pages typically have moderate text density
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        text_regions = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, text_kernel)
        text_density = np.sum(text_regions != binary) / (height * width)
        
        if 0.1 < text_density < 0.4:  # Moderate text density typical of web pages
            browser_score += 20
        
        # Cap browser bonus
        browser_score = min(browser_score, 150)
        
        if browser_score > 30:
            logger.debug(f"Web browser detection bonus: +{browser_score}")
        
        return browser_score
    
    def _detect_code_editor(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect code editor interfaces and award bonus points."""
        code_editor_score = 0
        height, width = gray.shape
        
        # 1. Dark theme detection (very common in code editors)
        mean_brightness = np.mean(gray)
        if mean_brightness < 60:  # Very dark background
            # Look for syntax highlighting colors on dark background
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Common syntax colors: green (comments), blue (keywords), orange/yellow (strings)
            green_syntax = cv2.inRange(hsv, (40, 50, 100), (80, 255, 255))   # Green text
            blue_syntax = cv2.inRange(hsv, (100, 50, 100), (130, 255, 255))  # Blue keywords
            yellow_syntax = cv2.inRange(hsv, (15, 50, 100), (35, 255, 255))  # Yellow/orange strings
            
            syntax_colors = np.sum(green_syntax > 0) + np.sum(blue_syntax > 0) + np.sum(yellow_syntax > 0)
            syntax_ratio = syntax_colors / (height * width)
            
            if syntax_ratio > 0.1:  # Significant syntax highlighting
                code_editor_score += 60
                logger.debug("Detected dark theme with syntax highlighting")
        
        # 2. Line number detection (left margin with numbers)
        left_margin = gray[:, :int(width * 0.1)]  # Leftmost 10%
        
        # Look for regular patterns that could be line numbers
        # Line numbers create regular vertical patterns
        vertical_patterns = 0
        for i in range(0, left_margin.shape[0] - 20, 20):  # Every 20 pixels
            row_section = left_margin[i:i+20, :]
            if np.std(row_section) > 20:  # Some variation (text)
                vertical_patterns += 1
        
        if vertical_patterns > 5:  # Regular patterns suggesting line numbers
            code_editor_score += 45
            logger.debug("Detected line number patterns")
        
        # 3. Indentation detection (code structure)
        # Look for regular indentation patterns
        indentation_score = 0
        for row in range(int(height * 0.2), int(height * 0.8), 10):  # Sample rows
            if row < height:
                line = gray[row, :]
                
                # Find the first non-background pixel (start of text)
                line_binary = line < 200  # Assuming dark text on light or text on dark
                first_text = np.argmax(line_binary) if np.any(line_binary) else 0
                
                # Common indentation levels (multiples of 2, 4, or 8 spaces/chars)
                if first_text > 0 and first_text % 4 < 2:  # Regular indentation
                    indentation_score += 1
        
        if indentation_score > 8:  # Multiple indented lines
            code_editor_score += 40
            logger.debug("Detected code indentation patterns")
        
        # 4. Monospace font detection (characteristic of code)
        # Monospace fonts have very regular character spacing
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for regular vertical patterns (monospace character boundaries)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Sample horizontal lines to check for regular spacing
        regular_spacing = 0
        for row in range(int(height * 0.3), int(height * 0.7), 15):
            if row < height:
                line_edges = vertical_edges[row, :]
                edge_positions = np.where(line_edges > 0)[0]
                
                if len(edge_positions) > 5:
                    # Check for regular spacing
                    spacings = np.diff(edge_positions)
                    spacing_std = np.std(spacings) if len(spacings) > 1 else float('inf')
                    if spacing_std < 3:  # Very regular spacing
                        regular_spacing += 1
        
        if regular_spacing > 3:
            code_editor_score += 35
            logger.debug("Detected monospace font patterns")
        
        # 5. Code structure detection (braces, parentheses, semicolons)
        # Look for specific patterns common in code
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Count small isolated features that could be punctuation
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_features = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 50:  # Small features like punctuation
                small_features += 1
        
        if small_features > 50:  # Lots of small features (punctuation, symbols)
            code_editor_score += 30
            logger.debug("Detected code punctuation patterns")
        
        # 6. File explorer/tree detection (common in IDEs)
        # Look for tree-like structures on the left side
        left_panel = gray[:, :int(width * 0.25)]  # Left 25%
        
        # Look for indented hierarchical structure
        tree_structure = 0
        for row in range(int(height * 0.1), int(height * 0.9), 20):
            if row < left_panel.shape[0]:
                line = left_panel[row, :]
                
                # Look for stepped patterns (folder/file hierarchy)
                non_bg = np.where(line < 230)[0]  # Non-background pixels
                if len(non_bg) > 0:
                    indent_level = non_bg[0]
                    if 10 < indent_level < left_panel.shape[1] * 0.8:
                        tree_structure += 1
        
        if tree_structure > 5:
            code_editor_score += 25
            logger.debug("Detected file tree structure")
        
        # Cap code editor bonus
        code_editor_score = min(code_editor_score, 180)
        
        if code_editor_score > 30:
            logger.debug(f"Code editor detection bonus: +{code_editor_score}")
        
        return code_editor_score
    
    def _detect_mouse_cursor(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect mouse cursor in the frame and award bonus points."""
        cursor_score = 0
        height, width = gray.shape
        
        # 1. Look for cursor shapes using template matching
        # Common cursor types: arrow, hand, text beam, etc.
        
        # Arrow cursor detection (most common)
        # Look for small triangular or arrow-like shapes
        contours, _ = cv2.findContours(cv2.Canny(gray, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Cursor-sized area
                # Check if it's roughly triangular (arrow cursor)
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 3:  # Triangle
                    # Check if it's small and positioned like a cursor
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Arrow cursors are roughly square/triangular
                    if 0.5 < aspect_ratio < 2.0:
                        cursor_score += 50
                        logger.debug(f"Detected arrow cursor at ({x}, {y})")
                        break
        
        # 2. Look for cursor shadows/outlines
        # Cursors often have white interior with black outline
        
        # Find small white regions with dark borders
        _, white_thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in white_contours:
            area = cv2.contourArea(contour)
            if 20 < area < 300:  # Small white region
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check for dark outline around white region
                padding = 3
                region_x1 = max(0, x - padding)
                region_y1 = max(0, y - padding)
                region_x2 = min(width, x + w + padding)
                region_y2 = min(height, y + h + padding)
                
                region = gray[region_y1:region_y2, region_x1:region_x2]
                if region.size > 0:
                    # Check if surrounded by darker pixels
                    mean_surrounding = np.mean(region)
                    mean_interior = np.mean(gray[y:y+h, x:x+w])
                    
                    if mean_interior > mean_surrounding + 50:  # White with dark outline
                        cursor_score += 45
                        logger.debug(f"Detected cursor with outline at ({x}, {y})")
                        break
        
        # 3. Hand cursor detection (pointing hand)
        # Look for hand-like shapes - more complex contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 800:  # Hand cursor size
                # Hand cursors have more complex shapes
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                
                # Hand cursors typically have 5-8 vertices (fingers)
                if 5 <= len(approx) <= 8:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Hand cursors are roughly square but can be slightly elongated
                    if 0.6 < aspect_ratio < 1.8:
                        cursor_score += 40
                        logger.debug(f"Detected hand cursor at ({x}, {y})")
                        break
        
        # 4. Text cursor (I-beam) detection
        # Look for thin vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(cv2.Canny(gray, 50, 150), cv2.MORPH_OPEN, vertical_kernel)
        
        # Find isolated vertical lines that could be text cursors
        line_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in line_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 15 and w < 5:  # Tall and thin
                cursor_score += 35
                logger.debug(f"Detected text cursor at ({x}, {y})")
                break
        
        # 5. Cursor highlighting detection
        # Some applications highlight cursors with special effects
        
        # Look for small bright spots that could be cursor highlights
        _, bright_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in bright_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 200:  # Small bright spot
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's roughly circular (cursor highlight)
                aspect_ratio = w / h if h > 0 else 0
                if 0.8 < aspect_ratio < 1.2:  # Nearly circular
                    cursor_score += 25
                    logger.debug(f"Detected cursor highlight at ({x}, {y})")
                    break
        
        # 6. Color-based cursor detection
        # Some cursors use distinctive colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Look for pure white or pure black small regions (common cursor colors)
        pure_white = cv2.inRange(hsv, (0, 0, 250), (180, 10, 255))
        pure_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        
        for mask in [pure_white, pure_black]:
            cursor_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in cursor_contours:
                area = cv2.contourArea(contour)
                if 30 < area < 400:  # Cursor-sized pure color region
                    cursor_score += 20
                    break
        
        # Cap cursor bonus
        cursor_score = min(cursor_score, 80)
        
        if cursor_score > 0:
            logger.debug(f"Mouse cursor detection bonus: +{cursor_score}")
        
        return cursor_score
    
    def _detect_selected_text(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect selected/highlighted text in the frame and award significant bonus points."""
        selected_text_score = 0
        height, width = gray.shape
        
        # 1. Blue selection highlighting (most common)
        # Look for blue/cyan highlighting backgrounds typical of text selection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Common selection colors
        blue_selection = cv2.inRange(hsv, (90, 100, 100), (130, 255, 255))   # Blue highlights
        cyan_selection = cv2.inRange(hsv, (80, 100, 100), (100, 255, 255))   # Cyan highlights
        purple_selection = cv2.inRange(hsv, (130, 50, 100), (160, 255, 255)) # Purple highlights
        
        # Combine all selection color masks
        selection_mask = cv2.bitwise_or(blue_selection, cyan_selection)
        selection_mask = cv2.bitwise_or(selection_mask, purple_selection)
        
        # Look for rectangular regions with selection colors
        selection_contours, _ = cv2.findContours(selection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_selections = 0
        total_selection_area = 0
        
        for contour in selection_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Reasonable size for text selection
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text selections are typically wider than they are tall
                if aspect_ratio > 2 and area > 500:  # Wide selection (multiple words/lines)
                    selected_text_score += 60
                    significant_selections += 1
                    total_selection_area += area
                    logger.debug(f"Detected large text selection at ({x}, {y}), area: {area}")
                elif aspect_ratio > 1.5 and area > 200:  # Medium selection (word/phrase)
                    selected_text_score += 40
                    significant_selections += 1
                    total_selection_area += area
                    logger.debug(f"Detected medium text selection at ({x}, {y}), area: {area}")
                elif area > 100:  # Small selection (partial word)
                    selected_text_score += 20
                    significant_selections += 1
                    total_selection_area += area
        
        # 2. Inverted text detection (white text on dark background, common in dark themes)
        # This often indicates selected text in dark mode editors
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Look for white text blocks on dark backgrounds
        white_text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        white_text_regions = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, white_text_kernel)
        
        # Find white text blocks that could be selections
        white_contours, _ = cv2.findContours(white_text_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in white_contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Substantial white text area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if this is in a dark region (indicating selection in dark theme)
                region = gray[max(0, y-5):min(height, y+h+5), max(0, x-5):min(width, x+w+5)]
                if region.size > 0:
                    background_brightness = np.mean(region)
                    
                    # Dark background with bright text suggests selection
                    if background_brightness < 80 and aspect_ratio > 2:
                        selected_text_score += 50
                        logger.debug(f"Detected dark theme text selection at ({x}, {y})")
        
        # 3. Look for selection highlighting patterns
        # Many applications use specific patterns for text selection
        
        # Edge-based selection detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for rectangular outlines that could be selection boxes
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, rect_kernel)
        
        rect_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, rect_kernel_v)
        
        # Combine to find rectangular selection boxes
        selection_boxes = cv2.bitwise_or(horizontal_edges, vertical_edges)
        box_contours, _ = cv2.findContours(selection_boxes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in box_contours:
            area = cv2.contourArea(contour)
            if area > 400:  # Large enough to be text selection box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Selection boxes are typically wider than tall
                if aspect_ratio > 3:
                    selected_text_score += 35
                    logger.debug(f"Detected selection box outline at ({x}, {y})")
        
        # 4. High contrast region detection (another selection indicator)
        # Selected text often has very high contrast
        
        # Calculate local contrast
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_contrast = np.sqrt(np.maximum(local_variance, 0))
        
        # Find high contrast regions
        high_contrast_mask = (local_contrast > 40).astype(np.uint8) * 255
        contrast_contours, _ = cv2.findContours(high_contrast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contrast_contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Significant high-contrast area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if aspect_ratio > 2:  # Text-like shape
                    selected_text_score += 25
        
        # 5. Multiple selection bonus
        # If we found multiple selections, it's very likely to be valuable content
        if significant_selections > 1:
            multi_selection_bonus = min(significant_selections * 30, 100)
            selected_text_score += multi_selection_bonus
            logger.debug(f"Multiple selections bonus: +{multi_selection_bonus}")
        
        # 6. Large selection area bonus
        # Bigger selections are typically more important
        selection_ratio = total_selection_area / (height * width)
        if selection_ratio > 0.05:  # More than 5% of screen is selected
            area_bonus = min(selection_ratio * 200, 80)
            selected_text_score += area_bonus
            logger.debug(f"Large selection area bonus: +{area_bonus:.1f}")
        
        # 7. Code selection bonus (if also detected as code editor)
        # Selected code is extremely valuable
        code_editor_score = self._detect_code_editor(image, gray)
        if code_editor_score > 50 and selected_text_score > 30:
            code_selection_bonus = 60
            selected_text_score += code_selection_bonus
            logger.debug(f"Code selection bonus: +{code_selection_bonus}")
        
        # 8. Browser selection bonus (selected text in web pages)
        browser_score = self._detect_web_browser(image, gray)
        if browser_score > 50 and selected_text_score > 20:
            web_selection_bonus = 40
            selected_text_score += web_selection_bonus
            logger.debug(f"Web text selection bonus: +{web_selection_bonus}")
        
        # Cap the selection bonus but make it very high since selected text is extremely valuable
        selected_text_score = min(selected_text_score, 250)
        
        if selected_text_score > 0:
            logger.debug(f"Selected text detection bonus: +{selected_text_score}")
        
        return selected_text_score
    
    def _detect_document_table(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect documents and tables in the frame and award bonus points."""
        document_score = 0
        height, width = gray.shape
        
        # 1. Table structure detection
        # Look for grid-like patterns with horizontal and vertical lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal lines (table rows)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        h_line_count = cv2.countNonZero(horizontal_lines)
        
        # Detect vertical lines (table columns)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        v_line_count = cv2.countNonZero(vertical_lines)
        
        # Combine to find table structure
        table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Check for intersections (strong indicator of tables)
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        intersection_count = cv2.countNonZero(intersections)
        
        if intersection_count > 20:  # Multiple intersections suggest table
            document_score += 80
            logger.debug(f"Table structure detected: {intersection_count} intersections")
        elif h_line_count > 500 and v_line_count > 500:  # Strong grid pattern
            document_score += 60
            logger.debug(f"Grid pattern detected: h_lines={h_line_count}, v_lines={v_line_count}")
        
        # 2. Document text patterns
        # Look for regular text layout typical of documents
        
        # Find text regions using MSER (Maximally Stable Extremal Regions)
        # This is good for detecting text blocks
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Look for rectangular text blocks
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_blocks = 0
        regular_spacing = 0
        
        # Analyze contours for document-like text layout
        y_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Text-sized regions
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text blocks are typically wider than tall
                if 2 < aspect_ratio < 20:  # Text line proportions
                    text_blocks += 1
                    y_positions.append(y)
        
        # Check for regular vertical spacing (typical of documents)
        if len(y_positions) > 5:
            y_positions.sort()
            spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            if len(spacings) > 0:
                spacing_std = np.std(spacings)
                if spacing_std < 10:  # Very regular line spacing
                    regular_spacing = len(spacings)
                    document_score += min(regular_spacing * 8, 60)
                    logger.debug(f"Regular text spacing detected: {regular_spacing} lines")
        
        if text_blocks > 10:
            document_score += 40
            logger.debug(f"Multiple text blocks detected: {text_blocks}")
        
        # 3. White background with organized content (typical of documents)
        # Check for predominantly white background with structured content
        white_pixels = np.sum(gray > 220)
        white_ratio = white_pixels / (height * width)
        
        if white_ratio > 0.6:  # Predominantly white background
            # Check for organized content structure
            edge_density = np.sum(edges > 0) / (height * width)
            if 0.05 < edge_density < 0.3:  # Moderate, organized content
                document_score += 35
                logger.debug(f"Document-like white background with organized content")
        
        # 4. Spreadsheet detection
        # Look for cell-like structures with numbers/data
        
        # Detect small rectangular regions in a grid
        small_rects = 0
        grid_pattern = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:  # Cell-sized regions
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Cells are roughly square or slightly rectangular
                if 0.5 < aspect_ratio < 3:
                    small_rects += 1
                    
                    # Check if it's part of a grid (aligned with others)
                    aligned_count = 0
                    for other_contour in contours:
                        if other_contour is contour:
                            continue
                        other_area = cv2.contourArea(other_contour)
                        if 50 < other_area < 1000:
                            ox, oy, ow, oh = cv2.boundingRect(other_contour)
                            # Check for alignment (same row or column)
                            if abs(y - oy) < 5 or abs(x - ox) < 5:
                                aligned_count += 1
                    
                    if aligned_count > 2:
                        grid_pattern += 1
        
        if grid_pattern > 5:
            document_score += 50
            logger.debug(f"Spreadsheet grid pattern detected: {grid_pattern} aligned cells")
        
        # 5. Document headers and formatting
        # Look for bold text, headers, and formatted content
        
        # Find thick horizontal lines (could be headers or separators)
        thick_h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        thick_h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, thick_h_kernel)
        thick_h_count = cv2.countNonZero(thick_h_lines)
        
        if thick_h_count > 100:
            document_score += 25
            logger.debug(f"Document headers/separators detected")
        
        # 6. PDF/document viewer detection
        # Look for typical document viewer UI elements
        
        # Check for scroll bars, page indicators
        right_edge = gray[:, int(width * 0.95):]
        if right_edge.size > 0:
            right_edges = cv2.Canny(right_edge, 50, 150)
            scrollbar_pattern = cv2.countNonZero(right_edges)
            if scrollbar_pattern > 50:
                document_score += 20
        
        # Look for page-like aspect ratios in content area
        content_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Large content areas
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                # Document pages are typically taller than wide
                if 1.2 < aspect_ratio < 1.6:  # Letter/A4 proportions
                    content_regions.append((x, y, w, h))
        
        if len(content_regions) > 0:
            document_score += 30
            logger.debug(f"Document page proportions detected")
        
        # Cap document bonus
        document_score = min(document_score, 200)
        
        if document_score > 30:
            logger.debug(f"Document/table detection bonus: +{document_score}")
        
        return document_score
    
    def _detect_smart_home_devices(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect smart home devices in the frame and award bonus points."""
        device_score = 0
        height, width = gray.shape
        
        # 1. LED indicators (very common on smart devices)
        # Look for small, bright circular spots
        _, bright_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        led_indicators = 0
        for contour in bright_contours:
            area = cv2.contourArea(contour)
            if 8 < area < 150:  # LED-sized bright spots
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.4:  # Roughly circular
                    led_indicators += 1
        
        if led_indicators > 1:
            device_score += min(led_indicators * 20, 60)
            logger.debug(f"LED indicators detected: {led_indicators}")
        
        # 2. Small black/white rectangular devices (common smart device form factor)
        contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        device_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 8000:  # Device-sized objects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if it's rectangular (typical device shape)
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangle
                    # Smart devices are often small, rectangular, with moderate aspect ratios
                    if 0.3 < aspect_ratio < 4:
                        # Check color - many smart devices are black, white, or gray
                        device_region = image[y:y+h, x:x+w]
                        if device_region.size > 0:
                            mean_color = np.mean(device_region, axis=(0,1))
                            # Check for neutral colors (black, white, gray)
                            color_variance = np.var(mean_color)
                            if color_variance < 500:  # Low color variance (neutral)
                                device_shapes += 1
        
        if device_shapes > 0:
            device_score += min(device_shapes * 25, 75)
            logger.debug(f"Smart device shapes detected: {device_shapes}")
        
        # 3. Smart speaker detection (cylindrical shapes)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                 param1=50, param2=30, minRadius=20, maxRadius=100)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            smart_speakers = 0
            for (x, y, r) in circles:
                # Check if it's in a reasonable position (on table, shelf, etc.)
                if y > height * 0.3:  # Not in upper sky area
                    # Check surrounding area for table/surface
                    surrounding = gray[max(0, y-10):min(height, y+r+20), 
                                     max(0, x-r-10):min(width, x+r+10)]
                    if surrounding.size > 0:
                        # Look for horizontal surface beneath
                        bottom_edge = surrounding[-10:, :] if surrounding.shape[0] > 10 else surrounding
                        if np.mean(bottom_edge) > 100:  # Lighter surface
                            smart_speakers += 1
            
            if smart_speakers > 0:
                device_score += min(smart_speakers * 30, 60)
                logger.debug(f"Smart speakers detected: {smart_speakers}")
        
        # 4. Smart display detection (rectangular screens with bezels)
        screen_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 20000:  # Screen-sized regions
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Smart displays often have wide aspect ratios
                if 1.2 < aspect_ratio < 2.5:
                    # Check for dark bezel around bright screen
                    screen_region = gray[y:y+h, x:x+w]
                    if screen_region.size > 0:
                        # Look for bright center with dark edges (typical of screens)
                        center_brightness = np.mean(screen_region[h//4:3*h//4, w//4:3*w//4])
                        edge_brightness = np.mean(np.concatenate([
                            screen_region[:h//8, :].flatten(),
                            screen_region[-h//8:, :].flatten(),
                            screen_region[:, :w//8].flatten(),
                            screen_region[:, -w//8:].flatten()
                        ]))
                        
                        if center_brightness > edge_brightness + 30:
                            screen_regions.append((x, y, w, h))
        
        if len(screen_regions) > 0:
            device_score += min(len(screen_regions) * 40, 80)
            logger.debug(f"Smart displays detected: {len(screen_regions)}")
        
        # 5. Smart switch/outlet detection (wall-mounted rectangular devices)
        # Look for small rectangular objects that could be switches
        wall_devices = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Switch/outlet sized
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Switches are typically wider than tall or square
                if 0.8 < aspect_ratio < 2.5:
                    # Check if it's positioned like a wall device
                    if y < height * 0.8:  # Not on floor
                        wall_devices += 1
        
        if wall_devices > 2:
            device_score += 30
            logger.debug(f"Wall-mounted devices detected: {wall_devices}")
        
        # 6. Color-based device detection (look for typical smart device colors)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Common smart device colors
        black_devices = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        white_devices = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        gray_devices = cv2.inRange(hsv, (0, 0, 50), (180, 30, 200))
        
        # Look for compact regions of these colors
        for mask in [black_devices, white_devices, gray_devices]:
            device_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            compact_devices = 0
            for contour in device_contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:  # Compact device size
                    compact_devices += 1
            
            if compact_devices > 0:
                device_score += min(compact_devices * 10, 30)
        
        # Cap smart device bonus
        device_score = min(device_score, 150)
        
        if device_score > 20:
            logger.debug(f"Smart home device detection bonus: +{device_score}")
        
        return device_score
    
    def _detect_tech_devices(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect small computers and tech devices in the frame and award bonus points."""
        tech_score = 0
        height, width = gray.shape
        
        # 1. Raspberry Pi / small computer detection
        # Look for small rectangular boards with components
        contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circuit_boards = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 5000:  # Small board size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Circuit boards are typically rectangular
                if 1.2 < aspect_ratio < 2.5:  # Typical board proportions
                    # Check for green color (PCB green)
                    board_region = image[y:y+h, x:x+w]
                    if board_region.size > 0:
                        hsv_board = cv2.cvtColor(board_region, cv2.COLOR_BGR2HSV)
                        green_pcb = cv2.inRange(hsv_board, (35, 40, 40), (85, 255, 255))
                        green_ratio = np.sum(green_pcb > 0) / board_region[:,:,0].size
                        
                        if green_ratio > 0.3:  # Significant green (PCB)
                            circuit_boards += 1
                            logger.debug(f"Circuit board detected at ({x}, {y})")
        
        if circuit_boards > 0:
            tech_score += min(circuit_boards * 50, 100)
            logger.debug(f"Circuit boards detected: {circuit_boards}")
        
        # 2. Small computer cases (mini PCs, NUCs, etc.)
        small_computers = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 15000:  # Small computer case size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Computer cases are often cube-like or slightly rectangular
                if 0.7 < aspect_ratio < 2:
                    # Check for typical computer case colors (black, silver, white)
                    case_region = image[y:y+h, x:x+w]
                    if case_region.size > 0:
                        mean_color = np.mean(case_region, axis=(0,1))
                        # Check for neutral/metallic colors
                        color_variance = np.var(mean_color)
                        brightness = np.mean(mean_color)
                        
                        # Dark cases or bright metallic cases
                        if (brightness < 80 and color_variance < 300) or \
                           (brightness > 180 and color_variance < 200):
                            small_computers += 1
        
        if small_computers > 0:
            tech_score += min(small_computers * 40, 80)
            logger.debug(f"Small computers detected: {small_computers}")
        
        # 3. USB devices and dongles
        # Look for very small rectangular objects that could be USB devices
        usb_devices = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 300:  # USB device size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # USB devices are often small rectangles
                if 1.5 < aspect_ratio < 4:  # USB stick proportions
                    usb_devices += 1
        
        if usb_devices > 1:
            tech_score += min(usb_devices * 15, 45)
            logger.debug(f"USB devices detected: {usb_devices}")
        
        # 4. Development boards (Arduino, ESP32, etc.)
        # Look for small boards with visible components
        dev_boards = 0
        
        # Look for small rectangular regions with lots of detail (components)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 3000:  # Dev board size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check component density
                board_region = gray[y:y+h, x:x+w]
                if board_region.size > 0:
                    edges_in_board = cv2.Canny(board_region, 50, 150)
                    component_density = np.sum(edges_in_board > 0) / board_region.size
                    
                    # High edge density suggests lots of small components
                    if component_density > 0.2:
                        dev_boards += 1
        
        if dev_boards > 0:
            tech_score += min(dev_boards * 35, 70)
            logger.debug(f"Development boards detected: {dev_boards}")
        
        # 5. Cables and connectors
        # Look for linear features that could be cables
        
        # Detect thin lines (cables)
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        cable_lines = cv2.morphologyEx(cv2.Canny(gray, 50, 150), cv2.MORPH_OPEN, line_kernel)
        
        # Also check for curved cables using contour analysis
        cable_contours, _ = cv2.findContours(cable_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cables = 0
        
        for contour in cable_contours:
            arc_length = cv2.arcLength(contour, False)
            area = cv2.contourArea(contour)
            
            # Cables have high perimeter to area ratio
            if area > 0 and arc_length / (area + 1) > 2:  # Thin, long objects
                if arc_length > 50:  # Reasonable cable length
                    cables += 1
        
        if cables > 2:
            tech_score += 25
            logger.debug(f"Cables detected: {cables}")
        
        # 6. Small displays and screens
        # Look for rectangular regions that could be small tech screens
        small_displays = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 8000:  # Small display size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Displays often have specific aspect ratios
                if 1.3 < aspect_ratio < 2.2:  # Common display ratios
                    # Check for dark border with bright center (typical of displays)
                    display_region = gray[y:y+h, x:x+w]
                    if display_region.size > 0:
                        border_width = min(w, h) // 8
                        if border_width > 0:
                            center = display_region[border_width:-border_width, border_width:-border_width]
                            border = np.concatenate([
                                display_region[:border_width, :].flatten(),
                                display_region[-border_width:, :].flatten(),
                                display_region[:, :border_width].flatten(),
                                display_region[:, -border_width:].flatten()
                            ])
                            
                            if center.size > 0 and border.size > 0:
                                if np.mean(center) > np.mean(border) + 20:
                                    small_displays += 1
        
        if small_displays > 0:
            tech_score += min(small_displays * 30, 60)
            logger.debug(f"Small displays detected: {small_displays}")
        
        # 7. Tech device labels and branding
        # Look for small text that could be device labels
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        device_labels = 0
        for contour in text_contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Small text size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Labels are typically wider than tall
                if 2 < aspect_ratio < 10:
                    device_labels += 1
        
        if device_labels > 5:
            tech_score += 20
            logger.debug(f"Device labels detected: {device_labels}")
        
        # Cap tech device bonus
        tech_score = min(tech_score, 180)
        
        if tech_score > 25:
            logger.debug(f"Tech device detection bonus: +{tech_score}")
        
        return tech_score
    
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