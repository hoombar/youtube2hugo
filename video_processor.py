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
        
    def extract_frames(self, video_path: str, output_dir: str) -> List[Dict]:
        """Extract frames from video at specified intervals."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get video info
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(video_info['duration'])
        fps = eval(video_info['r_frame_rate'])
        
        logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}")
        
        extracted_frames = []
        interval = self.config.get('frame_sample_interval', 15)
        
        # Sample frames at intervals
        for timestamp in np.arange(0, duration, interval):
            frame_path = os.path.join(output_dir, f"frame_{timestamp:.1f}s.jpg")
            
            # Extract frame using ffmpeg
            try:
                (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                # Analyze frame for face content
                face_ratio = self._analyze_frame_composition(frame_path)
                
                frame_info = {
                    'timestamp': timestamp,
                    'path': frame_path,
                    'face_ratio': face_ratio,
                    'should_include': self._should_include_frame(face_ratio)
                }
                
                extracted_frames.append(frame_info)
                logger.info(f"Frame at {timestamp:.1f}s: face_ratio={face_ratio:.3f}, include={frame_info['should_include']}")
                
            except ffmpeg.Error as e:
                logger.error(f"Error extracting frame at {timestamp}s: {e}")
                continue
                
        return extracted_frames
    
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
    
    def _should_include_frame(self, face_ratio: float) -> bool:
        """Determine if frame should be included based on face ratio."""
        min_face_ratio = self.config.get('min_face_ratio', 0.4)
        max_face_ratio = self.config.get('max_face_ratio', 0.2)
        
        # Skip frames with too much face content (talking head shots)
        if face_ratio > min_face_ratio:
            return False
            
        # Prefer frames with no face or small face (visual aids)
        return True
    
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