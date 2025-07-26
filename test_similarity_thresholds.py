#!/usr/bin/env python3
"""
Test script for analyzing image_similarity_threshold impact on frame selection.

Usage: python test_similarity_thresholds.py <video_path>

This script will:
1. Extract frames at different similarity thresholds (0.1 to 1.0)
2. Save results in separate folders for comparison
3. Skip Whisper transcription and Gemini processing
4. Focus purely on frame extraction and similarity filtering
"""

import sys
import os
import shutil
import tempfile
from typing import Dict, List
import yaml
import logging

# Import our existing modules
from video_processor import VideoProcessor
from hugo_generator import HugoGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimilarityThresholdTester:
    """Test different similarity thresholds and analyze their impact."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cached_transcript = None  # Cache transcript to avoid re-generating
        self.base_config = {
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
            'whisper_model': 'base',  # Same model as real script uses
            'language': 'en',
            'image_similarity_threshold': 0.5,  # Will be overridden
            # Additional config to match the real script
            'image_max_width': 1920,
            'image_max_height': 1080,
            'image_quality': 95,
            'cleanup_temp_files': True,
            'save_transcripts': False,
            'default_whisper_model': 'base'
        }
    
    def test_all_thresholds(self) -> None:
        """Test all similarity thresholds from 0.1 to 1.0."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        logger.info(f"🎯 Starting similarity threshold analysis for: {self.video_path}")
        logger.info(f"📊 Testing thresholds: {thresholds}")
        
        results = {}
        
        for threshold in thresholds:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 TESTING THRESHOLD: {threshold}")
            logger.info(f"{'='*60}")
            
            result = self.test_single_threshold(threshold)
            results[threshold] = result
            
            logger.info(f"✅ Threshold {threshold} complete: {result['frame_count']} frames saved")
        
        # Print summary
        self._print_summary(results)
    
    def test_single_threshold(self, threshold: float) -> Dict:
        """Test a single similarity threshold using the exact same logic as the real script."""
        output_dir = f"images_{threshold}"
        
        logger.info(f"🎯 Starting test for threshold {threshold}")
        logger.info(f"🎬 Video path: {self.video_path}")
        
        # Clean up existing directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Create config with specific threshold
        config = self.base_config.copy()
        config['image_similarity_threshold'] = threshold
        
        # Get transcript first (cached after first threshold)
        if self._cached_transcript is None:
            logger.info(f"🎙️  Running Whisper transcription...")
            self._cached_transcript = self._get_whisper_transcript()
        else:
            logger.info(f"🎙️  Using cached Whisper transcript ({len(self._cached_transcript)} segments)")
        transcript_segments = self._cached_transcript

        # Create temporary directory for frame extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize processors
            video_processor = VideoProcessor(config)
            hugo_gen = HugoGenerator(config)
            
            # Extract frames WITH transcript (same as real script)
            logger.info(f"🎬 Extracting frames using transcript context...")
            frames = video_processor.extract_frames(
                self.video_path, 
                temp_dir, 
                transcript_segments  # Use real transcript!
            )
            
            logger.info(f"📸 Initial frame extraction complete: {len(frames)} frames")
            
            # MISSING STEP: Optimize images (this filters frames to only should_include=True)
            logger.info(f"🖼️  Running image optimization to filter frames...")
            optimized_frames = video_processor.optimize_images(frames)
            logger.info(f"✅ Image optimization complete: {len(optimized_frames)} frames after filtering")
            
            # Now simulate the EXACT blog post generation process using optimized frames
            selected_frames = self._simulate_blog_content_generation(optimized_frames, hugo_gen, transcript_segments)
            
            logger.info(f"✨ After blog content simulation: {len(selected_frames)} frames selected")
            
            # Copy selected frames to output directory
            copied_count = 0
            for i, frame in enumerate(selected_frames):
                src_path = frame['path']
                if os.path.exists(src_path):
                    # Create descriptive filename with timestamp and score
                    timestamp = frame['timestamp']
                    score = frame.get('score', 0)
                    ext = os.path.splitext(src_path)[1]
                    dst_filename = f"frame_{i:03d}_t{timestamp:06.1f}s_score{score:.0f}{ext}"
                    dst_path = os.path.join(output_dir, dst_filename)
                    
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    
                    logger.debug(f"📋 Copied: {dst_filename}")
        
        result = {
            'threshold': threshold,
            'frame_count': copied_count,
            'output_dir': output_dir
        }
        
        logger.info(f"📁 Results saved to: {output_dir}")
        return result
    
    def _simulate_blog_content_generation(self, frames: List[Dict], hugo_gen: HugoGenerator, transcript_segments: List[Dict]) -> List[Dict]:
        """Simulate the exact blog content generation process."""
        
        # Filter frames to only those with should_include=True (exact same logic)
        frames_with_include = [f for f in frames if f.get('should_include', False)]
        sorted_frames = sorted(frames_with_include, key=lambda x: x['timestamp'])
        logger.info(f"📍 FRAME FILTERING: Using {len(sorted_frames)} frames with should_include=True")
        logger.info(f"🔧 CONFIG CHECK: image_similarity_threshold = {hugo_gen.config.get('image_similarity_threshold', 'NOT SET')}")
        
        # Group transcript segments into paragraphs (exact same logic as hugo_generator.py)
        paragraphs = hugo_gen._group_segments_into_paragraphs(transcript_segments)
        logger.info(f"📝 CONTENT STRUCTURE: {len(paragraphs)} paragraphs from transcript")
        
        # Debug: Show paragraph boundaries
        logger.info("📋 PARAGRAPH BOUNDARIES:")
        for i, paragraph in enumerate(paragraphs):
            para_start = paragraph[0]['start_time']
            para_end = paragraph[-1]['end_time']
            logger.info(f"   Paragraph {i+1}: {para_start:.1f}s - {para_end:.1f}s")
        
        # Now simulate the exact paragraph-by-paragraph frame selection process
        selected_frames = []
        used_frames = set()
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_start = paragraph[0]['start_time']
            paragraph_end = paragraph[-1]['end_time']
            logger.info(f"📝 PARAGRAPH {i+1}: {paragraph_start:.1f}s-{paragraph_end:.1f}s")
            
            # Find all relevant frames for this paragraph (EXACT same logic as hugo_generator.py)
            relevant_frames = hugo_gen._find_relevant_frames_for_paragraph(
                paragraph, sorted_frames, used_frames
            )
            
            if relevant_frames:
                logger.info(f"  🖼️  FOUND {len(relevant_frames)} relevant frames for paragraph {i+1}: {[f'{f['timestamp']:.1f}s' for f in relevant_frames]}")
                
                # Add these frames to our selection
                for frame in relevant_frames:
                    selected_frames.append(frame)
                    used_frames.add(frame['timestamp'])
                    
                logger.info(f"  ✅ SELECTED {len(relevant_frames)} frames for paragraph {i+1}")
        
        return selected_frames
    
    def _get_whisper_transcript(self) -> List[Dict]:
        """Get transcript using Whisper (same logic as transcript_extractor.py)."""
        # Import the existing transcript extractor
        try:
            from transcript_extractor import TranscriptExtractor
        except ImportError:
            logger.error("Could not import TranscriptExtractor. Make sure transcript_extractor.py is available.")
            raise
        
        # Create transcript extractor with same config as real script
        extractor_config = {
            'whisper_model': self.base_config.get('whisper_model', 'base'),  # Use same model
            'language': self.base_config.get('language', 'en')
        }
        
        extractor = TranscriptExtractor(extractor_config)
        
        # Extract transcript segments
        logger.info(f"🎤 Extracting transcript with Whisper model '{extractor_config['whisper_model']}'...")
        logger.info(f"🎬 Video path for transcript: {self.video_path}")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        transcript_segments = extractor.extract_transcript(self.video_path)
        
        logger.info(f"📝 Whisper extracted {len(transcript_segments)} transcript segments")
        return transcript_segments
    
    
    def _print_summary(self, results: Dict) -> None:
        """Print a summary of all threshold test results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 SIMILARITY THRESHOLD ANALYSIS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Video: {os.path.basename(self.video_path)}")
        logger.info("")
        
        # Sort by threshold
        sorted_results = sorted(results.items())
        
        logger.info("Threshold | Frame Count | Output Directory")
        logger.info("-" * 45)
        
        for threshold, result in sorted_results:
            logger.info(f"{threshold:9.1f} | {result['frame_count']:11d} | {result['output_dir']}")
        
        # Analysis
        max_frames = max(r['frame_count'] for r in results.values())
        min_frames = min(r['frame_count'] for r in results.values())
        
        logger.info("")
        logger.info(f"📈 Analysis:")
        logger.info(f"   • Maximum frames: {max_frames} (lower threshold = more frames)")
        logger.info(f"   • Minimum frames: {min_frames} (higher threshold = fewer frames)")
        logger.info(f"   • Frame count range: {max_frames - min_frames}")
        
        # Find the "sweet spot" recommendations
        mid_range_frames = [r for r in results.values() if min_frames < r['frame_count'] < max_frames]
        if mid_range_frames:
            recommended_count = sorted([r['frame_count'] for r in mid_range_frames])[len(mid_range_frames)//2]
            recommended_thresholds = [t for t, r in results.items() if r['frame_count'] == recommended_count]
            logger.info(f"   • Recommended threshold(s): {recommended_thresholds} ({recommended_count} frames)")
        
        logger.info("")
        logger.info("🔍 To compare results, examine the images in each directory:")
        for threshold, result in sorted_results:
            logger.info(f"   ls -la {result['output_dir']}/")


def main():
    """Main function to run the threshold test."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python test_similarity_thresholds.py <video_path> [threshold]")
        print("Examples:")
        print("  python test_similarity_thresholds.py /path/to/video.mp4          # Test all thresholds 0.1-1.0")
        print("  python test_similarity_thresholds.py /path/to/video.mp4 0.8      # Test only threshold 0.8")
        sys.exit(1)
    
    video_path = sys.argv[1]
    single_threshold = None
    
    if len(sys.argv) == 3:
        try:
            single_threshold = float(sys.argv[2])
            if single_threshold < 0.1 or single_threshold > 1.0:
                print("Error: Threshold must be between 0.1 and 1.0")
                sys.exit(1)
        except ValueError:
            print("Error: Threshold must be a valid number")
            sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run the test
    tester = SimilarityThresholdTester(video_path)
    
    if single_threshold:
        print(f"🎯 Testing single threshold: {single_threshold}")
        result = tester.test_single_threshold(single_threshold)
        print(f"\n✅ Single threshold test complete!")
        print(f"   • Threshold: {single_threshold}")
        print(f"   • Frame count: {result['frame_count']}")
        print(f"   • Results in: {result['output_dir']}/")
    else:
        print(f"🎯 Testing all thresholds: 0.1 to 1.0")
        tester.test_all_thresholds()
        print(f"\n✅ Threshold analysis complete! Check the images_X.X directories for results.")


if __name__ == "__main__":
    main()