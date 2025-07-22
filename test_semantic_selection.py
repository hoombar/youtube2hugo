#!/usr/bin/env python3
"""
Test script to compare semantic vs temporal frame selection approaches.

Usage: python test_semantic_selection.py <video_path> [--approach semantic|temporal|both]

This script demonstrates the new semantic-driven frame selection approach
compared to the existing temporal paragraph-based approach.
"""

import sys
import os
import shutil
import tempfile
import argparse
import logging
import yaml
from typing import Dict, List

# Import our modules
from semantic_frame_selector import SemanticFrameSelector
from video_processor import VideoProcessor
from transcript_extractor import TranscriptExtractor
from hugo_generator import HugoGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SemanticFrameSelectionTester:
    """Test and compare semantic vs temporal frame selection approaches."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cached_transcript = None
        
        # Load configuration from config file (same as main script)
        self.base_config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from config.local.yaml or environment variables."""
        
        config = {
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
            'whisper_model': 'base',
            'language': 'en',
            'image_similarity_threshold': 0.8,
            'image_max_width': 1920,
            'image_max_height': 1080,
            'image_quality': 95,
            'gemini_model': 'gemini-2.5-flash'
        }
        
        # Try to load from config.local.yaml
        config_path = 'config.local.yaml'
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            try:
                with open(config_path, 'r') as f:
                    local_config = yaml.safe_load(f) or {}
                
                # Extract Gemini API key from config structure
                if 'gemini' in local_config:
                    gemini_config = local_config['gemini']
                    if 'api_key' in gemini_config:
                        config['gemini_api_key'] = gemini_config['api_key']
                    if 'model' in gemini_config:
                        config['gemini_model'] = gemini_config['model']
                
                # Extract other settings
                if 'processing' in local_config:
                    processing = local_config['processing']
                    if 'image_similarity_threshold' in processing:
                        config['image_similarity_threshold'] = processing['image_similarity_threshold']
                    if 'default_whisper_model' in processing:
                        config['whisper_model'] = processing['default_whisper_model']
                
                logger.info("✅ Configuration loaded from config.local.yaml")
                
            except Exception as e:
                logger.warning(f"Failed to load config.local.yaml: {e}")
        else:
            logger.info(f"No {config_path} found, using defaults")
        
        # Fallback to environment variable
        if 'gemini_api_key' not in config:
            config['gemini_api_key'] = os.getenv('GOOGLE_API_KEY')
        
        return config
    
    def test_semantic_approach(self, output_dir: str = "semantic_frames") -> Dict:
        """Test the semantic frame selection approach."""
        
        logger.info("🧠 Testing SEMANTIC frame selection approach...")
        
        # Clean up existing directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Get transcript
        transcript_segments = self._get_transcript()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize processors
            video_processor = VideoProcessor(self.base_config)
            semantic_selector = SemanticFrameSelector(self.base_config, video_processor)
            
            try:
                # Run semantic frame selection
                selected_frames = semantic_selector.select_frames_semantically(
                    self.video_path, transcript_segments, temp_dir
                )
                
                # Copy selected frames to output directory
                copied_count = 0
                for i, frame in enumerate(selected_frames):
                    if os.path.exists(frame['path']):
                        timestamp = frame['timestamp']
                        score = frame.get('semantic_score', 0)
                        section = frame.get('section_title', 'Unknown')
                        importance = frame.get('section_importance', 'medium')
                        
                        ext = os.path.splitext(frame['path'])[1]
                        dst_filename = f"frame_{i:03d}_t{timestamp:06.1f}s_score{score:.0f}_{importance}_{section.replace(' ', '_')}{ext}"
                        dst_path = os.path.join(output_dir, dst_filename)
                        
                        shutil.copy2(frame['path'], dst_path)
                        copied_count += 1
                        
                        logger.info(f"📋 Copied: {dst_filename}")
                
                result = {
                    'approach': 'semantic',
                    'frame_count': copied_count,
                    'output_dir': output_dir,
                    'frames': selected_frames
                }
                
                logger.info(f"✅ Semantic approach: {copied_count} frames saved to {output_dir}/")
                return result
                
            except Exception as e:
                logger.error(f"❌ Semantic approach failed: {e}")
                return {'approach': 'semantic', 'frame_count': 0, 'error': str(e)}
    
    def test_temporal_approach(self, output_dir: str = "temporal_frames") -> Dict:
        """Test the traditional temporal frame selection approach."""
        
        logger.info("⏰ Testing TEMPORAL frame selection approach...")
        
        # Clean up existing directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Get transcript
        transcript_segments = self._get_transcript()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize processors
            video_processor = VideoProcessor(self.base_config)
            hugo_gen = HugoGenerator(self.base_config)
            
            try:
                # Extract frames using traditional approach
                frames = video_processor.extract_frames(
                    self.video_path, temp_dir, transcript_segments
                )
                
                # Optimize images (filter to should_include=True)
                optimized_frames = video_processor.optimize_images(frames)
                
                # Apply paragraph-based selection (existing approach)
                paragraphs = hugo_gen._group_segments_into_paragraphs(transcript_segments)
                selected_frames = self._simulate_temporal_selection(
                    optimized_frames, paragraphs, hugo_gen
                )
                
                # Copy selected frames to output directory
                copied_count = 0
                for i, frame in enumerate(selected_frames):
                    if os.path.exists(frame['path']):
                        timestamp = frame['timestamp']
                        score = frame.get('score', 0)
                        
                        ext = os.path.splitext(frame['path'])[1]
                        dst_filename = f"frame_{i:03d}_t{timestamp:06.1f}s_score{score:.0f}_temporal{ext}"
                        dst_path = os.path.join(output_dir, dst_filename)
                        
                        shutil.copy2(frame['path'], dst_path)
                        copied_count += 1
                        
                        logger.info(f"📋 Copied: {dst_filename}")
                
                result = {
                    'approach': 'temporal',
                    'frame_count': copied_count,
                    'output_dir': output_dir,
                    'frames': selected_frames,
                    'paragraph_count': len(paragraphs)
                }
                
                logger.info(f"✅ Temporal approach: {copied_count} frames from {len(paragraphs)} paragraphs saved to {output_dir}/")
                return result
                
            except Exception as e:
                logger.error(f"❌ Temporal approach failed: {e}")
                return {'approach': 'temporal', 'frame_count': 0, 'error': str(e)}
    
    def _simulate_temporal_selection(self, frames: List[Dict], paragraphs: List[List[Dict]], hugo_gen: HugoGenerator) -> List[Dict]:
        """Simulate the existing temporal paragraph-based frame selection."""
        
        frames_with_include = [f for f in frames if f.get('should_include', False)]
        sorted_frames = sorted(frames_with_include, key=lambda x: x['timestamp'])
        
        selected_frames = []
        used_frames = set()
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_start = paragraph[0]['start_time']
            paragraph_end = paragraph[-1]['end_time']
            
            # Find relevant frames for this paragraph (existing logic)
            relevant_frames = hugo_gen._find_relevant_frames_for_paragraph(
                paragraph, sorted_frames, used_frames
            )
            
            if relevant_frames:
                for frame in relevant_frames:
                    selected_frames.append(frame)
                    used_frames.add(frame['timestamp'])
                    
                logger.info(f"  📝 Paragraph {i+1} ({paragraph_start:.1f}s-{paragraph_end:.1f}s): {len(relevant_frames)} frames")
        
        return selected_frames
    
    def compare_approaches(self) -> Dict:
        """Compare both approaches and provide analysis."""
        
        logger.info("🔄 Running comparison between semantic and temporal approaches...")
        
        # Test both approaches
        semantic_result = self.test_semantic_approach()
        temporal_result = self.test_temporal_approach()
        
        # Analyze results
        comparison = {
            'semantic': semantic_result,
            'temporal': temporal_result
        }
        
        # Print comparison summary
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict):
        """Print detailed comparison analysis."""
        
        semantic = comparison['semantic']
        temporal = comparison['temporal']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 SEMANTIC vs TEMPORAL FRAME SELECTION COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"Video: {os.path.basename(self.video_path)}")
        logger.info("")
        
        # Frame counts
        logger.info("📊 FRAME SELECTION RESULTS:")
        logger.info(f"   Semantic approach: {semantic.get('frame_count', 0)} frames")
        logger.info(f"   Temporal approach: {temporal.get('frame_count', 0)} frames")
        
        if semantic.get('frame_count', 0) > 0 and temporal.get('frame_count', 0) > 0:
            ratio = semantic['frame_count'] / temporal['frame_count']
            logger.info(f"   Ratio (semantic/temporal): {ratio:.2f}")
        
        # Errors
        if semantic.get('error'):
            logger.info(f"   ❌ Semantic error: {semantic['error']}")
        if temporal.get('error'):
            logger.info(f"   ❌ Temporal error: {temporal['error']}")
        
        # Output directories
        logger.info("")
        logger.info("📁 OUTPUT LOCATIONS:")
        logger.info(f"   Semantic frames: {semantic.get('output_dir', 'N/A')}/")
        logger.info(f"   Temporal frames: {temporal.get('output_dir', 'N/A')}/")
        
        # Analysis recommendations
        logger.info("")
        logger.info("💡 ANALYSIS:")
        if semantic.get('frame_count', 0) > temporal.get('frame_count', 0):
            logger.info("   • Semantic approach selected MORE frames - may capture more content detail")
        elif semantic.get('frame_count', 0) < temporal.get('frame_count', 0):
            logger.info("   • Semantic approach selected FEWER frames - may be more focused on relevant content")
        else:
            logger.info("   • Both approaches selected same number of frames - interesting convergence!")
        
        logger.info("   • Compare frame quality and relevance by examining both output directories")
        logger.info("   • Semantic frames include section context in filenames for easier analysis")
    
    def _get_transcript(self) -> List[Dict]:
        """Get transcript (cached after first call)."""
        
        if self._cached_transcript is None:
            logger.info("🎙️  Extracting transcript with Whisper...")
            
            extractor_config = {
                'whisper_model': self.base_config.get('whisper_model', 'base'),
                'language': self.base_config.get('language', 'en')
            }
            
            extractor = TranscriptExtractor(extractor_config)
            self._cached_transcript = extractor.extract_transcript(self.video_path)
            
            logger.info(f"📝 Extracted {len(self._cached_transcript)} transcript segments")
        else:
            logger.info(f"🎙️  Using cached transcript ({len(self._cached_transcript)} segments)")
        
        return self._cached_transcript


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description="Compare semantic vs temporal frame selection")
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument(
        '--approach', 
        choices=['semantic', 'temporal', 'both'], 
        default='both',
        help='Which approach to test (default: both)'
    )
    parser.add_argument(
        '--semantic-dir',
        default='semantic_frames',
        help='Output directory for semantic frames (default: semantic_frames)'
    )
    parser.add_argument(
        '--temporal-dir', 
        default='temporal_frames',
        help='Output directory for temporal frames (default: temporal_frames)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Create tester to check configuration
    tester = SemanticFrameSelectionTester(args.video_path)
    
    # Check for Gemini API key if testing semantic approach
    if args.approach in ['semantic', 'both']:
        if not tester.base_config.get('gemini_api_key'):
            print("Error: Gemini API key required for semantic approach")
            print("Options:")
            print("  1. Add to config.local.yaml under gemini.api_key")
            print("  2. Set environment variable: export GOOGLE_API_KEY=your_api_key")
            sys.exit(1)
    
    # Run tests (tester already created above)
    
    try:
        if args.approach == 'semantic':
            result = tester.test_semantic_approach(args.semantic_dir)
        elif args.approach == 'temporal':
            result = tester.test_temporal_approach(args.temporal_dir)
        elif args.approach == 'both':
            result = tester.compare_approaches()
        
        print(f"\n✅ Testing complete! Check output directories for results.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()