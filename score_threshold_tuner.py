#!/usr/bin/env python3
"""
Score Threshold Tuner - Experiment with different scoring parameters for semantic frame selection.

This script helps you find the optimal scoring thresholds by running multiple experiments
with different parameter combinations and showing you the results.
"""

import os
import sys
import json
import tempfile
import shutil
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_frame_selector import SemanticFrameSelector
from video_processor import VideoProcessor
from transcript_extractor import TranscriptExtractor
from config import Config

class ScoreThresholdTuner:
    """Tunes scoring thresholds for optimal frame selection."""
    
    def __init__(self, config_path: str = None):
        """Initialize with configuration."""
        self.config = Config(config_path or 'config.yaml')
        self.config_dict = self.config.get_config()
        
        # Ensure we have required API keys
        if not self.config_dict.get('gemini_api_key') and not os.getenv('GOOGLE_API_KEY'):
            raise ValueError("Gemini API key required. Set GOOGLE_API_KEY or configure in config file.")
    
    def run_experiment(
        self, 
        video_path: str, 
        video_title: str,
        score_threshold: float,
        base_score_weight: float = 0.3,
        text_score_weight: float = 0.4,
        visual_score_weight: float = 0.3,
        max_frames_per_section: int = 3,
        min_frame_spacing: float = 10.0
    ) -> Dict:
        """Run a single experiment with given parameters."""
        
        print(f"\n🧪 EXPERIMENT: threshold={score_threshold}, weights=({base_score_weight:.1f}, {text_score_weight:.1f}, {visual_score_weight:.1f})")
        
        # Create temporary config with experimental parameters
        experimental_config = self.config_dict.copy()
        experimental_config.update({
            'semantic_frame_selection': {
                'score_threshold': score_threshold,
                'base_score_weight': base_score_weight,
                'text_score_weight': text_score_weight,
                'visual_score_weight': visual_score_weight,
                'max_frames_per_section': max_frames_per_section,
                'min_frame_spacing': min_frame_spacing
            }
        })
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract transcript
                transcript_extractor = TranscriptExtractor(experimental_config)
                transcript_segments = transcript_extractor.extract_transcript(video_path)
                
                # Run semantic frame selection
                video_processor = VideoProcessor(experimental_config)
                semantic_selector = SemanticFrameSelector(experimental_config, video_processor)
                
                # Modify scoring weights in the frame scorer
                semantic_selector.frame_scorer.base_score_weight = base_score_weight
                semantic_selector.frame_scorer.text_score_weight = text_score_weight
                semantic_selector.frame_scorer.visual_score_weight = visual_score_weight
                
                selected_frames = semantic_selector.select_frames_semantically(
                    video_path, transcript_segments, temp_dir, video_title
                )
                
                # Analyze results
                total_frames = len(selected_frames)
                sections = {}
                score_distribution = []
                timestamps = []
                
                for frame in selected_frames:
                    section = frame.get('section_title', 'Unknown')
                    if section not in sections:
                        sections[section] = 0
                    sections[section] += 1
                    
                    score_distribution.append(frame.get('semantic_score', 0))
                    timestamps.append(frame['timestamp'])
                
                # Calculate metrics
                avg_score = sum(score_distribution) / len(score_distribution) if score_distribution else 0
                min_score = min(score_distribution) if score_distribution else 0
                max_score = max(score_distribution) if score_distribution else 0
                
                # Calculate time coverage
                if timestamps:
                    time_span = max(timestamps) - min(timestamps)
                    video_duration = transcript_segments[-1]['end_time'] if transcript_segments else 0
                    coverage_percent = (time_span / video_duration * 100) if video_duration > 0 else 0
                else:
                    time_span = 0
                    coverage_percent = 0
                
                return {
                    'parameters': {
                        'score_threshold': score_threshold,
                        'base_score_weight': base_score_weight,
                        'text_score_weight': text_score_weight,
                        'visual_score_weight': visual_score_weight,
                        'max_frames_per_section': max_frames_per_section,
                        'min_frame_spacing': min_frame_spacing
                    },
                    'results': {
                        'total_frames': total_frames,
                        'sections_with_frames': len(sections),
                        'frames_per_section': sections,
                        'avg_score': avg_score,
                        'min_score': min_score,
                        'max_score': max_score,
                        'time_coverage_percent': coverage_percent,
                        'time_span_seconds': time_span,
                        'timestamps': timestamps[:10],  # First 10 for preview
                        'score_distribution': score_distribution[:10]  # First 10 for preview
                    }
                }
                
        except Exception as e:
            return {
                'parameters': {
                    'score_threshold': score_threshold,
                    'base_score_weight': base_score_weight,
                    'text_score_weight': text_score_weight,
                    'visual_score_weight': visual_score_weight
                },
                'results': {
                    'error': str(e),
                    'total_frames': 0
                }
            }
    
    def run_threshold_sweep(
        self, 
        video_path: str, 
        video_title: str,
        threshold_range: Tuple[float, float, float] = (10.0, 100.0, 10.0),
        output_file: str = None
    ) -> List[Dict]:
        """Run experiments across a range of score thresholds."""
        
        print(f"🔍 THRESHOLD SWEEP: {threshold_range[0]} to {threshold_range[1]} step {threshold_range[2]}")
        
        results = []
        thresholds = []
        
        # Generate threshold values
        current = threshold_range[0]
        while current <= threshold_range[1]:
            thresholds.append(current)
            current += threshold_range[2]
        
        for threshold in thresholds:
            result = self.run_experiment(video_path, video_title, threshold)
            results.append(result)
            
            # Print summary
            if 'error' not in result['results']:
                frames = result['results']['total_frames']
                sections = result['results']['sections_with_frames']
                avg_score = result['results']['avg_score']
                coverage = result['results']['time_coverage_percent']
                print(f"   Threshold {threshold:5.1f}: {frames:2d} frames, {sections} sections, avg_score={avg_score:.1f}, coverage={coverage:.1f}%")
            else:
                print(f"   Threshold {threshold:5.1f}: ERROR - {result['results']['error']}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def run_weight_experiment(
        self, 
        video_path: str, 
        video_title: str,
        fixed_threshold: float = 50.0
    ) -> List[Dict]:
        """Experiment with different scoring weight combinations."""
        
        print(f"⚖️  WEIGHT EXPERIMENTS: fixed threshold = {fixed_threshold}")
        
        # Different weight combinations to try
        weight_combinations = [
            (0.5, 0.3, 0.2),  # Base-heavy
            (0.3, 0.5, 0.2),  # Text-heavy  
            (0.3, 0.4, 0.3),  # Balanced (current)
            (0.2, 0.5, 0.3),  # Text-focused
            (0.4, 0.3, 0.3),  # Base-focused
            (0.2, 0.3, 0.5),  # Visual-heavy
            (0.33, 0.33, 0.34), # Equal weights
        ]
        
        results = []
        
        for base_w, text_w, visual_w in weight_combinations:
            result = self.run_experiment(
                video_path, video_title, fixed_threshold,
                base_score_weight=base_w,
                text_score_weight=text_w,
                visual_score_weight=visual_w
            )
            results.append(result)
            
            # Print summary
            if 'error' not in result['results']:
                frames = result['results']['total_frames']
                sections = result['results']['sections_with_frames']
                avg_score = result['results']['avg_score']
                print(f"   Weights ({base_w:.1f}, {text_w:.1f}, {visual_w:.1f}): {frames:2d} frames, {sections} sections, avg_score={avg_score:.1f}")
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze experiment results to find optimal parameters."""
        
        if not results:
            return {}
        
        # Filter out errored results
        valid_results = [r for r in results if 'error' not in r['results']]
        
        if not valid_results:
            print("❌ No valid results to analyze")
            return {}
        
        # Find results with good characteristics
        scored_results = []
        
        for result in valid_results:
            r = result['results']
            
            # Scoring criteria
            frame_count_score = min(r['total_frames'] / 15.0, 1.0) * 30  # Prefer 10-15 frames
            section_coverage_score = min(r['sections_with_frames'] / 5.0, 1.0) * 25  # Prefer 3-5 sections
            time_coverage_score = min(r['time_coverage_percent'] / 80.0, 1.0) * 25  # Prefer 60-80% coverage
            avg_score_bonus = min(r['avg_score'] / 100.0, 1.0) * 20  # Prefer higher average scores
            
            total_score = frame_count_score + section_coverage_score + time_coverage_score + avg_score_bonus
            
            scored_results.append({
                'result': result,
                'quality_score': total_score,
                'frame_count_score': frame_count_score,
                'section_coverage_score': section_coverage_score,
                'time_coverage_score': time_coverage_score,
                'avg_score_bonus': avg_score_bonus
            })
        
        # Sort by quality score
        scored_results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"\n🏆 TOP 3 CONFIGURATIONS:")
        for i, scored in enumerate(scored_results[:3]):
            result = scored['result']
            params = result['parameters']
            results_data = result['results']
            
            print(f"\n   #{i+1} (Quality Score: {scored['quality_score']:.1f})")
            print(f"      Threshold: {params['score_threshold']}")
            if 'base_score_weight' in params:
                print(f"      Weights: ({params['base_score_weight']:.1f}, {params['text_score_weight']:.1f}, {params['visual_score_weight']:.1f})")
            print(f"      Results: {results_data['total_frames']} frames, {results_data['sections_with_frames']} sections")
            print(f"      Coverage: {results_data['time_coverage_percent']:.1f}%, Avg Score: {results_data['avg_score']:.1f}")
        
        return {
            'best_result': scored_results[0]['result'] if scored_results else None,
            'all_scored_results': scored_results
        }

def main():
    parser = argparse.ArgumentParser(description='Tune semantic frame selection scoring thresholds')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--title', help='Video title (default: derived from filename)')
    parser.add_argument('--config', help='Config file path (default: config.yaml)')
    parser.add_argument('--mode', choices=['threshold', 'weights', 'both'], default='both',
                       help='Experiment mode (default: both)')
    parser.add_argument('--threshold-range', nargs=3, type=float, default=[20.0, 80.0, 10.0],
                       metavar=('START', 'END', 'STEP'), 
                       help='Threshold range: start end step (default: 20 80 10)')
    parser.add_argument('--fixed-threshold', type=float, default=50.0,
                       help='Fixed threshold for weight experiments (default: 50.0)')
    parser.add_argument('--output', help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    # Generate title if not provided
    title = args.title or Path(args.video).stem.replace('_', ' ').replace('-', ' ').title()
    
    print(f"🎬 Video: {args.video}")
    print(f"📝 Title: {title}")
    
    try:
        tuner = ScoreThresholdTuner(args.config)
        
        all_results = []
        
        if args.mode in ['threshold', 'both']:
            print(f"\n{'='*60}")
            print("THRESHOLD SWEEP EXPERIMENTS")
            print(f"{'='*60}")
            
            threshold_results = tuner.run_threshold_sweep(
                args.video, title, tuple(args.threshold_range), args.output
            )
            all_results.extend(threshold_results)
        
        if args.mode in ['weights', 'both']:
            print(f"\n{'='*60}")
            print("WEIGHT COMBINATION EXPERIMENTS")
            print(f"{'='*60}")
            
            weight_results = tuner.run_weight_experiment(
                args.video, title, args.fixed_threshold
            )
            all_results.extend(weight_results)
        
        # Analyze all results
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        
        analysis = tuner.analyze_results(all_results)
        
        if analysis.get('best_result'):
            best = analysis['best_result']
            print(f"\n✅ RECOMMENDED CONFIGURATION:")
            print(f"   Add to your config.yaml:")
            print(f"   semantic_frame_selection:")
            print(f"     score_threshold: {best['parameters']['score_threshold']}")
            if 'base_score_weight' in best['parameters']:
                print(f"     base_score_weight: {best['parameters']['base_score_weight']}")
                print(f"     text_score_weight: {best['parameters']['text_score_weight']}")  
                print(f"     visual_score_weight: {best['parameters']['visual_score_weight']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()