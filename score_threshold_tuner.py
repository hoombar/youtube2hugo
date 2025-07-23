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
import logging
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

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
        # Load configuration using the proper Config class methods
        if config_path:
            self.config_dict = Config.load_from_file(config_path)
        else:
            # Try local config first
            self.config_dict = Config.load_local_config()
            
            # If no local config, try main config
            if not self.config_dict and os.path.exists('config.yaml'):
                self.config_dict = Config.load_from_file('config.yaml')
        
        # Ensure we have a dict
        if not self.config_dict:
            self.config_dict = {}
        
        # Check for Gemini API key in various locations
        gemini_config = self.config_dict.get('gemini', {})
        gemini_key = (self.config_dict.get('gemini_api_key') or 
                     (gemini_config.get('api_key') if isinstance(gemini_config, dict) else None) or
                     os.getenv('GOOGLE_API_KEY'))
        
        if not gemini_key:
            raise ValueError("Gemini API key required. Add to config.local.yaml or set GOOGLE_API_KEY environment variable.")
        
        # Ensure the config has the key for experiments
        if 'gemini_api_key' not in self.config_dict:
            self.config_dict['gemini_api_key'] = gemini_key
    
    def run_experiment(
        self, 
        video_path: str, 
        video_title: str,
        transcript_segments: List[Dict],
        semantic_sections: List[Dict],
        score_threshold: float,
        base_score_weight: float = 0.3,
        text_score_weight: float = 0.4,
        visual_score_weight: float = 0.3,
        max_frames_per_section: int = 3,
        min_frame_spacing: float = 10.0,
        save_frames: bool = False,
        output_dir: str = None
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
                # Use pre-extracted transcript and sections
                video_processor = VideoProcessor(experimental_config)
                semantic_selector = SemanticFrameSelector(experimental_config, video_processor)
                
                # Modify scoring weights in the frame scorer
                semantic_selector.frame_scorer.base_score_weight = base_score_weight
                semantic_selector.frame_scorer.text_score_weight = text_score_weight
                semantic_selector.frame_scorer.visual_score_weight = visual_score_weight
                semantic_selector.frame_scorer.score_threshold = score_threshold
                
                # Run frame selection using pre-computed sections (skip transcript analysis)
                selected_frames = self._run_frame_selection_with_sections(
                    semantic_selector, video_path, transcript_segments, semantic_sections, temp_dir, video_title
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
                
                # Save frames if requested
                saved_frames_info = []
                if save_frames and output_dir and selected_frames:
                    experiment_name = f"threshold_{score_threshold}_weights_{base_score_weight:.1f}_{text_score_weight:.1f}_{visual_score_weight:.1f}"
                    experiment_dir = os.path.join(output_dir, experiment_name)
                    os.makedirs(experiment_dir, exist_ok=True)
                    
                    print(f"   💾 Attempting to save {total_frames} frames to: {experiment_dir}")
                    
                    saved_count = 0
                    for i, frame in enumerate(selected_frames):
                        # Debug frame structure
                        if i == 0:  # Log first frame structure for debugging
                            print(f"   🔍 Frame structure: {list(frame.keys())}")
                            if 'path' in frame:
                                print(f"   🔍 Frame path: {frame['path']}, exists: {os.path.exists(frame['path'])}")
                        
                        if 'path' in frame and os.path.exists(frame['path']):
                            # Copy frame to experiment directory
                            original_name = os.path.basename(frame['path'])
                            new_name = f"{i+1:02d}_{frame['timestamp']:.1f}s_{frame.get('section_title', 'unknown').replace(' ', '_')}.jpg"
                            dest_path = os.path.join(experiment_dir, new_name)
                            
                            try:
                                shutil.copy2(frame['path'], dest_path)
                                saved_frames_info.append({
                                    'original_path': frame['path'],
                                    'saved_path': dest_path,
                                    'timestamp': frame['timestamp'],
                                    'section': frame.get('section_title', 'Unknown'),
                                    'score': frame.get('semantic_score', 0)
                                })
                                saved_count += 1
                            except Exception as copy_error:
                                print(f"   ⚠️  Failed to copy frame {original_name}: {copy_error}")
                        else:
                            path_info = frame.get('path', 'NO_PATH_KEY')
                            print(f"   ⚠️  Frame {i+1} has no valid path: {path_info}")
                    
                    print(f"   ✅ Successfully saved {saved_count}/{total_frames} frames")
                    
                    # Create summary HTML for easy viewing
                    if saved_frames_info:
                        self._create_frame_summary_html(experiment_dir, saved_frames_info, experiment_name)
                    else:
                        print(f"   ❌ No frames saved - cannot create HTML summary")
                
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
                        'score_distribution': score_distribution[:10],  # First 10 for preview
                        'saved_frames': saved_frames_info if save_frames else []
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
    
    def _create_frame_summary_html(self, experiment_dir: str, frames_info: List[Dict], experiment_name: str):
        """Create HTML summary for easy visual inspection of selected frames."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Frame Selection Results - {experiment_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .frame-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .frame-item {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .frame-image {{ width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }}
        .frame-info {{ margin-top: 10px; }}
        .timestamp {{ font-weight: bold; color: #007acc; }}
        .section {{ color: #666; font-style: italic; }}
        .score {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Frame Selection Results</h1>
        <h2>{experiment_name.replace('_', ' ').title()}</h2>
        <p><strong>Total Frames:</strong> {len(frames_info)}</p>
    </div>
    <div class="frame-grid">
"""
        
        for frame in frames_info:
            filename = os.path.basename(frame['saved_path'])
            html_content += f"""
        <div class="frame-item">
            <img src="{filename}" alt="Frame at {frame['timestamp']:.1f}s" class="frame-image">
            <div class="frame-info">
                <div class="timestamp">⏱️ {frame['timestamp']:.1f}s</div>
                <div class="section">📋 {frame['section']}</div>
                <div class="score">⭐ Score: {frame['score']:.1f}</div>
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        html_path = os.path.join(experiment_dir, "index.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"   🌐 Created visual summary: {html_path}")
    
    def _run_frame_selection_with_sections(
        self, 
        semantic_selector: 'SemanticFrameSelector',
        video_path: str,
        transcript_segments: List[Dict],
        semantic_sections: List[Dict],
        temp_dir: str,
        video_title: str
    ) -> List[Dict]:
        """Run frame selection using pre-computed semantic sections."""
        
        # Set the video title for title sequence detection
        semantic_selector.frame_scorer.video_title = video_title or ""
        
        # Skip section analysis and jump straight to frame processing
        logger.info(f"📋 Using {len(semantic_sections)} pre-computed semantic sections")
        
        # Extract and score all candidate frames 
        all_frames = semantic_selector._extract_all_candidate_frames(
            video_path, semantic_sections, temp_dir
        )
        
        # Select best frames for each section
        selected_frames = []
        for section in semantic_sections:
            section_frames = semantic_selector._select_frames_for_section_optimized(
                section, all_frames
            )
            selected_frames.extend(section_frames)
        
        # Remove duplicates and sort by timestamp
        seen_timestamps = set()
        unique_frames = []
        for frame in selected_frames:
            timestamp = frame['timestamp']
            if timestamp not in seen_timestamps:
                unique_frames.append(frame)
                seen_timestamps.add(timestamp)
        
        unique_frames.sort(key=lambda x: x['timestamp'])
        return unique_frames
    
    def _extract_transcript_and_sections(self, video_path: str, video_title: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract transcript and semantic sections once for reuse across experiments."""
        
        print(f"🎬 Extracting transcript and analyzing semantic sections...")
        
        # Extract transcript
        transcript_extractor = TranscriptExtractor(self.config_dict)
        transcript_segments = transcript_extractor.extract_transcript(video_path)
        print(f"   📝 Extracted {len(transcript_segments)} transcript segments")
        
        # Analyze semantic sections
        from semantic_frame_selector import SemanticSectionAnalyzer
        section_analyzer = SemanticSectionAnalyzer(self.config_dict)
        semantic_sections = section_analyzer.analyze_transcript_sections(transcript_segments)
        print(f"   🧠 Identified {len(semantic_sections)} semantic sections")
        
        for i, section in enumerate(semantic_sections):
            print(f"      Section {i+1}: {section['start_time']:.1f}s-{section['end_time']:.1f}s - {section['title']}")
        
        return transcript_segments, semantic_sections
    
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
    
    def run_intelligent_variations(self, video_path: str, video_title: str, save_frames: bool = False, output_dir: str = None) -> List[Dict]:
        """Run several intelligent variations designed to yield different useful results."""
        
        print(f"🎯 INTELLIGENT VARIATIONS: Testing different frame selection strategies")
        
        # Extract transcript and semantic sections once for all experiments
        transcript_segments, semantic_sections = self._extract_transcript_and_sections(video_path, video_title)
        
        # Define variations with more dramatic differences for meaningful comparison
        variations = [
            {
                'name': 'Ultra Conservative',
                'description': 'Only the highest quality frames',
                'threshold': 90.0,
                'base_weight': 0.5,
                'text_weight': 0.3,
                'visual_weight': 0.2,
                'max_frames': 1
            },
            {
                'name': 'High Quality',
                'description': 'Strict quality with moderate coverage',
                'threshold': 75.0,
                'base_weight': 0.4,
                'text_weight': 0.3,
                'visual_weight': 0.3,
                'max_frames': 2
            },
            {
                'name': 'Balanced',
                'description': 'Good balance of quality and coverage',
                'threshold': 50.0,
                'base_weight': 0.3,
                'text_weight': 0.4,
                'visual_weight': 0.3,
                'max_frames': 3
            },
            {
                'name': 'High Coverage',
                'description': 'More frames with reasonable quality',
                'threshold': 25.0,
                'base_weight': 0.3,
                'text_weight': 0.4,
                'visual_weight': 0.3,
                'max_frames': 4
            },
            {
                'name': 'Maximum Coverage',
                'description': 'Maximum frames with minimal quality bar',
                'threshold': 10.0,
                'base_weight': 0.3,
                'text_weight': 0.4,
                'visual_weight': 0.3,
                'max_frames': 6
            },
            {
                'name': 'Text-Heavy Strategy',
                'description': 'Heavy preference for text/code content',
                'threshold': 35.0,
                'base_weight': 0.1,
                'text_weight': 0.8,
                'visual_weight': 0.1,
                'max_frames': 3
            }
        ]
        
        results = []
        
        for i, variation in enumerate(variations):
            print(f"\n📋 Variation {i+1}/{ len(variations)}: {variation['name']}")
            print(f"   {variation['description']}")
            
            result = self.run_experiment(
                video_path, video_title,
                transcript_segments, semantic_sections,
                score_threshold=variation['threshold'],
                base_score_weight=variation['base_weight'],
                text_score_weight=variation['text_weight'],
                visual_score_weight=variation['visual_weight'],
                max_frames_per_section=variation['max_frames'],
                save_frames=save_frames,
                output_dir=output_dir
            )
            
            # Add variation info to result
            result['variation_name'] = variation['name']
            result['variation_description'] = variation['description']
            results.append(result)
            
            # Print summary
            if 'error' not in result['results']:
                frames = result['results']['total_frames']
                sections = result['results']['sections_with_frames']
                avg_score = result['results']['avg_score']
                coverage = result['results']['time_coverage_percent']
                print(f"   📊 Results: {frames} frames across {sections} sections")
                print(f"      Coverage: {coverage:.1f}% | Avg Score: {avg_score:.1f}")
            else:
                print(f"   ❌ Error: {result['results']['error']}")
        
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
            
            # Show variation name if available
            if 'variation_name' in result:
                print(f"      Strategy: {result['variation_name']}")
                print(f"      Description: {result['variation_description']}")
            
            print(f"      Threshold: {params['score_threshold']}")
            if 'base_score_weight' in params:
                print(f"      Weights: Base={params['base_score_weight']:.1f}, Text={params['text_score_weight']:.1f}, Visual={params['visual_score_weight']:.1f}")
            print(f"      Results: {results_data['total_frames']} frames across {results_data['sections_with_frames']} sections")
            print(f"      Coverage: {results_data['time_coverage_percent']:.1f}% | Avg Score: {results_data['avg_score']:.1f}")
        
        return {
            'best_result': scored_results[0]['result'] if scored_results else None,
            'all_scored_results': scored_results
        }

def main():
    parser = argparse.ArgumentParser(description='Find optimal semantic frame selection settings')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--title', help='Video title (default: derived from filename)')
    parser.add_argument('--config', help='Config file path (default: config.local.yaml or config.yaml)')
    parser.add_argument('--mode', choices=['smart', 'threshold', 'weights', 'both'], default='smart',
                       help='Experiment mode - smart runs intelligent variations (default: smart)')
    parser.add_argument('--threshold-range', nargs=3, type=float, default=[20.0, 80.0, 10.0],
                       metavar=('START', 'END', 'STEP'), 
                       help='Threshold range for threshold mode: start end step (default: 20 80 10)')
    parser.add_argument('--fixed-threshold', type=float, default=50.0,
                       help='Fixed threshold for weight experiments (default: 50.0)')
    parser.add_argument('--output', help='Output JSON file for detailed results')
    parser.add_argument('--save-frames', action='store_true', 
                       help='Save selected frames as images for visual inspection')
    parser.add_argument('--frames-dir', default='tuning_results',
                       help='Directory to save frame images (default: tuning_results)')
    
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
        
        if args.mode == 'smart':
            print(f"\n{'='*60}")
            print("INTELLIGENT VARIATION EXPERIMENTS")
            print(f"{'='*60}")
            
            smart_results = tuner.run_intelligent_variations(
                args.video, title, 
                save_frames=args.save_frames,
                output_dir=args.frames_dir if args.save_frames else None
            )
            all_results.extend(smart_results)
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(smart_results, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
            
            # Show frame saving info
            if args.save_frames:
                print(f"\n📁 Frame images saved to: {args.frames_dir}/")
                print(f"   Open the index.html files in each subdirectory to view results")
            
        elif args.mode in ['threshold', 'both']:
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
            
            if 'variation_name' in best:
                print(f"   Best Strategy: {best['variation_name']}")
                print(f"   {best['variation_description']}")
                print(f"\n   Add this to your config.local.yaml or config.yaml:")
            else:
                print(f"   Add this to your config.local.yaml or config.yaml:")
                
            print(f"   semantic_frame_selection:")
            print(f"     score_threshold: {best['parameters']['score_threshold']}")
            if 'base_score_weight' in best['parameters']:
                print(f"     base_score_weight: {best['parameters']['base_score_weight']}")
                print(f"     text_score_weight: {best['parameters']['text_score_weight']}")  
                print(f"     visual_score_weight: {best['parameters']['visual_score_weight']}")
                print(f"     max_frames_per_section: {best['parameters'].get('max_frames_per_section', 3)}")
            
            # Show what this will achieve
            r = best['results']
            print(f"\n   Expected Results:")
            print(f"     • {r['total_frames']} frames across {r['sections_with_frames']} sections")
            print(f"     • {r['time_coverage_percent']:.1f}% video coverage")
            print(f"     • Average frame quality score: {r['avg_score']:.1f}")
        else:
            print(f"\n❌ No suitable configuration found. Try running with --mode threshold to find working parameters.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()