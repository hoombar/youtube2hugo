#!/usr/bin/env python3
"""
Rapid frame selection testing script.
Test frame selection algorithm on first 35 seconds of video for quick iteration.
"""

import os
import sys
import cv2
import numpy as np
import ffmpeg
from typing import List, Dict, Optional
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our video processor
from video_processor import VideoProcessor
from config import Config

class FrameSelectionTester:
    """Test and analyze frame selection algorithm."""
    
    def __init__(self):
        self.config = Config.get_default_config()
        self.video_processor = VideoProcessor(self.config)
    
    def test_selection(self, video_path: str, test_duration: float = 35.0, output_dir: str = "frame_test_output"):
        """Test frame selection on first N seconds of video."""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"🎬 Testing frame selection on first {test_duration}s of video...")
        print(f"📁 Output directory: {output_dir}")
        
        # Create temporary test video (first 35 seconds)
        temp_video = os.path.join(output_dir, "test_segment.mp4")
        print(f"✂️  Extracting test segment...")
        
        try:
            (
                ffmpeg
                .input(video_path, t=test_duration)
                .output(temp_video, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"❌ Error creating test segment: {e}")
            return
        
        # Test frame extraction
        temp_frames_dir = os.path.join(output_dir, "temp_frames")
        extracted_frames = self.video_processor.extract_frames(temp_video, temp_frames_dir, None)
        
        # Analyze results
        print(f"\n📊 Results Summary:")
        print(f"Selected {len(extracted_frames)} frames from {test_duration}s video")
        
        # Create visual summary
        self.create_visual_summary(extracted_frames, output_dir, test_duration)
        
        # Create detailed analysis
        self.create_detailed_analysis(extracted_frames, output_dir)
        
        # Generate all candidate frames for comparison
        self.generate_all_candidates(temp_video, output_dir, test_duration)
        
        print(f"\n✅ Testing complete! Check {output_dir} for results.")
        print(f"🖼️  View summary.png for visual overview")
        print(f"📄 View analysis.json for detailed scores")
        
        # Cleanup temp video
        if os.path.exists(temp_video):
            os.remove(temp_video)
    
    def reverse_engineer(self, video_path: str, target_timestamps: List[float], output_dir: str = "reverse_analysis"):
        """Test algorithm to include ALL target timestamps and exclude non-targets."""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"🔍 Reverse engineering to include ALL targets: {target_timestamps}")
        
        # Test first 60 seconds with dense sampling
        test_duration = 60.0
        temp_video = os.path.join(output_dir, "test_60s.mp4")
        
        print(f"✂️  Extracting first {test_duration}s...")
        try:
            (
                ffmpeg
                .input(video_path, t=test_duration)
                .output(temp_video, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"❌ Error creating test segment: {e}")
            return
        
        # Test with current algorithm
        print("\n🤖 Testing current algorithm...")
        temp_frames_dir = os.path.join(output_dir, "algorithm_frames")
        extracted_frames = self.video_processor.extract_frames(temp_video, temp_frames_dir, None)
        
        # Analyze results vs targets
        selected_timestamps = [f['timestamp'] for f in extracted_frames]
        
        print(f"\n📊 Algorithm Results:")
        print(f"Selected: {[f'{ts:.1f}s' for ts in selected_timestamps]}")
        
        # Check which targets were missed/included
        tolerance = 1.0  # Consider frames within 1 second as matching
        included_targets = []
        missed_targets = []
        
        for target in target_timestamps:
            found_match = any(abs(target - selected) <= tolerance for selected in selected_timestamps)
            if found_match:
                included_targets.append(target)
            else:
                missed_targets.append(target)
        
        # Check for unwanted selections (not near any target)
        unwanted_selections = []
        for selected in selected_timestamps:
            is_near_target = any(abs(selected - target) <= tolerance for target in target_timestamps)
            if not is_near_target:
                unwanted_selections.append(selected)
        
        print(f"\n✅ Included targets: {[f'{ts:.1f}s' for ts in included_targets]}")
        print(f"❌ Missed targets: {[f'{ts:.1f}s' for ts in missed_targets]}")
        print(f"⚠️  Unwanted selections: {[f'{ts:.1f}s' for ts in unwanted_selections]}")
        
        # Calculate success metrics
        target_recall = len(included_targets) / len(target_timestamps) if target_timestamps else 0
        precision = len(included_targets) / len(selected_timestamps) if selected_timestamps else 0
        
        print(f"\n📈 Performance Metrics:")
        print(f"Target Recall: {target_recall:.2f} ({len(included_targets)}/{len(target_timestamps)} targets found)")
        print(f"Precision: {precision:.2f} ({len(included_targets)}/{len(selected_timestamps)} selections were targets)")
        
        # Analyze target vs non-target frame characteristics
        self.analyze_inclusion_patterns(target_timestamps, missed_targets, unwanted_selections, temp_video, output_dir)
        
        # Generate recommendations for algorithm tuning
        self.generate_tuning_recommendations(included_targets, missed_targets, unwanted_selections, output_dir)
        
        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        print(f"\n✅ Analysis complete! Check {output_dir} for detailed results.")
    
    def create_visual_summary(self, frames: List[Dict], output_dir: str, test_duration: float):
        """Create a visual summary of selected frames."""
        
        if not frames:
            print("⚠️  No frames selected to visualize")
            return
        
        # Create timeline visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Timeline plot
        timestamps = [f['timestamp'] for f in frames]
        scores = [f.get('score', 0) for f in frames]
        
        ax1.scatter(timestamps, scores, c='red', s=100, zorder=5, label='Selected frames')
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Quality threshold')
        ax1.set_xlim(0, test_duration)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Frame Score')
        ax1.set_title('Frame Selection Timeline')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add timestamp labels
        for i, (ts, score) in enumerate(zip(timestamps, scores)):
            ax1.annotate(f'{ts:.1f}s\n{score:.0f}', (ts, score), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        # Frame thumbnails
        n_frames = len(frames)
        if n_frames > 0:
            thumb_width = 1.0 / n_frames
            
            for i, frame in enumerate(frames):
                if os.path.exists(frame['path']):
                    img = Image.open(frame['path'])
                    img.thumbnail((200, 200))
                    
                    # Create subplot for thumbnail
                    ax_thumb = fig.add_axes([i * thumb_width, 0.02, thumb_width * 0.9, 0.25])
                    ax_thumb.imshow(img)
                    ax_thumb.set_title(f"{frame['timestamp']:.1f}s\nScore: {frame.get('score', 0):.0f}", fontsize=8)
                    ax_thumb.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visual summary saved to summary.png")
    
    def create_detailed_analysis(self, frames: List[Dict], output_dir: str):
        """Create detailed JSON analysis of frame selection."""
        
        analysis = {
            'summary': {
                'total_frames_selected': len(frames),
                'average_score': np.mean([f.get('score', 0) for f in frames]) if frames else 0,
                'score_range': {
                    'min': min([f.get('score', 0) for f in frames]) if frames else 0,
                    'max': max([f.get('score', 0) for f in frames]) if frames else 0
                }
            },
            'frames': []
        }
        
        for frame in frames:
            if os.path.exists(frame['path']):
                # Get detailed analysis
                image = cv2.imread(frame['path'])
                gray = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
                
                # Analyze components
                device_score = self.video_processor._detect_devices_and_gadgets(image, gray)
                intro_penalty = self.video_processor._detect_specific_intro_sequence(image, gray)
                
                # Face detection
                faces = self.video_processor.face_cascade.detectMultiScale(gray)
                face_count = len(faces)
                face_ratio = self.video_processor._analyze_frame_composition(frame['path'])
                
                frame_analysis = {
                    'timestamp': frame['timestamp'],
                    'total_score': frame.get('score', 0),
                    'components': {
                        'device_score': device_score,
                        'intro_penalty': intro_penalty,
                        'face_count': face_count,
                        'face_ratio': face_ratio
                    },
                    'filename': os.path.basename(frame['path'])
                }
                
                analysis['frames'].append(frame_analysis)
        
        # Save analysis
        with open(os.path.join(output_dir, 'analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📄 Detailed analysis saved to analysis.json")
    
    def generate_all_candidates(self, video_path: str, output_dir: str, test_duration: float):
        """Generate all candidate frames for comparison (every 2 seconds)."""
        
        candidates_dir = os.path.join(output_dir, "all_candidates")
        if not os.path.exists(candidates_dir):
            os.makedirs(candidates_dir)
        
        print(f"🎯 Generating all candidate frames (every 2s)...")
        
        candidate_data = []
        
        for timestamp in np.arange(0, test_duration, 2.0):
            frame_path = os.path.join(candidates_dir, f"candidate_{timestamp:.1f}s.jpg")
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                # Score this candidate
                score = self.video_processor._score_frame_quality(frame_path)
                
                candidate_data.append({
                    'timestamp': timestamp,
                    'score': score,
                    'filename': os.path.basename(frame_path),
                    'selected': False  # Will update based on actual selection
                })
                
            except Exception as e:
                print(f"⚠️  Could not extract candidate at {timestamp}s: {e}")
        
        # Save candidate analysis
        with open(os.path.join(output_dir, 'all_candidates.json'), 'w') as f:
            json.dump(candidate_data, f, indent=2)
        
        print(f"🎯 Generated {len(candidate_data)} candidate frames")
    
    def analyze_inclusion_patterns(self, target_timestamps: List[float], missed_targets: List[float], unwanted_selections: List[float], video_path: str, output_dir: str):
        """Analyze characteristics of included vs missed targets vs unwanted selections."""
        
        print(f"\n🔬 Analyzing frame characteristics...")
        
        all_analysis = {
            'targets': [],
            'missed': [],
            'unwanted': []
        }
        
        # Analyze target frames
        for timestamp in target_timestamps:
            frame_path = os.path.join(output_dir, f"target_{timestamp:.1f}s.jpg")
            self._extract_and_analyze_frame(video_path, timestamp, frame_path, all_analysis['targets'])
        
        # Analyze missed targets
        for timestamp in missed_targets:
            frame_path = os.path.join(output_dir, f"missed_{timestamp:.1f}s.jpg")
            self._extract_and_analyze_frame(video_path, timestamp, frame_path, all_analysis['missed'])
        
        # Analyze unwanted selections
        for timestamp in unwanted_selections:
            frame_path = os.path.join(output_dir, f"unwanted_{timestamp:.1f}s.jpg")
            self._extract_and_analyze_frame(video_path, timestamp, frame_path, all_analysis['unwanted'])
        
        # Save detailed analysis
        with open(os.path.join(output_dir, 'inclusion_analysis.json'), 'w') as f:
            json.dump(all_analysis, f, indent=2)
        
        # Print summary statistics
        if all_analysis['targets']:
            target_scores = [f['score'] for f in all_analysis['targets']]
            print(f"Target frames - Avg score: {np.mean(target_scores):.1f}, Range: {np.min(target_scores):.1f}-{np.max(target_scores):.1f}")
        
        if all_analysis['missed']:
            missed_scores = [f['score'] for f in all_analysis['missed']]
            print(f"Missed frames - Avg score: {np.mean(missed_scores):.1f}, Range: {np.min(missed_scores):.1f}-{np.max(missed_scores):.1f}")
        
        if all_analysis['unwanted']:
            unwanted_scores = [f['score'] for f in all_analysis['unwanted']]
            print(f"Unwanted frames - Avg score: {np.mean(unwanted_scores):.1f}, Range: {np.min(unwanted_scores):.1f}-{np.max(unwanted_scores):.1f}")
    
    def _extract_and_analyze_frame(self, video_path: str, timestamp: float, frame_path: str, results_list: List[Dict]):
        """Extract and analyze a single frame."""
        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Analyze this frame
            score = self.video_processor._score_frame_quality(frame_path)
            image = cv2.imread(frame_path)
            gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            
            device_score = self.video_processor._detect_devices_and_gadgets(image, gray)
            intro_penalty = self.video_processor._detect_specific_intro_sequence(image, gray)
            
            results_list.append({
                'timestamp': timestamp,
                'path': frame_path,
                'score': score,
                'device_score': device_score,
                'intro_penalty': intro_penalty
            })
            
        except Exception as e:
            print(f"⚠️  Could not analyze frame at {timestamp}s: {e}")
    
    def generate_tuning_recommendations(self, included_targets: List[float], missed_targets: List[float], unwanted_selections: List[float], output_dir: str):
        """Generate specific recommendations for algorithm tuning."""
        
        recommendations = {
            'summary': {
                'targets_found': len(included_targets),
                'targets_missed': len(missed_targets),
                'unwanted_selections': len(unwanted_selections)
            },
            'recommendations': []
        }
        
        if missed_targets:
            recommendations['recommendations'].append({
                'issue': 'Missed target frames',
                'description': f'Algorithm missed {len(missed_targets)} target frames',
                'suggested_fixes': [
                    'Lower the minimum score threshold',
                    'Reduce minimum spacing between frames',
                    'Add more dense sampling in early video sections',
                    'Adjust content-aware timestamp generation'
                ]
            })
        
        if unwanted_selections:
            recommendations['recommendations'].append({
                'issue': 'Unwanted frame selections',
                'description': f'Algorithm selected {len(unwanted_selections)} frames not in target list',
                'suggested_fixes': [
                    'Raise the minimum score threshold',
                    'Improve content filtering',
                    'Reduce sampling density',
                    'Strengthen quality criteria'
                ]
            })
        
        if not missed_targets and not unwanted_selections:
            recommendations['recommendations'].append({
                'issue': 'Perfect match!',
                'description': 'Algorithm successfully found all targets with no unwanted selections',
                'suggested_fixes': ['No changes needed - algorithm is well-tuned']
            })
        
        # Save recommendations
        with open(os.path.join(output_dir, 'tuning_recommendations.json'), 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\n💡 Tuning Recommendations:")
        for rec in recommendations['recommendations']:
            print(f"   {rec['issue']}: {rec['description']}")
            for fix in rec['suggested_fixes']:
                print(f"     - {fix}")
    
    def analyze_target_patterns(self, target_frames: List[Dict], output_dir: str):
        """Analyze patterns in user-provided target frames."""
        
        if not target_frames:
            print("❌ No target frames to analyze")
            return
        
        # Calculate statistics
        scores = [f['score'] for f in target_frames]
        device_scores = [f['device_score'] for f in target_frames]
        intro_penalties = [f['intro_penalty'] for f in target_frames]
        
        patterns = {
            'statistics': {
                'count': len(target_frames),
                'score_stats': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                },
                'device_score_stats': {
                    'mean': float(np.mean(device_scores)),
                    'std': float(np.std(device_scores)),
                    'min': float(np.min(device_scores)),
                    'max': float(np.max(device_scores))
                },
                'intro_penalty_stats': {
                    'mean': float(np.mean(intro_penalties)),
                    'std': float(np.std(intro_penalties)),
                    'min': float(np.min(intro_penalties)),
                    'max': float(np.max(intro_penalties))
                }
            },
            'recommendations': {
                'suggested_min_score': float(np.mean(scores) - np.std(scores)),
                'suggested_device_bonus': float(np.mean(device_scores)) if np.mean(device_scores) > 0 else 50,
                'suggested_intro_penalty': float(np.mean(intro_penalties)) if np.mean(intro_penalties) > 0 else 100
            },
            'frames': target_frames
        }
        
        # Save analysis
        with open(os.path.join(output_dir, 'target_analysis.json'), 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"\n🎯 Target Frame Analysis:")
        print(f"   Average score: {patterns['statistics']['score_stats']['mean']:.1f}")
        print(f"   Average device score: {patterns['statistics']['device_score_stats']['mean']:.1f}")
        print(f"   Average intro penalty: {patterns['statistics']['intro_penalty_stats']['mean']:.1f}")
        print(f"\n💡 Recommendations:")
        print(f"   Set minimum score to: {patterns['recommendations']['suggested_min_score']:.1f}")
        print(f"   Device bonus should be: {patterns['recommendations']['suggested_device_bonus']:.1f}")
        print(f"   Intro penalty should be: {patterns['recommendations']['suggested_intro_penalty']:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Test frame selection algorithm')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--mode', choices=['test', 'reverse'], default='test',
                       help='Test mode: test algorithm or reverse engineer from timestamps')
    parser.add_argument('--duration', type=float, default=35.0,
                       help='Test duration in seconds (default: 35)')
    parser.add_argument('--timestamps', type=str,
                       help='Comma-separated timestamps for reverse engineering (e.g., "5.2,10.1,15.8")')
    parser.add_argument('--output', default='frame_test_output',
                       help='Output directory (default: frame_test_output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    tester = FrameSelectionTester()
    
    if args.mode == 'test':
        tester.test_selection(args.video, args.duration, args.output)
    
    elif args.mode == 'reverse':
        if not args.timestamps:
            print("❌ --timestamps required for reverse engineering mode")
            print("   Example: --timestamps '5.2,10.1,15.8'")
            sys.exit(1)
        
        try:
            timestamps = [float(t.strip()) for t in args.timestamps.split(',')]
            tester.reverse_engineer(args.video, timestamps, args.output)
        except ValueError:
            print("❌ Invalid timestamp format. Use comma-separated numbers like '5.2,10.1,15.8'")
            sys.exit(1)

if __name__ == "__main__":
    main()