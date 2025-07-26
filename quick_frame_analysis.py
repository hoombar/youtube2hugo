#!/usr/bin/env python3
"""
Quick Frame Analysis Tool

A faster analysis tool that focuses on the current semantic frame selection
and provides immediate insights into why frames are being filtered out.
"""

import os
import sys
import logging
import click
from typing import Dict, List

from config import Config
from video_processor import VideoProcessor
from transcript_extractor import TranscriptExtractor
from semantic_frame_selector import SemanticFrameSelector
from blog_formatter import BlogFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_analysis(video_path: str, title: str, config_dict: Dict):
    """Run quick analysis of frame selection pipeline."""
    
    print(f"\n🔍 QUICK FRAME SELECTION ANALYSIS")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Title: {title}")
    print("=" * 60)
    
    # Initialize components
    video_processor = VideoProcessor(config_dict)
    transcript_extractor = TranscriptExtractor(config_dict)
    semantic_frame_selector = SemanticFrameSelector(config_dict, video_processor)
    blog_formatter = BlogFormatter(config_dict)
    
    # Step 1: Extract transcript
    print("\n📝 TRANSCRIPT ANALYSIS")
    transcript_segments = transcript_extractor.extract_transcript(video_path)
    total_duration = max(seg.get('end', 0) for seg in transcript_segments) if transcript_segments else 0
    
    print(f"   ✓ Segments: {len(transcript_segments)}")
    print(f"   ✓ Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"   ✓ Sample text: {transcript_segments[0].get('text', 'N/A')[:100]}..." if transcript_segments else "   ❌ No transcript")
    
    # Step 2: Blog formatting and boundary analysis
    print("\n📖 BLOG FORMATTING ANALYSIS")
    formatted_blog_content = blog_formatter.format_transcript_content(transcript_segments, title)
    boundary_map = getattr(blog_formatter, 'boundary_map', {})
    
    # Extract headers manually to compare
    import re
    headers = []
    for line in formatted_blog_content.split('\n'):
        match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if match:
            headers.append(match.group(2).strip())
    
    print(f"   ✓ Blog length: {len(formatted_blog_content)} characters")
    print(f"   ✓ Headers found: {len(headers)}")
    print(f"   ✓ Boundary markers: {len(boundary_map)}")
    print(f"   ✓ Mapping success: {len(boundary_map)}/{len(headers)} ({len(boundary_map)/len(headers)*100:.1f}%)")
    
    if boundary_map:
        print("   📍 Section timestamps:")
        for section, timestamp in sorted(boundary_map.items(), key=lambda x: x[1]):
            print(f"      {timestamp:6.1f}s - {section}")
    else:
        print("   ❌ No boundary markers found!")
    
    # Step 3: Candidate frame extraction
    print("\n🎬 FRAME EXTRACTION ANALYSIS")
    temp_dir = "temp_analysis_frames"
    
    try:
        # Try to get candidate frames info (this would require modifying the semantic selector to expose this)
        print("   ℹ️  Running semantic frame selection to analyze...")
        
        semantic_frames = semantic_frame_selector.select_frames_from_blog_content(
            video_path, transcript_segments, formatted_blog_content, temp_dir, title, blog_formatter
        )
        
        print(f"   ✓ Final frames selected: {len(semantic_frames)}")
        
        if semantic_frames:
            print(f"   ✓ Frame timestamps:")
            for i, frame in enumerate(semantic_frames):
                timestamp = frame['timestamp']
                score = frame.get('semantic_score', 0)
                section = frame.get('section_title', 'Unknown')
                print(f"      {i+1:2d}. {timestamp:6.1f}s (score: {score:4.1f}) - {section}")
            
            # Analyze temporal distribution
            timestamps = [f['timestamp'] for f in semantic_frames]
            if len(timestamps) > 1:
                gaps = []
                for i in range(len(timestamps) - 1):
                    gap = timestamps[i + 1] - timestamps[i]
                    gaps.append(gap)
                
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(gaps)
                
                print(f"   📊 Distribution:")
                print(f"      Average gap: {avg_gap:.1f}s")
                print(f"      Largest gap: {max_gap:.1f}s")
                print(f"      Coverage: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s ({timestamps[-1] - timestamps[0]:.1f}s)")
        else:
            print("   ❌ NO FRAMES SELECTED!")
    
    except Exception as e:
        print(f"   ❌ Frame selection failed: {e}")
    
    # Step 4: Current configuration analysis
    print("\n⚙️ CONFIGURATION ANALYSIS")
    semantic_config = config_dict.get('semantic_frame_selection', {})
    
    current_threshold = semantic_config.get('score_threshold', 50.0)
    base_weight = semantic_config.get('base_score_weight', 0.3)
    text_weight = semantic_config.get('text_score_weight', 0.4)
    visual_weight = semantic_config.get('visual_score_weight', 0.3)
    
    print(f"   ✓ Score threshold: {current_threshold}")
    print(f"   ✓ Weights: base={base_weight}, text={text_weight}, visual={visual_weight}")
    
    # Step 5: Quick recommendations
    print("\n💡 QUICK RECOMMENDATIONS")
    
    if len(boundary_map) < len(headers) * 0.7:
        print("   🔴 CRITICAL: Poor boundary mapping. Consider:")
        print("      - Check if Gemini is preserving timestamp markers")
        print("      - Verify transcript quality")
        print("      - Review section titles vs transcript content")
    
    if len(semantic_frames) < 5:
        print(f"   🔴 CRITICAL: Only {len(semantic_frames)} frames selected. Consider:")
        print(f"      - Lower score_threshold from {current_threshold} to 35-40")
        print("      - Increase base_score_weight to 0.4-0.5")
        print("      - Check if sections are too short")
    elif len(semantic_frames) > 20:
        print(f"   🟡 WARNING: {len(semantic_frames)} frames may be too many. Consider:")
        print(f"      - Raise score_threshold from {current_threshold} to 50-55")
        print("      - Decrease base_score_weight")
    else:
        print(f"   ✅ Frame count looks good: {len(semantic_frames)} frames")
    
    # Analyze gaps
    if semantic_frames and len(semantic_frames) > 1:
        timestamps = [f['timestamp'] for f in semantic_frames]
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        max_gap = max(gaps)
        
        if max_gap > 120:  # 2 minutes
            print(f"   🟡 Large gap detected: {max_gap:.1f}s. Video may have uncovered sections.")
    
    print("\n" + "=" * 60)
    print("💡 To run full analysis: python frame_selection_analyzer.py --video [path] --title [title]")
    
    # Cleanup
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

@click.command()
@click.option('--video', '-v', required=True, help='Path to video file')
@click.option('--title', '-t', required=True, help='Video title')
@click.option('--config', '-c', help='Path to configuration file')
def analyze(video, title, config):
    """Quick analysis of frame selection pipeline."""
    
    # Load configuration
    config_dict = Config.get_default_config()
    
    if config and os.path.exists(config):
        custom_config = Config.load_from_file(config)
        config_dict.update(custom_config)
    
    # Check requirements
    if not config_dict.get('gemini_api_key') and not os.environ.get('GOOGLE_API_KEY'):
        click.echo("❌ Error: No Gemini API key found.", err=True)
        sys.exit(1)
    
    if not os.path.exists(video):
        click.echo(f"❌ Error: Video file not found: {video}", err=True)
        sys.exit(1)
    
    try:
        quick_analysis(video, title, config_dict)
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    analyze()