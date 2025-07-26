#!/usr/bin/env python3
"""
Debug Boundary Markers

This script helps debug why boundary markers are all mapping to 0.0s
"""

import os
import sys
import logging
import click
from typing import Dict, List

from config import Config
from transcript_extractor import TranscriptExtractor
from blog_formatter import BlogFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_boundary_markers(video_path: str, title: str, config_dict: Dict):
    """Debug the boundary marker system step by step."""
    
    print("\n🔍 BOUNDARY MARKER DEBUG")
    print("=" * 50)
    
    # Initialize components
    transcript_extractor = TranscriptExtractor(config_dict)
    blog_formatter = BlogFormatter(config_dict)
    
    # Step 1: Extract transcript
    print("\n📝 Step 1: Extract Transcript")
    transcript_segments = transcript_extractor.extract_transcript(video_path)
    
    print(f"   ✓ Segments: {len(transcript_segments)}")
    if transcript_segments:
        print(f"   ✓ First segment: {transcript_segments[0].get('start', 0):.1f}s - {transcript_segments[0].get('text', '')[:50]}...")
        print(f"   ✓ Last segment: {transcript_segments[-1].get('end', 0):.1f}s - {transcript_segments[-1].get('text', '')[:50]}...")
    
    # Step 2: Create text with markers
    print("\n🏷️  Step 2: Create Text with Timestamp Markers")
    raw_content_with_markers = blog_formatter._transcript_segments_to_text_with_markers(transcript_segments)
    
    print(f"   ✓ Content length: {len(raw_content_with_markers)} characters")
    print(f"   ✓ First 200 chars: {raw_content_with_markers[:200]}...")
    
    # Count markers
    import re
    marker_pattern = r'__TIMESTAMP_(\d+\.\d+)__'
    markers_in_raw = re.findall(marker_pattern, raw_content_with_markers)
    print(f"   ✓ Markers found: {len(markers_in_raw)}")
    print(f"   ✓ Sample markers: {markers_in_raw[:5]}...")
    
    # Step 3: Apply technical corrections
    print("\n🔧 Step 3: Apply Technical Corrections")
    corrected_content = blog_formatter._apply_technical_corrections(raw_content_with_markers)
    
    markers_after_corrections = re.findall(marker_pattern, corrected_content)
    print(f"   ✓ Markers after corrections: {len(markers_after_corrections)}")
    if len(markers_after_corrections) != len(markers_in_raw):
        print("   ❌ MARKERS LOST DURING CORRECTIONS!")
    
    # Step 4: Format with Gemini (with boundary preservation)
    print("\n🤖 Step 4: Format with Gemini (preserving boundaries)")
    formatted_content_with_markers = blog_formatter._format_as_blog_post_with_boundaries(corrected_content, title)
    
    print(f"   ✓ Formatted length: {len(formatted_content_with_markers)} characters")
    print(f"   ✓ First 300 chars:\n{formatted_content_with_markers[:300]}...")
    
    markers_after_gemini = re.findall(marker_pattern, formatted_content_with_markers)
    print(f"   ✓ Markers after Gemini: {len(markers_after_gemini)}")
    if len(markers_after_gemini) != len(markers_in_raw):
        print("   ❌ MARKERS LOST DURING GEMINI FORMATTING!")
        print(f"   ❌ Lost {len(markers_in_raw) - len(markers_after_gemini)} markers")
    
    print(f"   ✓ Sample markers after Gemini: {markers_after_gemini[:5]}...")
    
    # Step 5: Extract and clean boundaries
    print("\n🎯 Step 5: Extract and Clean Boundaries")
    formatted_content, boundary_map = blog_formatter._extract_and_clean_boundaries(formatted_content_with_markers)
    
    print(f"   ✓ Clean content length: {len(formatted_content)} characters")
    print(f"   ✓ Boundary map: {boundary_map}")
    
    # Step 6: Analyze what went wrong
    print("\n🔍 Step 6: Analysis")
    
    if len(markers_in_raw) == 0:
        print("   ❌ PROBLEM: No markers created in step 2")
    elif len(markers_after_corrections) < len(markers_in_raw):
        print("   ❌ PROBLEM: Markers lost during technical corrections")
    elif len(markers_after_gemini) < len(markers_after_corrections):
        print("   ❌ PROBLEM: Gemini is removing/ignoring timestamp markers")
        print("   💡 SOLUTION: Need to improve Gemini prompt to preserve markers")
    elif len(boundary_map) == 0:
        print("   ❌ PROBLEM: Boundary extraction failed")
    elif all(timestamp == 0.0 for timestamp in boundary_map.values()):
        print("   ❌ PROBLEM: All boundaries map to 0.0s")
        print("   💡 SOLUTION: Boundary extraction logic needs fixing")
    else:
        print("   ✅ Boundary system appears to work correctly")
    
    # Step 7: Show sample formatted content with markers (for debugging)
    print("\n📋 Step 7: Sample Formatted Content with Markers")
    lines = formatted_content_with_markers.split('\n')[:20]
    for i, line in enumerate(lines):
        if re.search(marker_pattern, line) or line.startswith('#'):
            print(f"   {i:2d}: {line}")
    
    print("\n" + "=" * 50)

@click.command()
@click.option('--video', '-v', required=True, help='Path to video file')
@click.option('--title', '-t', required=True, help='Video title')
@click.option('--config', '-c', help='Path to configuration file')
def debug(video, title, config):
    """Debug boundary marker system."""
    
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
        debug_boundary_markers(video, title, config_dict)
    except Exception as e:
        click.echo(f"❌ Debug failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    debug()