#!/usr/bin/env python3
"""Debug script to test actual session creation timing."""

import yaml
import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, '/Users/ben/dev/youtube2hugo')

from transcript_extractor import TranscriptExtractor
from blog_formatter import BlogFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_session_creation(video_path):
    """Debug the actual session creation process."""
    
    print(f"🔍 Debugging session creation for: {video_path}")
    
    # Load config
    config_path = "config.local.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Extract transcript
    print("\n📝 Step 1: Extracting transcript...")
    transcript_extractor = TranscriptExtractor(config)
    transcript_segments = transcript_extractor.extract_transcript(video_path)
    
    if transcript_segments:
        total_duration = transcript_segments[-1]['end_time']
        print(f"✅ Transcript extracted: {len(transcript_segments)} segments, {total_duration:.1f}s total")
        print(f"📊 First segment: {transcript_segments[0]['start_time']:.1f}s - {transcript_segments[0]['end_time']:.1f}s")
        print(f"📊 Last segment: {transcript_segments[-1]['start_time']:.1f}s - {transcript_segments[-1]['end_time']:.1f}s")
    else:
        print("❌ No transcript extracted")
        return
    
    # Step 2: Test AI processing
    print("\n🤖 Step 2: Testing AI processing...")
    blog_formatter = BlogFormatter(config)
    
    title = "Debug Test Video"
    
    try:
        # This might fail due to safety filters - that's what we want to test
        blog_content = blog_formatter.format_transcript_content(transcript_segments, title)
        print(f"✅ AI processing succeeded! Content length: {len(blog_content)} chars")
        
        # Test section extraction
        boundary_map = getattr(blog_formatter, 'boundary_map', {})
        print(f"📍 AI generated {len(boundary_map)} boundaries: {list(boundary_map.keys())}")
        
    except Exception as e:
        print(f"❌ AI processing failed: {e}")
        print("🔄 This means the system will fall back to basic sections...")
        
        # Test the fallback section creation
        print("\n🔄 Testing fallback section creation...")
        
        # Simulate enhanced basic sections
        sections = []
        current_section_segments = []
        current_section_duration = 0
        target_section_duration = 90  # 1.5 minutes - THIS MIGHT BE THE ISSUE
        
        for segment in transcript_segments:
            segment_duration = segment['end_time'] - segment['start_time']
            current_section_segments.append(segment)
            current_section_duration += segment_duration
            
            if current_section_duration >= target_section_duration and len(current_section_segments) >= 3:
                start_time = current_section_segments[0]['start_time']
                end_time = current_section_segments[-1]['end_time']
                sections.append({
                    'title': f'Section {len(sections) + 1}',
                    'start_time': start_time,
                    'end_time': end_time,
                    'content': f'Section content from {start_time:.1f}s to {end_time:.1f}s'
                })
                print(f"📑 Created section {len(sections)}: {start_time:.1f}s - {end_time:.1f}s")
                current_section_segments = []
                current_section_duration = 0
        
        # Handle remaining segments
        if current_section_segments:
            start_time = current_section_segments[0]['start_time']
            end_time = current_section_segments[-1]['end_time']
            sections.append({
                'title': f'Section {len(sections) + 1}',
                'start_time': start_time,
                'end_time': end_time,
                'content': f'Section content from {start_time:.1f}s to {end_time:.1f}s'
            })
            print(f"📑 Created final section {len(sections)}: {start_time:.1f}s - {end_time:.1f}s")
        
        print(f"\n📊 Fallback section summary:")
        print(f"   Total sections: {len(sections)}")
        if sections:
            video_coverage = sections[-1]['end_time']
            coverage_percent = (video_coverage / total_duration) * 100
            print(f"   Video coverage: {video_coverage:.1f}s / {total_duration:.1f}s ({coverage_percent:.1f}%)")
            
            if coverage_percent < 90:
                print(f"❌ PROBLEM FOUND: Sections only cover {coverage_percent:.1f}% of video!")
                print(f"   This explains why frames after {video_coverage:.1f}s are missing!")
            else:
                print(f"✅ Good coverage: {coverage_percent:.1f}% of video")

if __name__ == "__main__":
    video_path = "/Users/ben/Documents/YouTube videos raw/LLMVision doorbell/0601/0601.mov"
    debug_session_creation(video_path)