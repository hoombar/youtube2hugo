#!/usr/bin/env python3
"""Debug script to check frame timing and section alignment."""

import subprocess
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_video_timing(video_path):
    """Debug the video timing and frame extraction."""
    
    print(f"🔍 Debugging video: {video_path}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get video duration
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'csv=p=0', 
            '-select_streams', 'v:0', '-show_entries', 'format=duration', 
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        print(f"📹 Video duration: {duration:.1f} seconds")
    except Exception as e:
        print(f"❌ Failed to get video duration: {e}")
        return
    
    # Test frame extraction timing
    print("\n🎬 Testing frame extraction timing...")
    frame_times = []
    current_time = 0.0
    while current_time <= min(duration, 120):  # Only test first 2 minutes
        frame_times.append(current_time)
        current_time += 1.0
    
    print(f"📊 Would extract {len(frame_times)} frames")
    print(f"📊 Frame timestamps: {frame_times[:10]}... (showing first 10)")
    
    # Test section creation (simulate what happens in create_blog_session)
    print("\n📑 Testing section creation...")
    
    # Simulate 5 sections distributed over video duration
    num_sections = 5
    section_duration = duration / num_sections
    
    sections = []
    for i in range(num_sections):
        start_time = i * section_duration
        end_time = min((i + 1) * section_duration, duration)
        sections.append({
            'title': f'Section {i+1}',
            'start_time': start_time,
            'end_time': end_time
        })
    
    print(f"📑 Created {len(sections)} sections:")
    for i, section in enumerate(sections):
        print(f"   Section {i+1}: {section['start_time']:.1f}s - {section['end_time']:.1f}s")
    
    # Test frame-to-section matching
    print("\n🎯 Testing frame-to-section matching...")
    for section_idx, section in enumerate(sections):
        matching_frames = []
        for frame_time in frame_times:
            if section['start_time'] <= frame_time <= section['end_time']:
                matching_frames.append(frame_time)
        
        print(f"   Section {section_idx+1} ({section['start_time']:.1f}s-{section['end_time']:.1f}s): {len(matching_frames)} frames")
        if matching_frames:
            print(f"     Frame times: {matching_frames[:5]}... (showing first 5)")
        else:
            print(f"     ❌ NO FRAMES FOUND!")
            # Find closest frames
            closest_frames = sorted(frame_times, key=lambda x: min(abs(x - section['start_time']), abs(x - section['end_time'])))[:3]
            print(f"     Closest frames: {closest_frames}")

if __name__ == "__main__":
    video_path = "/Users/ben/Documents/YouTube videos raw/LLMVision doorbell/0601/0601.mov"
    debug_video_timing(video_path)