#!/usr/bin/env python3
"""Test the actual frame extraction to see where it stops."""

import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_frame_extraction(video_path):
    """Test frame extraction with the actual ffmpeg commands."""
    
    # Create temp directory
    output_dir = "test_frames"
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get video duration
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'csv=p=0', 
            '-select_streams', 'v:0', '-show_entries', 'format=duration', 
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        print(f"📹 Video duration: {duration:.1f} seconds")
        
        # Generate frame times (same logic as hybrid_blog_creator.py)
        frame_times = []
        current_time = 0.0
        while current_time <= duration:
            frame_times.append(current_time)
            current_time += 1.0
        
        print(f"📊 Will extract {len(frame_times)} frames from 0s to {frame_times[-1]:.1f}s")
        
        # Test extracting first 10 frames to see actual ffmpeg behavior
        frames_to_test = frame_times[:10]  # Test first 10 frames only
        successful_frames = []
        
        for i, timestamp in enumerate(frames_to_test):
            filename = f"frame_{i:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Use same ffmpeg command as hybrid_blog_creator.py
            cmd = [
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1',  # Extract exactly 1 frame
                '-q:v', '2',      # High quality
                '-an',            # No audio
                filepath
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Verify the frame was actually created
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    successful_frames.append((timestamp, filename, file_size))
                    print(f"✅ Frame {i+1}: {timestamp:.1f}s -> {filename} ({file_size} bytes)")
                else:
                    print(f"❌ Frame {i+1}: {timestamp:.1f}s -> {filename} NOT CREATED")
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Frame {i+1}: {timestamp:.1f}s -> ffmpeg failed: {e}")
                print(f"   stderr: {e.stderr}")
        
        print(f"\n📊 Summary: {len(successful_frames)}/{len(frames_to_test)} frames extracted successfully")
        
        # Now test a frame much later in the video
        print(f"\n🔍 Testing frame at 200s (middle of video)...")
        test_timestamp = 200.0
        test_filename = "test_frame_200s.jpg"
        test_filepath = os.path.join(output_dir, test_filename)
        
        cmd = [
            'ffmpeg', '-y', '-ss', str(test_timestamp), '-i', video_path,
            '-vframes', '1', '-q:v', '2', '-an', test_filepath
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if os.path.exists(test_filepath):
                file_size = os.path.getsize(test_filepath)
                print(f"✅ Test frame at 200s: SUCCESS ({file_size} bytes)")
            else:
                print(f"❌ Test frame at 200s: FILE NOT CREATED")
        except subprocess.CalledProcessError as e:
            print(f"❌ Test frame at 200s: ffmpeg failed: {e}")
            print(f"   stderr: {e.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    video_path = "/Users/ben/Documents/YouTube videos raw/LLMVision doorbell/0601/0601.mov"
    test_frame_extraction(video_path)