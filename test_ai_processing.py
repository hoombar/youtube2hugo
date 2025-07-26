#!/usr/bin/env python3
"""
Simple test script to verify AI processing works independently of frame processing mode.
"""

import yaml
import logging
from transcript_extractor import TranscriptExtractor
from blog_formatter import BlogFormatter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_ai_processing():
    """Test AI processing with a sample video."""
    
    # Load config
    with open('config.local.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with a sample video (you'll need to provide a real video path)
    video_path = input("Enter a video file path to test: ").strip()
    if not video_path:
        print("No video path provided, exiting.")
        return
    
    title = "Test Blog Post"
    
    print(f"🎬 Testing AI processing with: {video_path}")
    
    # Step 1: Extract transcript
    print("📝 Extracting transcript...")
    transcript_extractor = TranscriptExtractor(config)
    transcript_segments = transcript_extractor.extract_transcript(video_path)
    print(f"✅ Got {len(transcript_segments)} transcript segments")
    
    # Step 2: Test AI processing
    print("🤖 Testing AI content processing...")
    blog_formatter = BlogFormatter(config)
    
    try:
        blog_content = blog_formatter.format_transcript_content(transcript_segments, title)
        boundary_map = getattr(blog_formatter, 'boundary_map', {})
        
        print(f"✅ AI processing succeeded!")
        print(f"📝 Generated content length: {len(blog_content)} chars")
        print(f"📍 Generated {len(boundary_map)} boundaries: {list(boundary_map.keys())}")
        print(f"📄 Content preview (first 300 chars):")
        print("-" * 50)
        print(blog_content[:300])
        print("-" * 50)
        
        # Check if content looks like blog or transcript
        has_headers = "##" in blog_content
        has_transcript_markers = any(word in blog_content.lower() for word in ["right,", "so,", "now,", "let's"])
        
        print(f"✅ Blog structure analysis:")
        print(f"   - Has ## headers: {has_headers}")
        print(f"   - Has transcript markers: {has_transcript_markers}")
        
        if has_headers and not has_transcript_markers:
            print("✅ Content looks like properly formatted blog post!")
        elif has_transcript_markers:
            print("⚠️  Content still has transcript-like language")
        else:
            print("❌ Content doesn't have proper blog structure")
        
        return True
        
    except Exception as e:
        print(f"❌ AI processing failed: {e}")
        return False

if __name__ == "__main__":
    test_ai_processing()