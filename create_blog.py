#!/usr/bin/env python3
"""
Simple launcher for the Hybrid Blog Creator.

Usage:
    python create_blog.py

Then open your browser to http://127.0.0.1:5001
"""

from hybrid_blog_creator import HybridBlogCreator

if __name__ == "__main__":
    print("🎬 Starting Hybrid Blog Creator...")
    print("📝 This tool combines:")
    print("   • AI-powered transcript cleaning and section generation")
    print("   • Smart frame extraction for each section") 
    print("   • Manual frame selection with visual interface")
    print("   • Automatic blog post generation with selected frames")
    print()
    print("🌐 Opening web interface at http://127.0.0.1:5002")
    print("💡 Have your video file path and desired title ready!")
    print()
    
    creator = HybridBlogCreator()
    creator.run(port=5002)