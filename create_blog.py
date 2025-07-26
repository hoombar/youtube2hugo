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
    print("   • Optimized frame extraction with perceptual hashing") 
    print("   • Manual frame selection with consistent image sizing")
    print("   • Automatic blog post generation with selected frames")
    print()
    print("⚡ Performance features:")
    print("   • Fast duplicate removal using perceptual hashing")
    print("   • Adaptive sampling for static scenes")
    print("   • Intelligent frame caching to reduce I/O")
    print("   • Multi-strategy AI prompting to work around safety filters")
    print("   • Enhanced fallback content generation when AI processing fails")
    print()
    print("🌐 Opening web interface at http://127.0.0.1:5002")
    print("💡 Have your video file path and desired title ready!")
    print()
    
    creator = HybridBlogCreator()
    creator.run(port=5002)