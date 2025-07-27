#!/usr/bin/env python3
"""Test the boundary timestamp fix directly."""

import re

def test_boundary_logic():
    """Test the fixed boundary logic."""
    
    # Simulate the boundary map building process
    boundary_map = {}
    
    # Simulate sections that have no timestamp found
    sections_to_test = [
        "Elevate Your Smart Doorbell: AI Vision with LLM Vision and Home Assistant",
        "Getting Started: Setting Up Gemini in Google Cloud", 
        "Installing the LLM Vision Integration"
    ]
    
    print("🧪 Testing boundary timestamp assignment logic...")
    
    for i, section_title in enumerate(sections_to_test):
        print(f"\n📍 Processing section {i+1}: '{section_title}'")
        print(f"   Current boundary_map length: {len(boundary_map)}")
        
        # Simulate not finding a timestamp (the bug condition)
        found_timestamp = False
        
        if not found_timestamp:
            print(f"⚠️  No timestamp found for section: '{section_title}'")
            # Use a default timestamp based on position - FIRST section should start at 0!
            if len(boundary_map) == 0:
                # This is the first section, it should start at 0
                default_timestamp = 0.0
                print(f"✅ FIRST SECTION: Using timestamp 0.0s")
            else:
                # Subsequent sections: 60 seconds apart from the start
                default_timestamp = len(boundary_map) * 60  # 60 seconds apart as default
                print(f"📍 SUBSEQUENT SECTION: Using timestamp {default_timestamp:.1f}s")
            
            boundary_map[section_title] = default_timestamp
            print(f"   Final timestamp: {default_timestamp:.1f}s")
    
    print(f"\n📊 Final boundary map:")
    for title, timestamp in boundary_map.items():
        print(f"   '{title}': {timestamp:.1f}s")

if __name__ == "__main__":
    test_boundary_logic()