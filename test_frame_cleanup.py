#!/usr/bin/env python3
"""
Test script to verify frame cleanup functionality.
"""

import os
import tempfile
import shutil
from frame_selection_trainer import FrameSelectionTrainer

def test_frame_cleanup():
    """Test that frame cleanup properly clears old frames."""
    
    print("🧪 Testing frame cleanup functionality...")
    
    # Create trainer instance
    trainer = FrameSelectionTrainer()
    
    # Create temp_frames directory with some fake files
    temp_frames_dir = "temp_frames"
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    # Add some fake frame files
    test_files = [
        "frame_10.0s.jpg",
        "frame_20.0s.jpg", 
        "old_candidate_1.jpg",
        "leftover_frame.png"
    ]
    
    for filename in test_files:
        filepath = os.path.join(temp_frames_dir, filename)
        with open(filepath, 'w') as f:
            f.write("fake image data")
    
    print(f"📁 Created {len(test_files)} fake frame files")
    
    # Test the cleanup method
    print("🧹 Testing cleanup method...")
    trainer._clear_temp_frames(temp_frames_dir)
    
    # Verify cleanup worked
    if os.path.exists(temp_frames_dir):
        remaining_files = os.listdir(temp_frames_dir)
        if remaining_files:
            print(f"❌ Cleanup failed - {len(remaining_files)} files remain: {remaining_files}")
            return False
        else:
            print("✅ Directory exists but is empty - partial success")
    else:
        print("✅ Directory completely removed - full success")
    
    print("✅ Frame cleanup test passed!")
    return True

if __name__ == "__main__":
    test_frame_cleanup()