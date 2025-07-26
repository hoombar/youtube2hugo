#!/usr/bin/env python3
"""
Simple launcher for the Frame Selection Trainer.

Usage:
    python train_frame_selection.py

Then open your browser to http://127.0.0.1:5000
"""

from frame_selection_trainer import FrameSelectionTrainer

if __name__ == "__main__":
    print("🎯 Starting Frame Selection Trainer...")
    print("📝 This tool will help you:")
    print("   1. Process your video and extract candidate frames")
    print("   2. Let you manually select the best frames")
    print("   3. Analyze your selections to improve the algorithm")
    print()
    print("🌐 Opening web interface at http://127.0.0.1:5000")
    print("💡 Tip: Have your video file path ready!")
    print()
    
    trainer = FrameSelectionTrainer()
    trainer.run()