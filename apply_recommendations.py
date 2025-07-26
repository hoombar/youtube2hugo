#!/usr/bin/env python3
"""
Apply frame selection recommendations to config.local.yaml

Usage:
    python apply_recommendations.py --video-id 0713
"""

import argparse
import json
import yaml
import os
import shutil
from datetime import datetime

def apply_recommendations(video_id: str):
    """Apply frame selection recommendations to config file."""
    
    # Load training data
    training_dir = os.path.join("training_data", video_id)
    selections_file = os.path.join(training_dir, "manual_selections.json")
    
    if not os.path.exists(selections_file):
        print(f"❌ No training data found for video ID: {video_id}")
        return
    
    print(f"📊 Loading training data for video: {video_id}")
    
    # Re-run analysis to get latest recommendations
    from frame_selection_trainer import FrameSelectionTrainer
    trainer = FrameSelectionTrainer()
    
    try:
        analysis = trainer.analyze_training_data(video_id)
        detailed = analysis.get('detailed_analysis', {})
        actionable_changes = detailed.get('actionable_changes', [])
        
        if not actionable_changes:
            print("✅ No config changes recommended")
            return
        
        # Backup current config
        config_file = "config.local.yaml"
        backup_file = f"config.local.yaml.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(config_file, backup_file)
        print(f"💾 Backed up config to: {backup_file}")
        
        # Load current config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply config updates
        changes_applied = []
        for change in actionable_changes:
            if change['type'] == 'config_update':
                section = change['section']
                changes = change['changes']
                
                # Ensure section exists
                if section not in config:
                    config[section] = {}
                
                # Apply changes
                for key, value in changes.items():
                    old_value = config[section].get(key)
                    config[section][key] = value
                    changes_applied.append(f"  {section}.{key}: {old_value} → {value}")
                    print(f"✅ Updated {section}.{key}: {old_value} → {value}")
        
        if changes_applied:
            # Save updated config
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print(f"\n🎯 Applied {len(changes_applied)} configuration changes:")
            for change in changes_applied:
                print(change)
            
            print(f"\n💡 Config updated successfully!")
            print(f"📁 Backup saved as: {backup_file}")
            print(f"\n🚀 Run your video processing again to test the improvements!")
        else:
            print("✅ No config changes needed")
            
    except Exception as e:
        print(f"❌ Error applying recommendations: {e}")

def list_available_videos():
    """List available training videos."""
    training_dir = "training_data"
    if not os.path.exists(training_dir):
        print("❌ No training data directory found")
        return
    
    videos = []
    for item in os.listdir(training_dir):
        item_path = os.path.join(training_dir, item)
        if os.path.isdir(item_path):
            selections_file = os.path.join(item_path, "manual_selections.json")
            if os.path.exists(selections_file):
                videos.append(item)
    
    if videos:
        print("📹 Available trained videos:")
        for video in videos:
            print(f"  - {video}")
    else:
        print("❌ No trained videos found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply frame selection training recommendations")
    parser.add_argument("--video-id", help="Video ID to apply recommendations from")
    parser.add_argument("--list", action="store_true", help="List available trained videos")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_videos()
    elif args.video_id:
        apply_recommendations(args.video_id)
    else:
        parser.print_help()