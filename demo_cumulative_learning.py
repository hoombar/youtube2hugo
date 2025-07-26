#!/usr/bin/env python3
"""
Demo script showing cumulative learning behavior across multiple videos.

This demonstrates how the cumulative trainer preserves and combines insights
from multiple training sessions rather than replacing them.
"""

import json
import numpy as np
from cumulative_trainer import CumulativeTrainer

def simulate_additional_training_data():
    """Create simulated training data for a second video to demonstrate cumulative learning."""
    
    # Simulate a different video with complementary patterns
    simulated_data = {
        "video_id": "0714_demo",
        "timestamp": "2025-07-25T15:20:00.000000",
        "selected_count": 18,
        "rejected_count": 142,
        "patterns": {
            # Different but complementary preferences
            "has_ha_colors_preference": 0.15,  # Opposite to first video (-0.29)
            "has_ui_elements_preference": 0.22,  # Opposite to first video (-0.12)
            "has_screen_content_preference": 0.18,  # Opposite to first video (-0.05)
            "has_text_preference": 0.08,  # Similar to first video (0.11)
            "selected_means": {
                "brightness": 185.2,  # Lower than first video (202.5)
                "contrast": 58.1,     # Higher than first video (50.6)
                "edge_density": 0.12,  # Higher than first video (0.094)
                "text_length": 95.3    # Lower than first video (124.0)
            },
            "rejected_means": {
                "brightness": 158.9,
                "contrast": 48.2,
                "edge_density": 0.085,
                "text_length": 38.7
            },
            "differences": {
                "brightness": 26.3,   # Lower difference than first video (54.7)
                "contrast": 9.9,      # Higher than first video (-2.2)
                "edge_density": 0.035, # Higher than first video (0.0003)
                "text_length": 56.6   # Lower than first video (81.7)
            }
        }
    }
    
    return simulated_data

def demonstrate_cumulative_learning():
    """Demonstrate how cumulative learning combines insights from multiple videos."""
    
    print("🎯 Cumulative Learning Demonstration")
    print("=" * 50)
    
    trainer = CumulativeTrainer()
    
    # Show current state (single video)
    print(f"\n📊 Current State: {trainer.cumulative_data['video_count']} video(s) trained")
    
    if trainer.cumulative_data['video_count'] > 0:
        current_metrics = trainer.cumulative_data['cumulative_metrics']
        print(f"Current brightness preference: {current_metrics['differences']['brightness']:.1f}")
        print(f"Current text preference: {current_metrics['differences']['text_length']:.1f}")
        print(f"Current selection ratio: {current_metrics['selection_ratio']:.1%}")
        
        # Generate current recommendations
        current_recs = trainer.generate_cumulative_recommendations()
        current_weights = current_recs['confidence_weighted_weights']
        print(f"\nCurrent recommended weights:")
        print(f"  Base: {current_weights['base_score_weight']:.3f}")
        print(f"  Text: {current_weights['text_score_weight']:.3f}")
        print(f"  Visual: {current_weights['visual_score_weight']:.3f}")
        print(f"  Overall confidence: {current_weights['confidence_scores']['overall']:.1%}")
    
    # Simulate adding a second video with different patterns
    print(f"\n🎬 Simulating addition of second video with different patterns...")
    simulated_data = simulate_additional_training_data()
    
    # Manually add to show the combination effect
    trainer.cumulative_data['training_sessions'].append(simulated_data)
    trainer.cumulative_data['video_count'] += 1
    trainer._update_cumulative_metrics()
    
    # Show combined state
    print(f"\n📊 After Adding Second Video: {trainer.cumulative_data['video_count']} videos trained")
    combined_metrics = trainer.cumulative_data['cumulative_metrics']
    
    print(f"\n🔄 How metrics combined:")
    print(f"Brightness preference: {current_metrics['differences']['brightness']:.1f} + {simulated_data['patterns']['differences']['brightness']:.1f} → {combined_metrics['differences']['brightness']:.1f}")
    print(f"Text preference: {current_metrics['differences']['text_length']:.1f} + {simulated_data['patterns']['differences']['text_length']:.1f} → {combined_metrics['differences']['text_length']:.1f}")
    print(f"Selection ratio: {current_metrics['selection_ratio']:.1%} + {(simulated_data['selected_count']/(simulated_data['selected_count']+simulated_data['rejected_count'])):.1%} → {combined_metrics['selection_ratio']:.1%}")
    
    # Generate new recommendations
    new_recs = trainer.generate_cumulative_recommendations()
    new_weights = new_recs['confidence_weighted_weights']
    
    print(f"\n⚖️ Weight changes after combining videos:")
    print(f"  Base: {current_weights['base_score_weight']:.3f} → {new_weights['base_score_weight']:.3f}")
    print(f"  Text: {current_weights['text_score_weight']:.3f} → {new_weights['text_score_weight']:.3f}")
    print(f"  Visual: {current_weights['visual_score_weight']:.3f} → {new_weights['visual_score_weight']:.3f}")
    print(f"  Overall confidence: {current_weights['confidence_scores']['overall']:.1%} → {new_weights['confidence_scores']['overall']:.1%}")
    
    # Show confidence improvement
    print(f"\n📈 Confidence Improvements:")
    current_conf = trainer.cumulative_data['confidence_scores']
    for metric in current_conf:
        print(f"  {metric}: {current_conf[metric]:.1%} (higher confidence with more data)")
    
    print(f"\n🎯 Key Insights:")
    print(f"  • Values are COMBINED, not replaced")
    print(f"  • Confidence increases with more training data")
    print(f"  • Conflicting patterns are balanced (brightness: strong + weak = moderate)")
    print(f"  • Consistent patterns are reinforced (text preference maintained)")
    print(f"  • System learns from diverse video types without losing previous knowledge")
    
    # Restore original state
    trainer.cumulative_data['training_sessions'].pop()
    trainer.cumulative_data['video_count'] -= 1
    trainer._update_cumulative_metrics()
    
    print(f"\n✅ Demonstration complete - original state restored")

if __name__ == "__main__":
    demonstrate_cumulative_learning()