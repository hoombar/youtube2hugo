#!/usr/bin/env python3
"""
Cumulative Frame Selection Trainer - Combines insights from multiple videos.

This system builds cumulative knowledge across training sessions instead of
replacing previous learnings with each new video.
"""

import os
import json
import yaml
import numpy as np
from typing import Dict, List
from datetime import datetime
from frame_selection_trainer import FrameSelectionTrainer

class CumulativeTrainer:
    """Builds cumulative knowledge from multiple video training sessions."""
    
    def __init__(self, config_path: str = "config.local.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_data_dir = "training_data"
        self.cumulative_data_file = os.path.join(self.training_data_dir, "cumulative_insights.json")
        
        # Load existing cumulative data
        self.cumulative_data = self._load_cumulative_data()
    
    def _load_cumulative_data(self) -> Dict:
        """Load existing cumulative training data."""
        if os.path.exists(self.cumulative_data_file):
            with open(self.cumulative_data_file, 'r') as f:
                return json.load(f)
        
        return {
            'training_sessions': [],
            'cumulative_metrics': {},
            'confidence_scores': {},
            'video_count': 0,
            'last_updated': None
        }
    
    def add_training_session(self, video_id: str) -> Dict:
        """Add a new training session to cumulative knowledge."""
        print(f"📊 Adding training session: {video_id}")
        
        # Get analysis for this video
        trainer = FrameSelectionTrainer()
        video_analysis = trainer.analyze_training_data(video_id)
        
        # Extract key insights
        session_data = {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'selected_count': len(video_analysis['selected_frames_analysis']),
            'rejected_count': len(video_analysis['rejected_frames_analysis']),
            'patterns': video_analysis['patterns']
        }
        
        # Add to cumulative data
        self.cumulative_data['training_sessions'].append(session_data)
        self.cumulative_data['video_count'] += 1
        self.cumulative_data['last_updated'] = datetime.now().isoformat()
        
        # Update cumulative metrics
        self._update_cumulative_metrics()
        
        # Save updated data
        self._save_cumulative_data()
        
        return self.cumulative_data
    
    def _update_cumulative_metrics(self):
        """Update cumulative metrics by combining all training sessions."""
        sessions = self.cumulative_data['training_sessions']
        
        if not sessions:
            return
        
        # Combine selected/rejected frame analyses from all sessions
        all_selected_metrics = {}
        all_rejected_metrics = {}
        all_feature_preferences = {}
        
        total_selected = 0
        total_rejected = 0
        
        # Aggregate metrics across all sessions
        for session in sessions:
            patterns = session['patterns']
            selected_means = patterns.get('selected_means', {})
            rejected_means = patterns.get('rejected_means', {})
            
            session_selected = session['selected_count']
            session_rejected = session['rejected_count']
            
            # Weighted aggregation of metrics
            for metric in ['brightness', 'contrast', 'edge_density', 'text_length']:
                if metric in selected_means:
                    if metric not in all_selected_metrics:
                        all_selected_metrics[metric] = []
                    # Weight by number of frames in this session
                    all_selected_metrics[metric].extend([selected_means[metric]] * session_selected)
                
                if metric in rejected_means:
                    if metric not in all_rejected_metrics:
                        all_rejected_metrics[metric] = []
                    all_rejected_metrics[metric].extend([rejected_means[metric]] * session_rejected)
            
            # Aggregate feature preferences
            for feature in ['has_ha_colors', 'has_ui_elements', 'has_screen_content', 'has_text']:
                pref_key = f'{feature}_preference'
                if pref_key in patterns:
                    if feature not in all_feature_preferences:
                        all_feature_preferences[feature] = []
                    # Weight by total frames in session
                    total_frames = session_selected + session_rejected
                    all_feature_preferences[feature].extend([patterns[pref_key]] * total_frames)
            
            total_selected += session_selected
            total_rejected += session_rejected
        
        # Calculate cumulative means and preferences
        cumulative_selected_means = {}
        cumulative_rejected_means = {}
        cumulative_differences = {}
        
        for metric in all_selected_metrics:
            if metric in all_rejected_metrics:
                cumulative_selected_means[metric] = float(np.mean(all_selected_metrics[metric]))
                cumulative_rejected_means[metric] = float(np.mean(all_rejected_metrics[metric]))
                cumulative_differences[metric] = cumulative_selected_means[metric] - cumulative_rejected_means[metric]
        
        cumulative_feature_prefs = {}
        for feature in all_feature_preferences:
            cumulative_feature_prefs[f'{feature}_preference'] = float(np.mean(all_feature_preferences[feature]))
        
        # Store cumulative metrics
        self.cumulative_data['cumulative_metrics'] = {
            'selected_means': cumulative_selected_means,
            'rejected_means': cumulative_rejected_means,
            'differences': cumulative_differences,
            'feature_preferences': cumulative_feature_prefs,
            'total_selected_frames': total_selected,
            'total_rejected_frames': total_rejected,
            'selection_ratio': total_selected / (total_selected + total_rejected) if total_rejected > 0 else 0.5
        }
        
        # Calculate confidence scores (higher confidence with more data)
        self._calculate_confidence_scores()
    
    def _calculate_confidence_scores(self):
        """Calculate confidence scores for each metric based on data volume and consistency."""
        cumulative = self.cumulative_data['cumulative_metrics']
        sessions = self.cumulative_data['training_sessions']
        
        confidence = {}
        
        # Base confidence increases with number of videos and total frames
        video_count = len(sessions)
        total_frames = cumulative.get('total_selected_frames', 0) + cumulative.get('total_rejected_frames', 0)
        
        base_confidence = min(1.0, (video_count * 0.2) + (total_frames * 0.001))
        
        # Metric-specific confidence based on consistency across sessions
        for metric, difference in cumulative.get('differences', {}).items():
            metric_values = []
            for session in sessions:
                session_diff = session['patterns'].get('differences', {}).get(metric)
                if session_diff is not None:
                    metric_values.append(session_diff)
            
            if len(metric_values) > 1:
                # Lower confidence if values are inconsistent (high variance)
                consistency = 1.0 / (1.0 + np.std(metric_values) / max(1.0, abs(np.mean(metric_values))))
                confidence[metric] = min(1.0, base_confidence * consistency)
            else:
                confidence[metric] = base_confidence * 0.5  # Lower confidence with single data point
        
        self.cumulative_data['confidence_scores'] = confidence
    
    def generate_cumulative_recommendations(self) -> Dict:
        """Generate config recommendations based on cumulative data."""
        cumulative = self.cumulative_data['cumulative_metrics']
        confidence = self.cumulative_data['confidence_scores']
        
        if not cumulative:
            return {}
        
        current_config = self.config.get('semantic_frame_selection', {})
        
        # Generate recommendations with confidence weighting
        recommendations = {
            'confidence_weighted_weights': {},
            'high_confidence_changes': [],
            'experimental_changes': [],
            'cumulative_stats': {
                'videos_trained': self.cumulative_data['video_count'],
                'total_frames_analyzed': cumulative.get('total_selected_frames', 0) + cumulative.get('total_rejected_frames', 0),
                'overall_selection_ratio': cumulative.get('selection_ratio', 0.5)
            }
        }
        
        # Calculate new weights based on cumulative preferences
        differences = cumulative.get('differences', {})
        current_base = current_config.get('base_score_weight', 0.5)
        current_text = current_config.get('text_score_weight', 0.2)
        current_visual = current_config.get('visual_score_weight', 0.3)
        
        # Adjust weights based on cumulative importance and confidence
        brightness_importance = abs(differences.get('brightness', 0)) / 30
        contrast_importance = abs(differences.get('contrast', 0)) / 15
        edge_importance = abs(differences.get('edge_density', 0)) / 0.03
        text_importance = abs(differences.get('text_length', 0)) / 30
        
        # Weight adjustments by confidence
        brightness_conf = confidence.get('brightness', 0.5)
        contrast_conf = confidence.get('contrast', 0.5)
        edge_conf = confidence.get('edge_density', 0.5)
        text_conf = confidence.get('text_length', 0.5)
        
        # Calculate confidence-weighted importance
        base_importance = (brightness_importance * brightness_conf + 
                          contrast_importance * contrast_conf + 
                          edge_importance * edge_conf) / 3
        
        text_importance_weighted = text_importance * text_conf
        
        # Adjust weights
        new_base_weight = current_base + (base_importance - 0.5) * 0.3 * np.mean([brightness_conf, contrast_conf, edge_conf])
        new_text_weight = current_text + (text_importance_weighted - 0.5) * 0.4 * text_conf
        
        new_base_weight = max(0.2, min(0.8, new_base_weight))
        new_text_weight = max(0.1, min(0.6, new_text_weight))
        new_visual_weight = max(0.1, 1.0 - new_base_weight - new_text_weight)
        
        # Normalize
        total = new_base_weight + new_text_weight + new_visual_weight
        new_base_weight /= total
        new_text_weight /= total
        new_visual_weight /= total
        
        recommendations['confidence_weighted_weights'] = {
            'base_score_weight': round(new_base_weight, 3),
            'text_score_weight': round(new_text_weight, 3),
            'visual_score_weight': round(new_visual_weight, 3),
            'confidence_scores': {
                'base_metrics': round(np.mean([brightness_conf, contrast_conf, edge_conf]), 3),
                'text_metrics': round(text_conf, 3),
                'overall': round(np.mean(list(confidence.values())), 3)
            }
        }
        
        # Separate high confidence vs experimental changes
        for metric, diff in differences.items():
            conf = confidence.get(metric, 0.5)
            if conf > 0.7 and abs(diff) > 10:  # High confidence, significant difference
                recommendations['high_confidence_changes'].append({
                    'metric': metric,
                    'difference': round(diff, 2),
                    'confidence': round(conf, 3),
                    'recommendation': f"Strong evidence for {metric} preference"
                })
            elif conf > 0.4 and abs(diff) > 5:  # Medium confidence
                recommendations['experimental_changes'].append({
                    'metric': metric,
                    'difference': round(diff, 2),
                    'confidence': round(conf, 3),
                    'recommendation': f"Experimental {metric} adjustment"
                })
        
        return recommendations
    
    def _save_cumulative_data(self):
        """Save cumulative data to file."""
        with open(self.cumulative_data_file, 'w') as f:
            json.dump(self.cumulative_data, f, indent=2)
    
    def apply_cumulative_recommendations(self, apply_experimental: bool = False) -> bool:
        """Apply cumulative recommendations to config."""
        recommendations = self.generate_cumulative_recommendations()
        
        if not recommendations:
            print("❌ No cumulative recommendations available")
            return False
        
        # Backup config
        config_file = "config.local.yaml"
        backup_file = f"config.local.yaml.cumulative_backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        import shutil
        shutil.copy2(config_file, backup_file)
        print(f"💾 Backed up config to: {backup_file}")
        
        # Load current config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply confidence-weighted recommendations
        section = 'semantic_frame_selection'
        if section not in config:
            config[section] = {}
        
        weights = recommendations['confidence_weighted_weights']
        confidence_scores = weights['confidence_scores']
        
        print(f"\n🎯 Applying cumulative recommendations from {recommendations['cumulative_stats']['videos_trained']} videos:")
        print(f"📊 Total frames analyzed: {recommendations['cumulative_stats']['total_frames_analyzed']}")
        print(f"🎯 Overall confidence: {confidence_scores['overall']:.1%}")
        
        # Apply weight changes
        old_base = config[section].get('base_score_weight', 0.5)
        old_text = config[section].get('text_score_weight', 0.2)
        old_visual = config[section].get('visual_score_weight', 0.3)
        
        # Ensure values are Python floats, not NumPy types
        config[section]['base_score_weight'] = float(weights['base_score_weight'])
        config[section]['text_score_weight'] = float(weights['text_score_weight'])
        config[section]['visual_score_weight'] = float(weights['visual_score_weight'])
        
        print(f"✅ Base weight: {old_base:.3f} → {weights['base_score_weight']:.3f} (confidence: {confidence_scores['base_metrics']:.1%})")
        print(f"✅ Text weight: {old_text:.3f} → {weights['text_score_weight']:.3f} (confidence: {confidence_scores['text_metrics']:.1%})")
        print(f"✅ Visual weight: {old_visual:.3f} → {weights['visual_score_weight']:.3f}")
        
        # Apply high confidence changes
        high_conf = recommendations.get('high_confidence_changes', [])
        if high_conf:
            print(f"\n🔥 High confidence insights ({len(high_conf)} found):")
            for change in high_conf:
                print(f"  📈 {change['metric']}: {change['recommendation']} (confidence: {change['confidence']:.1%})")
        
        # Optionally apply experimental changes
        experimental = recommendations.get('experimental_changes', [])
        if experimental and apply_experimental:
            print(f"\n🧪 Experimental insights ({len(experimental)} applied):")
            for change in experimental:
                print(f"  🔬 {change['metric']}: {change['recommendation']} (confidence: {change['confidence']:.1%})")
        elif experimental:
            print(f"\n🧪 Experimental insights available ({len(experimental)} found, use --experimental to apply)")
        
        # Save updated config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"\n💡 Cumulative recommendations applied successfully!")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cumulative frame selection training")
    parser.add_argument("--add", help="Add training session from video ID")
    parser.add_argument("--apply", action="store_true", help="Apply cumulative recommendations")
    parser.add_argument("--experimental", action="store_true", help="Include experimental changes")
    parser.add_argument("--status", action="store_true", help="Show cumulative training status")
    
    args = parser.parse_args()
    
    trainer = CumulativeTrainer()
    
    if args.add:
        trainer.add_training_session(args.add)
        print(f"✅ Added training session: {args.add}")
        
    if args.apply:
        trainer.apply_cumulative_recommendations(apply_experimental=args.experimental)
        
    if args.status:
        data = trainer.cumulative_data
        print(f"📊 Cumulative Training Status:")
        print(f"  Videos trained: {data['video_count']}")
        print(f"  Last updated: {data.get('last_updated', 'Never')}")
        if data.get('cumulative_metrics'):
            metrics = data['cumulative_metrics']
            print(f"  Total frames: {metrics.get('total_selected_frames', 0) + metrics.get('total_rejected_frames', 0)}")
            print(f"  Selection ratio: {metrics.get('selection_ratio', 0):.1%}")