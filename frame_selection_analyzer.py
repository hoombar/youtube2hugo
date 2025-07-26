#!/usr/bin/env python3
"""
Frame Selection Analysis Tool

This script provides comprehensive analysis of the frame selection pipeline
to help optimize semantic frame selection and identify why frames are being filtered out.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import click

from config import Config
from video_processor import VideoProcessor
from transcript_parser import TranscriptParser
from transcript_extractor import TranscriptExtractor
from semantic_frame_selector import SemanticFrameSelector
from blog_formatter import BlogFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameSelectionAnalyzer:
    """Analyzes the entire frame selection pipeline to identify optimization opportunities."""
    
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.video_processor = VideoProcessor(config_dict)
        self.transcript_parser = TranscriptParser(config_dict)
        self.transcript_extractor = TranscriptExtractor(config_dict)
        self.semantic_frame_selector = SemanticFrameSelector(config_dict, self.video_processor)
        self.blog_formatter = BlogFormatter(config_dict)
    
    def analyze_full_pipeline(self, video_path: str, title: str, output_dir: str) -> Dict:
        """Run comprehensive analysis of the entire frame selection pipeline."""
        
        logger.info(f"🔍 Starting comprehensive frame selection analysis for: {video_path}")
        
        # Create analysis output directory
        analysis_dir = os.path.join(output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        analysis_results = {
            'video_path': video_path,
            'title': title,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.get('semantic_frame_selection', {}),
        }
        
        try:
            # Step 1: Extract transcript
            logger.info("📝 Extracting transcript...")
            transcript_segments = self.transcript_extractor.extract_transcript(video_path)
            analysis_results['transcript_info'] = {
                'total_segments': len(transcript_segments),
                'duration': max(seg.get('end', 0) for seg in transcript_segments) if transcript_segments else 0,
                'sample_segments': transcript_segments[:5]  # First 5 for inspection
            }
            
            # Step 2: Format blog content with boundary markers
            logger.info("📖 Formatting blog content with boundary analysis...")
            formatted_blog_content = self.blog_formatter.format_transcript_content(transcript_segments, title)
            boundary_map = getattr(self.blog_formatter, 'boundary_map', {})
            
            analysis_results['blog_formatting'] = {
                'sections_found': len(boundary_map),
                'boundary_map': boundary_map,
                'content_length': len(formatted_blog_content),
                'headers': self._extract_headers_from_content(formatted_blog_content)
            }
            
            # Step 3: Analyze candidate frame extraction at different intervals
            logger.info("🎬 Analyzing candidate frame extraction strategies...")
            temp_dir = os.path.join(analysis_dir, 'temp_frames')
            
            frame_extraction_analysis = self._analyze_frame_extraction_strategies(
                video_path, transcript_segments, temp_dir
            )
            analysis_results['frame_extraction'] = frame_extraction_analysis
            
            # Step 4: Analyze semantic section mapping
            logger.info("🧠 Analyzing semantic section mapping...")
            section_analysis = self._analyze_semantic_sections(
                formatted_blog_content, transcript_segments, boundary_map
            )
            analysis_results['semantic_mapping'] = section_analysis
            
            # Step 5: Analyze scoring thresholds and weights
            logger.info("⚖️ Analyzing scoring system...")
            scoring_analysis = self._analyze_scoring_system(
                video_path, transcript_segments, formatted_blog_content, temp_dir
            )
            analysis_results['scoring_analysis'] = scoring_analysis
            
            # Step 6: Compare different selection strategies
            logger.info("📊 Comparing selection strategies...")
            strategy_comparison = self._compare_selection_strategies(
                video_path, transcript_segments, formatted_blog_content, temp_dir, title
            )
            analysis_results['strategy_comparison'] = strategy_comparison
            
            # Step 7: Generate recommendations
            logger.info("💡 Generating optimization recommendations...")
            recommendations = self._generate_recommendations(analysis_results)
            analysis_results['recommendations'] = recommendations
            
            # Save detailed analysis
            analysis_file = os.path.join(analysis_dir, 'frame_selection_analysis.json')
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # Generate human-readable report
            self._generate_analysis_report(analysis_results, analysis_dir)
            
            logger.info(f"✅ Analysis complete! Results saved to: {analysis_dir}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    def _analyze_frame_extraction_strategies(self, video_path: str, transcript_segments: List[Dict], temp_dir: str) -> Dict:
        """Analyze different frame extraction strategies."""
        
        strategies = {
            'every_10s': {'interval': 10, 'method': 'temporal'},
            'every_15s': {'interval': 15, 'method': 'temporal'},
            'every_30s': {'interval': 30, 'method': 'temporal'},
            'quality_based': {'method': 'quality', 'threshold': 30.0},
            'scene_change': {'method': 'scene_change', 'threshold': 0.3}
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            logger.info(f"   Testing strategy: {strategy_name}")
            try:
                if params['method'] == 'temporal':
                    frames = self._extract_temporal_frames(video_path, params['interval'], temp_dir)
                elif params['method'] == 'quality':
                    frames = self._extract_quality_frames(video_path, params['threshold'], temp_dir)
                elif params['method'] == 'scene_change':
                    frames = self._extract_scene_change_frames(video_path, params['threshold'], temp_dir)
                
                results[strategy_name] = {
                    'frame_count': len(frames),
                    'timestamps': [f['timestamp'] for f in frames],
                    'average_quality': sum(f.get('quality_score', 0) for f in frames) / len(frames) if frames else 0,
                    'distribution': self._analyze_temporal_distribution(frames, transcript_segments)
                }
                
            except Exception as e:
                results[strategy_name] = {'error': str(e)}
        
        return results
    
    def _analyze_semantic_sections(self, formatted_content: str, transcript_segments: List[Dict], boundary_map: Dict) -> Dict:
        """Analyze how well semantic sections map to transcript content."""
        
        # Extract sections from formatted content
        import re
        header_pattern = r'^(#{1,3})\s+(.+)$'
        lines = formatted_content.split('\n')
        
        sections_analysis = []
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            if header_match:
                section_title = header_match.group(2).strip()
                
                # Analyze this section
                section_info = {
                    'title': section_title,
                    'has_boundary_marker': section_title in boundary_map,
                    'boundary_timestamp': boundary_map.get(section_title),
                    'key_terms': self._extract_key_terms(section_title),
                    'transcript_matches': self._find_transcript_matches(section_title, transcript_segments)
                }
                
                sections_analysis.append(section_info)
        
        return {
            'total_sections': len(sections_analysis),
            'sections_with_boundaries': sum(1 for s in sections_analysis if s['has_boundary_marker']),
            'sections_detail': sections_analysis,
            'boundary_coverage': len([s for s in sections_analysis if s['has_boundary_marker']]) / len(sections_analysis) if sections_analysis else 0
        }
    
    def _analyze_scoring_system(self, video_path: str, transcript_segments: List[Dict], formatted_content: str, temp_dir: str) -> Dict:
        """Analyze how the scoring system performs with different thresholds and weights."""
        
        # Test different threshold values
        thresholds_to_test = [20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 70.0]
        
        # Test different weight combinations
        weight_combinations = [
            {'base': 0.3, 'text': 0.4, 'visual': 0.3},  # Default
            {'base': 0.5, 'text': 0.2, 'visual': 0.3},  # Your optimized
            {'base': 0.4, 'text': 0.3, 'visual': 0.3},  # Balanced
            {'base': 0.2, 'text': 0.5, 'visual': 0.3},  # Text-heavy
            {'base': 0.3, 'text': 0.2, 'visual': 0.5},  # Visual-heavy
        ]
        
        scoring_results = {
            'threshold_analysis': {},
            'weight_analysis': {},
            'current_config': self.config.get('semantic_frame_selection', {})
        }
        
        # Analyze thresholds
        for threshold in thresholds_to_test:
            logger.info(f"   Testing threshold: {threshold}")
            try:
                # Temporarily modify config
                original_threshold = self.config.get('semantic_frame_selection', {}).get('score_threshold', 50.0)
                self.config.setdefault('semantic_frame_selection', {})['score_threshold'] = threshold
                
                frames = self.semantic_frame_selector.select_frames_from_blog_content(
                    video_path, transcript_segments, formatted_content, temp_dir, None, self.blog_formatter
                )
                
                scoring_results['threshold_analysis'][threshold] = {
                    'frame_count': len(frames),
                    'average_score': sum(f.get('semantic_score', 0) for f in frames) / len(frames) if frames else 0,
                    'score_distribution': self._analyze_score_distribution(frames),
                    'timestamps': [f['timestamp'] for f in frames]
                }
                
                # Restore original threshold
                self.config['semantic_frame_selection']['score_threshold'] = original_threshold
                
            except Exception as e:
                scoring_results['threshold_analysis'][threshold] = {'error': str(e)}
        
        return scoring_results
    
    def _compare_selection_strategies(self, video_path: str, transcript_segments: List[Dict], 
                                   formatted_content: str, temp_dir: str, title: str) -> Dict:
        """Compare different frame selection strategies."""
        
        strategies = {
            'current_semantic': 'Current semantic selection',
            'temporal_uniform': 'Uniform temporal selection (every 30s)',
            'quality_based': 'Quality-based selection',
            'hybrid': 'Hybrid semantic + quality'
        }
        
        comparison_results = {}
        
        for strategy_name, description in strategies.items():
            logger.info(f"   Testing strategy: {strategy_name}")
            try:
                if strategy_name == 'current_semantic':
                    frames = self.semantic_frame_selector.select_frames_from_blog_content(
                        video_path, transcript_segments, formatted_content, temp_dir, title, self.blog_formatter
                    )
                elif strategy_name == 'temporal_uniform':
                    frames = self._extract_temporal_frames(video_path, 30, temp_dir)
                elif strategy_name == 'quality_based':
                    frames = self._extract_quality_frames(video_path, 35.0, temp_dir)
                elif strategy_name == 'hybrid':
                    # Combine semantic with quality filtering
                    semantic_frames = self.semantic_frame_selector.select_frames_from_blog_content(
                        video_path, transcript_segments, formatted_content, temp_dir, title, self.blog_formatter
                    )
                    frames = [f for f in semantic_frames if f.get('quality_score', 0) > 30.0]
                
                comparison_results[strategy_name] = {
                    'description': description,
                    'frame_count': len(frames),
                    'timestamps': [f['timestamp'] for f in frames],
                    'quality_stats': self._calculate_quality_stats(frames),
                    'coverage_analysis': self._analyze_video_coverage(frames, transcript_segments)
                }
                
            except Exception as e:
                comparison_results[strategy_name] = {'error': str(e)}
        
        return comparison_results
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        
        recommendations = []
        
        # Analyze frame count issues
        semantic_frames = analysis_results.get('strategy_comparison', {}).get('current_semantic', {}).get('frame_count', 0)
        if semantic_frames < 5:
            recommendations.append(f"🔴 CRITICAL: Only {semantic_frames} frames selected. Consider lowering score threshold or adjusting weights.")
        elif semantic_frames < 10:
            recommendations.append(f"🟡 WARNING: Only {semantic_frames} frames selected. May want to increase frame count.")
        
        # Analyze boundary mapping
        boundary_coverage = analysis_results.get('semantic_mapping', {}).get('boundary_coverage', 0)
        if boundary_coverage < 0.7:
            recommendations.append(f"🔴 Poor boundary mapping: {boundary_coverage:.1%} sections mapped. Improve boundary marker system.")
        
        # Analyze scoring thresholds
        threshold_analysis = analysis_results.get('scoring_analysis', {}).get('threshold_analysis', {})
        best_threshold = None
        best_frame_count = 0
        for threshold, results in threshold_analysis.items():
            if isinstance(results, dict) and 'frame_count' in results:
                if 8 <= results['frame_count'] <= 15:  # Ideal range
                    if results['frame_count'] > best_frame_count:
                        best_threshold = threshold
                        best_frame_count = results['frame_count']
        
        if best_threshold:
            current_threshold = analysis_results.get('scoring_analysis', {}).get('current_config', {}).get('score_threshold', 50.0)
            if best_threshold != current_threshold:
                recommendations.append(f"🟢 OPTIMIZE: Change score_threshold from {current_threshold} to {best_threshold} for {best_frame_count} frames.")
        
        # Analyze extraction strategies
        extraction_results = analysis_results.get('frame_extraction', {})
        best_extraction = None
        best_extraction_count = 0
        for strategy, results in extraction_results.items():
            if isinstance(results, dict) and 'frame_count' in results:
                if results['frame_count'] > best_extraction_count:
                    best_extraction = strategy
                    best_extraction_count = results['frame_count']
        
        if best_extraction_count > semantic_frames * 2:
            recommendations.append(f"📈 INFO: {best_extraction} extracts {best_extraction_count} candidate frames vs {semantic_frames} semantic frames. Consider relaxing semantic filtering.")
        
        return recommendations
    
    def _generate_analysis_report(self, analysis_results: Dict, output_dir: str):
        """Generate a human-readable analysis report."""
        
        report_path = os.path.join(output_dir, 'analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Frame Selection Analysis Report\n\n")
            f.write(f"**Video:** {analysis_results['video_path']}\n")
            f.write(f"**Title:** {analysis_results['title']}\n")
            f.write(f"**Analysis Time:** {analysis_results['timestamp']}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            semantic_frames = analysis_results.get('strategy_comparison', {}).get('current_semantic', {}).get('frame_count', 0)
            f.write(f"- **Current Selection:** {semantic_frames} frames\n")
            
            boundary_coverage = analysis_results.get('semantic_mapping', {}).get('boundary_coverage', 0)
            f.write(f"- **Boundary Mapping:** {boundary_coverage:.1%} of sections mapped\n")
            
            sections_count = analysis_results.get('blog_formatting', {}).get('sections_found', 0)
            f.write(f"- **Blog Sections:** {sections_count} sections identified\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in analysis_results.get('recommendations', []):
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # Detailed Analysis
            f.write("## Strategy Comparison\n\n")
            f.write("| Strategy | Frame Count | Description |\n")
            f.write("|----------|-------------|-------------|\n")
            
            strategy_comparison = analysis_results.get('strategy_comparison', {})
            for strategy, data in strategy_comparison.items():
                if isinstance(data, dict) and 'frame_count' in data:
                    f.write(f"| {strategy} | {data['frame_count']} | {data.get('description', '')} |\n")
            f.write("\n")
            
            # Threshold Analysis
            f.write("## Threshold Analysis\n\n")
            f.write("| Threshold | Frame Count | Avg Score |\n")
            f.write("|-----------|-------------|----------|\n")
            
            threshold_analysis = analysis_results.get('scoring_analysis', {}).get('threshold_analysis', {})
            for threshold, data in threshold_analysis.items():
                if isinstance(data, dict) and 'frame_count' in data:
                    f.write(f"| {threshold} | {data['frame_count']} | {data.get('average_score', 0):.1f} |\n")
            f.write("\n")
            
            # Section Mapping Details
            f.write("## Section Mapping Analysis\n\n")
            sections_detail = analysis_results.get('semantic_mapping', {}).get('sections_detail', [])
            for section in sections_detail:
                f.write(f"### {section['title']}\n")
                f.write(f"- **Boundary Marker:** {'✅' if section['has_boundary_marker'] else '❌'}\n")
                if section['boundary_timestamp']:
                    f.write(f"- **Timestamp:** {section['boundary_timestamp']:.1f}s\n")
                f.write(f"- **Key Terms:** {', '.join(section['key_terms'])}\n")
                f.write(f"- **Transcript Matches:** {section['transcript_matches']}\n\n")
        
        logger.info(f"📄 Analysis report generated: {report_path}")
    
    # Helper methods
    def _extract_headers_from_content(self, content: str) -> List[str]:
        """Extract headers from markdown content."""
        import re
        header_pattern = r'^(#{1,3})\s+(.+)$'
        headers = []
        for line in content.split('\n'):
            match = re.match(header_pattern, line)
            if match:
                headers.append(match.group(2).strip())
        return headers
    
    def _extract_temporal_frames(self, video_path: str, interval: int, temp_dir: str) -> List[Dict]:
        """Extract frames at regular temporal intervals."""
        # This would call video processor methods - simplified for now
        return []
    
    def _extract_quality_frames(self, video_path: str, threshold: float, temp_dir: str) -> List[Dict]:
        """Extract frames based on quality metrics."""
        # This would call video processor methods - simplified for now
        return []
    
    def _extract_scene_change_frames(self, video_path: str, threshold: float, temp_dir: str) -> List[Dict]:
        """Extract frames at scene changes."""
        # This would call video processor methods - simplified for now
        return []
    
    def _analyze_temporal_distribution(self, frames: List[Dict], transcript_segments: List[Dict]) -> Dict:
        """Analyze how frames are distributed across the video timeline."""
        if not frames or not transcript_segments:
            return {}
        
        total_duration = max(seg.get('end', 0) for seg in transcript_segments)
        frame_timestamps = [f['timestamp'] for f in frames]
        
        return {
            'total_duration': total_duration,
            'frame_density': len(frames) / (total_duration / 60),  # frames per minute
            'coverage_gaps': self._find_coverage_gaps(frame_timestamps, total_duration),
            'distribution_evenness': self._calculate_distribution_evenness(frame_timestamps, total_duration)
        }
    
    def _extract_key_terms(self, section_title: str) -> List[str]:
        """Extract key terms from section title."""
        import re
        words = re.findall(r'\b\w+\b', section_title.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _find_transcript_matches(self, section_title: str, transcript_segments: List[Dict]) -> int:
        """Find how many transcript segments contain terms from the section title."""
        key_terms = self._extract_key_terms(section_title)
        matches = 0
        
        for segment in transcript_segments:
            text = segment.get('text', '').lower()
            if any(term in text for term in key_terms):
                matches += 1
        
        return matches
    
    def _analyze_score_distribution(self, frames: List[Dict]) -> Dict:
        """Analyze the distribution of semantic scores."""
        if not frames:
            return {}
        
        scores = [f.get('semantic_score', 0) for f in frames]
        return {
            'min': min(scores),
            'max': max(scores),
            'avg': sum(scores) / len(scores),
            'count_above_threshold': len([s for s in scores if s >= 45.0])
        }
    
    def _calculate_quality_stats(self, frames: List[Dict]) -> Dict:
        """Calculate quality statistics for frames."""
        if not frames:
            return {}
        
        quality_scores = [f.get('quality_score', 0) for f in frames if 'quality_score' in f]
        if not quality_scores:
            return {'no_quality_data': True}
        
        return {
            'avg_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'quality_variance': self._calculate_variance(quality_scores)
        }
    
    def _analyze_video_coverage(self, frames: List[Dict], transcript_segments: List[Dict]) -> Dict:
        """Analyze how well frames cover the video timeline."""
        if not frames or not transcript_segments:
            return {}
        
        total_duration = max(seg.get('end', 0) for seg in transcript_segments)
        frame_timestamps = sorted([f['timestamp'] for f in frames])
        
        # Calculate gaps between frames
        gaps = []
        for i in range(len(frame_timestamps) - 1):
            gap = frame_timestamps[i + 1] - frame_timestamps[i]
            gaps.append(gap)
        
        return {
            'total_duration': total_duration,
            'first_frame': frame_timestamps[0] if frame_timestamps else 0,
            'last_frame': frame_timestamps[-1] if frame_timestamps else 0,
            'coverage_percentage': ((frame_timestamps[-1] - frame_timestamps[0]) / total_duration * 100) if frame_timestamps and total_duration > 0 else 0,
            'average_gap': sum(gaps) / len(gaps) if gaps else 0,
            'largest_gap': max(gaps) if gaps else 0
        }
    
    def _find_coverage_gaps(self, timestamps: List[float], total_duration: float) -> List[Dict]:
        """Find large gaps in frame coverage."""
        gaps = []
        sorted_times = sorted(timestamps)
        
        for i in range(len(sorted_times) - 1):
            gap_size = sorted_times[i + 1] - sorted_times[i]
            if gap_size > 60:  # Gaps larger than 1 minute
                gaps.append({
                    'start': sorted_times[i],
                    'end': sorted_times[i + 1],
                    'duration': gap_size
                })
        
        return gaps
    
    def _calculate_distribution_evenness(self, timestamps: List[float], total_duration: float) -> float:
        """Calculate how evenly distributed frames are (0 = perfectly even, 1 = very uneven)."""
        if len(timestamps) < 2:
            return 0
        
        expected_interval = total_duration / len(timestamps)
        sorted_times = sorted(timestamps)
        
        variance = 0
        for i in range(len(sorted_times) - 1):
            actual_interval = sorted_times[i + 1] - sorted_times[i]
            variance += (actual_interval - expected_interval) ** 2
        
        return min(1.0, variance / (len(sorted_times) * expected_interval ** 2))
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


@click.command()
@click.option('--video', '-v', required=True, help='Path to video file')
@click.option('--title', required=True, help='Video title for analysis')
@click.option('--output', '-o', default='./analysis_output', help='Output directory for analysis results')
@click.option('--config', '-c', help='Path to configuration file')
def analyze(video, title, output, config):
    """Analyze frame selection pipeline to optimize semantic selection."""
    
    # Load configuration
    config_dict = Config.get_default_config()
    
    if config and os.path.exists(config):
        custom_config = Config.load_from_file(config)
        config_dict.update(custom_config)
    
    # Check if we have Gemini API key
    if not config_dict.get('gemini_api_key') and not os.environ.get('GOOGLE_API_KEY'):
        click.echo("❌ Error: No Gemini API key found. Set GOOGLE_API_KEY env var or configure in config.local.yaml", err=True)
        sys.exit(1)
    
    if not os.path.exists(video):
        click.echo(f"❌ Error: Video file not found: {video}", err=True)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Run analysis
    analyzer = FrameSelectionAnalyzer(config_dict)
    
    try:
        results = analyzer.analyze_full_pipeline(video, title, output)
        
        click.echo(f"✅ Analysis complete!")
        click.echo(f"📊 Results saved to: {output}")
        
        # Print key insights
        semantic_frames = results.get('strategy_comparison', {}).get('current_semantic', {}).get('frame_count', 0)
        click.echo(f"🎯 Current semantic selection: {semantic_frames} frames")
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            click.echo("\n💡 Key Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                click.echo(f"   {rec}")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    analyze()