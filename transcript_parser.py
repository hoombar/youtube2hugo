"""Transcript parsing module for multiple formats."""

import pysrt
import webvtt
import re
import os
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptParser:
    """Handles parsing of various transcript formats."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def parse_transcript(self, transcript_path: str) -> List[Dict]:
        """Parse transcript file and return list of segments with timestamps."""
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        file_ext = os.path.splitext(transcript_path)[1].lower()
        
        if file_ext == '.srt':
            return self._parse_srt(transcript_path)
        elif file_ext == '.vtt':
            return self._parse_vtt(transcript_path)
        elif file_ext in ['.txt', '.md']:
            return self._parse_text_with_timestamps(transcript_path)
        else:
            raise ValueError(f"Unsupported transcript format: {file_ext}")
    
    def _parse_srt(self, srt_path: str) -> List[Dict]:
        """Parse SRT subtitle file."""
        segments = []
        
        try:
            subs = pysrt.open(srt_path)
            
            for sub in subs:
                start_time = self._timedelta_to_seconds(sub.start.to_time())
                end_time = self._timedelta_to_seconds(sub.end.to_time())
                
                segment = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': self._clean_text(sub.text),
                    'duration': end_time - start_time
                }
                segments.append(segment)
                
        except Exception as e:
            logger.error(f"Error parsing SRT file: {e}")
            raise
            
        logger.info(f"Parsed {len(segments)} SRT segments")
        return segments
    
    def _parse_vtt(self, vtt_path: str) -> List[Dict]:
        """Parse WebVTT subtitle file."""
        segments = []
        
        try:
            for caption in webvtt.read(vtt_path):
                start_time = self._time_to_seconds(caption.start)
                end_time = self._time_to_seconds(caption.end)
                
                segment = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': self._clean_text(caption.text),
                    'duration': end_time - start_time
                }
                segments.append(segment)
                
        except Exception as e:
            logger.error(f"Error parsing VTT file: {e}")
            raise
            
        logger.info(f"Parsed {len(segments)} VTT segments")
        return segments
    
    def _parse_text_with_timestamps(self, text_path: str) -> List[Dict]:
        """Parse plain text file with timestamps."""
        segments = []
        
        # Patterns for different timestamp formats
        patterns = [
            r'(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\s*:?\s*(.*?)(?=\d{1,2}:\d{2}:\d{2}|$)',
            r'\[(\d{1,2}:\d{2}:\d{2})\]\s*(.*?)(?=\[\d{1,2}:\d{2}:\d{2}\]|$)',
            r'(\d{1,2}:\d{2}:\d{2})\s+(.*?)(?=\d{1,2}:\d{2}:\d{2}|$)'
        ]
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try each pattern
            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
                if matches:
                    segments = self._process_text_matches(matches, pattern)
                    break
            
            if not segments:
                # Fallback: treat as plain text without timestamps
                segments = [{'start_time': 0, 'end_time': 0, 'text': content, 'duration': 0}]
                
        except Exception as e:
            logger.error(f"Error parsing text file: {e}")
            raise
            
        logger.info(f"Parsed {len(segments)} text segments")
        return segments
    
    def _process_text_matches(self, matches: List, pattern: str) -> List[Dict]:
        """Process regex matches into transcript segments."""
        segments = []
        
        for match in matches:
            if len(match) == 3:  # start_time, end_time, text
                start_time = self._time_to_seconds(match[0])
                end_time = self._time_to_seconds(match[1])
                text = self._clean_text(match[2])
            elif len(match) == 2:  # timestamp, text
                start_time = self._time_to_seconds(match[0])
                end_time = start_time + 30  # Default 30-second segments
                text = self._clean_text(match[1])
            else:
                continue
                
            if text.strip():
                segment = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text,
                    'duration': end_time - start_time
                }
                segments.append(segment)
                
        return segments
    
    def find_transcript_context(self, timestamp: float, segments: List[Dict]) -> Optional[Dict]:
        """Find transcript segment that matches or is closest to given timestamp."""
        context_window = self.config.get('context_window', 30)
        
        # Find segments within context window
        matching_segments = []
        for segment in segments:
            if (timestamp >= segment['start_time'] - context_window and 
                timestamp <= segment['end_time'] + context_window):
                matching_segments.append(segment)
        
        if not matching_segments:
            return None
        
        # Return the segment closest to the timestamp
        closest_segment = min(
            matching_segments,
            key=lambda s: min(
                abs(timestamp - s['start_time']),
                abs(timestamp - s['end_time'])
            )
        )
        
        return closest_segment
    
    def get_context_around_timestamp(self, timestamp: float, segments: List[Dict]) -> str:
        """Get transcript context around a specific timestamp."""
        context_window = self.config.get('context_window', 30)
        
        # Find segments within the context window
        context_segments = []
        for segment in segments:
            if (segment['start_time'] <= timestamp + context_window and
                segment['end_time'] >= timestamp - context_window):
                context_segments.append(segment)
        
        # Sort by start time
        context_segments.sort(key=lambda x: x['start_time'])
        
        # Combine text from context segments
        context_text = ' '.join(seg['text'] for seg in context_segments)
        return self._clean_text(context_text)
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS or MM:SS) to seconds."""
        parts = time_str.split(':')
        
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return float(parts[0])
    
    def _timedelta_to_seconds(self, td) -> float:
        """Convert timedelta to seconds."""
        return td.total_seconds()
    
    def _clean_text(self, text: str) -> str:
        """Clean transcript text by removing formatting and extra whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove speaker indicators like "[Speaker 1]" or ">> "
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'^>+\s*', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        return text.strip()