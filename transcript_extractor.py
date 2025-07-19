"""Transcript extraction module using Whisper and Claude API for cleanup."""

import os
import tempfile
import whisper
import ffmpeg
from typing import List, Dict, Optional
import logging
from anthropic import Anthropic
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptExtractor:
    """Handles transcript extraction from video using Whisper and cleanup with Claude."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.whisper_model = None
        self.anthropic_client = None
        
        # Initialize Anthropic client if API key is provided
        api_key = config.get('claude_api_key') or os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.anthropic_client = Anthropic(api_key=api_key)
            logger.info("Claude API client initialized for transcript cleanup")
        else:
            logger.warning("No Claude API key found. Transcript cleanup will be skipped.")
    
    def extract_transcript(self, video_path: str) -> List[Dict]:
        """Extract transcript from video using Whisper."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting transcript from: {video_path}")
        
        try:
            # Load Whisper model
            model_size = self.config.get('whisper_model', 'base')
            if self.whisper_model is None:
                logger.info(f"Loading Whisper model: {model_size}")
                self.whisper_model = whisper.load_model(model_size)
            
            # Extract audio from video
            audio_path = self._extract_audio(video_path)
            
            try:
                # Transcribe with Whisper
                logger.info("Transcribing audio with Whisper...")
                result = self.whisper_model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    verbose=False
                )
                
                if not result.get('segments'):
                    logger.warning("No transcript segments found in video")
                    return []
                
                # Convert Whisper output to our format
                segments = self._convert_whisper_segments(result['segments'])
                
                # Note: First pass transcript cleanup removed to avoid technical term errors
                
                logger.info(f"Extracted {len(segments)} transcript segments")
                return segments
                
            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
        except Exception as e:
            logger.error(f"Failed to extract transcript from {video_path}: {e}")
            raise RuntimeError(f"Transcript extraction failed: {e}") from e
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        temp_dir = tempfile.gettempdir()
        audio_filename = f"temp_audio_{os.getpid()}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(
                    audio_path,
                    acodec='pcm_s16le',
                    ac=1,  # mono
                    ar='16000'  # 16kHz sample rate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"Extracted audio to: {audio_path}")
            return audio_path
            
        except ffmpeg.Error as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def _convert_whisper_segments(self, whisper_segments: List[Dict]) -> List[Dict]:
        """Convert Whisper segments to our standard format."""
        segments = []
        
        for segment in whisper_segments:
            converted_segment = {
                'start_time': segment['start'],
                'end_time': segment['end'],
                'text': segment['text'].strip(),
                'duration': segment['end'] - segment['start'],
                'confidence': segment.get('avg_logprob', 0.0)
            }
            segments.append(converted_segment)
        
        return segments
    
    def _cleanup_transcript_with_claude(self, segments: List[Dict]) -> List[Dict]:
        """Clean up transcript segments using Claude API."""
        logger.info("Cleaning up transcript with Claude API...")
        
        # Combine all text for processing
        full_text = ' '.join(segment['text'] for segment in segments)
        
        # Prepare the prompt for Claude
        cleanup_prompt = self._get_cleanup_prompt(full_text)
        
        try:
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model=self.config.get('claude_model', 'claude-3-haiku-20240307'),
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": cleanup_prompt
                    }
                ]
            )
            
            cleaned_text = response.content[0].text.strip()
            
            # Split cleaned text back into segments
            cleaned_segments = self._redistribute_cleaned_text(segments, cleaned_text)
            
            logger.info("Transcript cleanup completed successfully")
            return cleaned_segments
            
        except Exception as e:
            logger.error(f"Error cleaning transcript with Claude: {e}")
            logger.info("Continuing with original transcript...")
            return segments
    
    def _get_cleanup_prompt(self, text: str) -> str:
        """Generate the prompt for Claude to clean up the transcript."""
        return f"""Please clean up this video transcript by fixing obvious errors from speech recognition while preserving the original meaning and structure. Only make minimal changes to:

1. Fix obvious typos and misheard words
2. Correct grammatical errors that clearly result from speech-to-text mistakes
3. Add proper punctuation where clearly missing
4. Fix capitalization issues

Do NOT:
- Change the overall meaning or tone
- Rephrase sentences significantly
- Add new content
- Remove content unless it's clearly noise/artifacts
- Change technical terms unless obviously wrong

Return only the cleaned transcript text, nothing else:

{text}"""
    
    def _redistribute_cleaned_text(self, original_segments: List[Dict], cleaned_text: str) -> List[Dict]:
        """Redistribute cleaned text back to original segment timing."""
        # Simple approach: split cleaned text by sentences and map to segments
        import re
        
        # Split cleaned text into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have fewer sentences than segments, try word-based distribution
        if len(sentences) < len(original_segments):
            words = cleaned_text.split()
            words_per_segment = max(1, len(words) // len(original_segments))
            
            cleaned_segments = []
            word_idx = 0
            
            for i, segment in enumerate(original_segments):
                start_word = word_idx
                end_word = min(word_idx + words_per_segment, len(words))
                
                # For the last segment, take all remaining words
                if i == len(original_segments) - 1:
                    end_word = len(words)
                
                segment_text = ' '.join(words[start_word:end_word])
                
                cleaned_segment = segment.copy()
                cleaned_segment['text'] = segment_text
                cleaned_segments.append(cleaned_segment)
                
                word_idx = end_word
            
            return cleaned_segments
        
        # Map sentences to segments
        cleaned_segments = []
        sentence_idx = 0
        
        for segment in original_segments:
            if sentence_idx < len(sentences):
                cleaned_segment = segment.copy()
                cleaned_segment['text'] = sentences[sentence_idx]
                sentence_idx += 1
            else:
                # If we run out of sentences, keep original text
                cleaned_segment = segment
            
            cleaned_segments.append(cleaned_segment)
        
        return cleaned_segments
    
    def save_transcript(self, segments: List[Dict], output_path: str, format: str = 'srt') -> None:
        """Save transcript to file in specified format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'srt':
            self._save_as_srt(segments, output_path)
        elif format.lower() == 'vtt':
            self._save_as_vtt(segments, output_path)
        elif format.lower() == 'txt':
            self._save_as_text(segments, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved transcript to: {output_path}")
    
    def _save_as_srt(self, segments: List[Dict], output_path: str) -> None:
        """Save transcript in SRT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment['start_time'])
                end_time = self._seconds_to_srt_time(segment['end_time'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _save_as_vtt(self, segments: List[Dict], output_path: str) -> None:
        """Save transcript in VTT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in segments:
                start_time = self._seconds_to_vtt_time(segment['start_time'])
                end_time = self._seconds_to_vtt_time(segment['end_time'])
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _save_as_text(self, segments: List[Dict], output_path: str) -> None:
        """Save transcript as plain text with timestamps."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                timestamp = self._seconds_to_timestamp(segment['start_time'])
                f.write(f"{timestamp} {segment['text']}\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to simple timestamp format (MM:SS)."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"